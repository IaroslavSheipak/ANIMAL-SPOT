"""
Module: convlstm_encoder.py
ConvLSTM-based encoder for temporal modeling in bioacoustics.

This architecture combines:
1. CNN backbone for spatial feature extraction (ResNet/ConvNeXt)
2. LSTM layers for temporal sequence modeling
3. Attention mechanism for focusing on relevant time steps

Advantages for killer whale call classification:
- Captures temporal dependencies within calls
- Models syllable sequences (important for call types like K21 vs K7)
- Handles variable-length inputs better than pure CNNs

Authors: CETACEANS Project
Last Access: December 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from models.residual_encoder import ResidualEncoder
from models.convnext_encoder import ConvNextEncoder


# Default options for ConvLSTM encoder
DefaultConvLSTMEncoderOpts = {
    "backbone": "resnet",  # "resnet" or "convnext"
    "backbone_pretrained": False,
    "resnet_size": 18,
    "lstm_hidden_size": 256,
    "lstm_num_layers": 2,
    "lstm_dropout": 0.3,
    "lstm_bidirectional": True,
    "use_attention": True,
    "attention_heads": 4,
    "freeze_backbone": False,  # Freeze CNN backbone, train only LSTM
}


class TemporalAttention(nn.Module):
    """
    Multi-head self-attention for temporal sequences.
    Helps the model focus on the most relevant time steps.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            Attended features: (batch, seq_len, embed_dim)
        """
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        return self.layer_norm(x + attn_out)


class ConvLSTMEncoder(nn.Module):
    """
    CNN + LSTM encoder for temporal modeling.

    Architecture:
    1. CNN backbone processes spectrograms to extract spatial features
    2. Features are reshaped to temporal sequence
    3. LSTM processes the sequence to capture temporal patterns
    4. Optional attention layer focuses on important time steps
    5. Global pooling produces final representation

    This is particularly useful for:
    - Long calls with internal structure (syllables)
    - Distinguishing calls that share initial syllables but differ later
    - Modeling temporal context for better classification
    """

    def __init__(self, opts: Dict[str, Any] = None):
        super().__init__()

        if opts is None:
            opts = DefaultConvLSTMEncoderOpts.copy()

        self.opts = opts

        # Create backbone
        self.backbone_type = opts.get("backbone", "resnet")
        if self.backbone_type == "convnext":
            backbone_opts = {"pretrained": opts.get("backbone_pretrained", False)}
            self.backbone = ConvNextEncoder(backbone_opts)
            self.cnn_out_channels = 768  # ConvNeXt Tiny output
        else:
            backbone_opts = {
                "resnet_size": opts.get("resnet_size", 18),
                "pretrained": opts.get("backbone_pretrained", False)
            }
            self.backbone = ResidualEncoder(backbone_opts)
            # ResNet output channels depend on block type
            if opts.get("resnet_size", 18) in [18, 34]:
                self.cnn_out_channels = 512
            else:
                self.cnn_out_channels = 2048

        # Optionally freeze backbone
        if opts.get("freeze_backbone", False):
            for param in self.backbone.parameters():
                param.requires_grad = False

        # LSTM configuration
        self.lstm_hidden_size = opts.get("lstm_hidden_size", 256)
        self.lstm_num_layers = opts.get("lstm_num_layers", 2)
        self.bidirectional = opts.get("lstm_bidirectional", True)
        lstm_dropout = opts.get("lstm_dropout", 0.3) if self.lstm_num_layers > 1 else 0

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_channels,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=self.bidirectional
        )

        # Calculate LSTM output size
        lstm_out_size = self.lstm_hidden_size * (2 if self.bidirectional else 1)

        # Optional attention layer
        self.use_attention = opts.get("use_attention", True)
        if self.use_attention:
            self.attention = TemporalAttention(
                embed_dim=lstm_out_size,
                num_heads=opts.get("attention_heads", 4),
                dropout=opts.get("lstm_dropout", 0.3)
            )

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(lstm_out_size)

        # Output projection (to match expected classifier input)
        self.output_channels = lstm_out_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input spectrogram (batch, 1, freq_bins, time_steps)

        Returns:
            Encoded features (batch, output_channels, 1, 1)
            Shape compatible with standard classifier
        """
        batch_size = x.size(0)

        # CNN backbone: extract spatial features
        # Output: (batch, channels, h, w)
        cnn_features = self.backbone(x)

        # Get spatial dimensions
        _, channels, h, w = cnn_features.shape

        # Reshape for LSTM: treat width (time) as sequence
        # Average pool over height (frequency)
        # (batch, channels, h, w) -> (batch, channels, w)
        features = F.adaptive_avg_pool2d(cnn_features, (1, w)).squeeze(2)

        # Transpose to (batch, seq_len, features)
        features = features.permute(0, 2, 1)  # (batch, w, channels)

        # LSTM: temporal modeling
        lstm_out, (h_n, c_n) = self.lstm(features)
        # lstm_out: (batch, seq_len, lstm_hidden * num_directions)

        # Optional attention
        if self.use_attention:
            lstm_out = self.attention(lstm_out)

        # Layer normalization
        lstm_out = self.layer_norm(lstm_out)

        # Global pooling options:
        # 1. Mean pooling (default): captures overall representation
        # 2. Last hidden state: captures final temporal state
        # 3. Max pooling: captures strongest activations

        # Using mean pooling
        output = lstm_out.mean(dim=1)  # (batch, lstm_hidden * num_directions)

        # Reshape to match classifier input format: (batch, channels, 1, 1)
        output = output.view(batch_size, -1, 1, 1)

        return output

    def get_output_channels(self) -> int:
        """Return the number of output channels for the classifier."""
        return self.output_channels


class ConvLSTMWithContext(nn.Module):
    """
    Extended ConvLSTM that processes multiple spectrogram windows
    and models their relationships.

    This is useful for:
    - Processing longer audio segments
    - Modeling context around detected calls
    - Improving classification by considering surrounding audio
    """

    def __init__(
        self,
        opts: Dict[str, Any] = None,
        num_windows: int = 3,
        window_overlap: float = 0.5
    ):
        super().__init__()

        self.num_windows = num_windows
        self.window_overlap = window_overlap

        # Per-window encoder
        self.window_encoder = ConvLSTMEncoder(opts)

        # Cross-window LSTM
        lstm_out_size = self.window_encoder.output_channels
        self.context_lstm = nn.LSTM(
            input_size=lstm_out_size,
            hidden_size=lstm_out_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.output_channels = lstm_out_size

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """
        Args:
            windows: (batch, num_windows, 1, freq_bins, time_steps)

        Returns:
            Context-aware features: (batch, output_channels, 1, 1)
        """
        batch_size, num_windows = windows.shape[:2]

        # Process each window
        window_features = []
        for i in range(num_windows):
            feat = self.window_encoder(windows[:, i])  # (batch, channels, 1, 1)
            window_features.append(feat.squeeze(-1).squeeze(-1))

        # Stack: (batch, num_windows, channels)
        features = torch.stack(window_features, dim=1)

        # Cross-window context modeling
        context_out, _ = self.context_lstm(features)

        # Take center window with context
        center_idx = num_windows // 2
        output = context_out[:, center_idx]  # (batch, channels)

        return output.view(batch_size, -1, 1, 1)


# Factory function for easy creation
def create_convlstm_encoder(
    backbone: str = "resnet",
    pretrained: bool = False,
    lstm_hidden: int = 256,
    lstm_layers: int = 2,
    bidirectional: bool = True,
    use_attention: bool = True,
    **kwargs
) -> ConvLSTMEncoder:
    """
    Factory function to create ConvLSTM encoder.

    Args:
        backbone: "resnet" or "convnext"
        pretrained: Use pretrained backbone weights
        lstm_hidden: LSTM hidden size
        lstm_layers: Number of LSTM layers
        bidirectional: Use bidirectional LSTM
        use_attention: Add temporal attention
        **kwargs: Additional options

    Returns:
        Configured ConvLSTMEncoder
    """
    opts = {
        "backbone": backbone,
        "backbone_pretrained": pretrained,
        "lstm_hidden_size": lstm_hidden,
        "lstm_num_layers": lstm_layers,
        "lstm_bidirectional": bidirectional,
        "use_attention": use_attention,
        **kwargs
    }
    return ConvLSTMEncoder(opts)


if __name__ == "__main__":
    # Test the encoder
    print("Testing ConvLSTM Encoder...")

    # Create encoder
    encoder = create_convlstm_encoder(
        backbone="resnet",
        lstm_hidden=256,
        lstm_layers=2,
        bidirectional=True,
        use_attention=True
    )

    print(f"Output channels: {encoder.get_output_channels()}")

    # Test forward pass
    batch_size = 4
    freq_bins = 256
    time_steps = 128

    x = torch.randn(batch_size, 1, freq_bins, time_steps)
    output = encoder(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("Test passed!")
