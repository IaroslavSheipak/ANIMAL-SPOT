"""
Module: convnext_encoder.py
Authors: Christian Bergler, Hendrik Schroeter (adapted for ConvNeXt)
GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 23.10.2025
"""

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny

DefaultEncoderOpts = {
    "input_channels": 1,
    "pretrained": False,  # Set to True to enable ImageNet pre-training
}

"""
Defines the convolutional or feature extraction part using ConvNeXt Tiny architecture.
The ConvNeXt Tiny model is adapted to work with single-channel spectrograms instead of 3-channel RGB images.
"""
class ConvNextEncoder(nn.Module):
    def __init__(self, opts: dict = DefaultEncoderOpts):
        super().__init__()
        self._opts = opts

        # Load ConvNeXt Tiny model
        if opts.get("pretrained", False):
            self.backbone = convnext_tiny(weights="DEFAULT")
        else:
            self.backbone = convnext_tiny(weights=None)

        # Modify the first convolution layer to accept single-channel input (spectrograms)
        # Original: Conv2d(3, 96, kernel_size=(4, 4), stride=(4, 4))
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            opts["input_channels"],
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False if original_conv.bias is None else True
        )

        # Initialize the new conv layer
        if opts["input_channels"] == 1:
            # Average the weights across the 3 input channels for single channel initialization
            with torch.no_grad():
                self.backbone.features[0][0].weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)

        # Use only the feature extraction part (without the classifier head)
        self.features = self.backbone.features

        # Output channels for ConvNeXt Tiny is 768
        self.output_channels = 768

    def forward(self, x):
        x = self.features(x)
        return x

    def model_opts(self):
        return self._opts
