#!/usr/bin/env python3
"""
Unified Training Experiment Script for ANIMAL-SPOT
===================================================
This script consolidates comprehensive_training.py, k12_k14_optimization.py,
and final_convnext_experiment.py into a single configurable training system.

Usage:
    # Run with custom config
    python training_experiment.py --data_dir /path/to/data --output_dir /path/to/output

    # Run predefined experiment
    python training_experiment.py --data_dir /path/to/data --preset best_config

    # List available presets
    python training_experiment.py --list_presets

Author: Iaroslav
Date: 2024-12
"""

import os
import sys
import json
import argparse
import logging
import time
import math
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    import librosa
    import soundfile as sf
    from scipy.ndimage import zoom
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False
    print("Warning: librosa/soundfile not available. Install with: pip install librosa soundfile")

# CUDA optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


@dataclass
class ExperimentConfig:
    """Unified configuration for all experiment types."""
    name: str = "experiment"

    # Model architecture
    backbone: str = "resnet18"  # resnet18, resnet34, convnext_tiny
    pretrained: bool = False

    # Training hyperparameters
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 100
    early_stop_patience: int = 25
    grad_clip: float = 1.0

    # Audio processing
    sr: int = 44100
    sequence_len: int = 1000  # ms
    n_fft: int = 1024
    hop_length: int = 172
    n_freq_bins: int = 256
    fmin: int = 500
    fmax: int = 10000
    freq_compression: str = "linear"

    # Loss function
    loss_type: str = "label_smoothing"  # cross_entropy, label_smoothing, focal, weighted
    label_smoothing: float = 0.1
    focal_gamma: float = 2.0
    class_weights: Optional[Dict[str, float]] = None

    # Sampling strategy
    weighted_sampling: bool = True
    sampling_strategy: str = "sqrt_inverse_freq"  # inverse_freq, sqrt_inverse_freq, uniform
    rare_class_boost: float = 1.5  # Extra boost for rare classes
    rare_classes: List[str] = field(default_factory=lambda: ["K12", "K14"])

    # Scheduler
    scheduler: str = "onecycle"  # onecycle, plateau

    # Optimization
    use_compile: bool = True
    mixed_precision: bool = True

    # Data loading
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 4
    persistent_workers: bool = True
    use_cache: bool = True

    # Data splits
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42


# =============================================================================
# Predefined Experiment Presets
# =============================================================================

def get_presets() -> Dict[str, ExperimentConfig]:
    """Return predefined experiment configurations."""
    presets = {}

    # Best overall configuration (99.49% accuracy)
    presets["best_config"] = ExperimentConfig(
        name="best_config",
        backbone="resnet18",
        pretrained=False,
        batch_size=64,
        lr=3e-4,
        weight_decay=1e-4,
        sequence_len=1000,
        loss_type="label_smoothing",
        label_smoothing=0.1,
        weighted_sampling=True,
        sampling_strategy="sqrt_inverse_freq",
        rare_class_boost=1.5,
    )

    # ConvNeXt experiment
    presets["convnext"] = ExperimentConfig(
        name="convnext_experiment",
        backbone="convnext_tiny",
        pretrained=False,
        batch_size=32,
        lr=3e-4,
        weight_decay=0.01,
        sequence_len=1000,
        loss_type="label_smoothing",
        label_smoothing=0.1,
        weighted_sampling=True,
        sampling_strategy="sqrt_inverse_freq",
        rare_class_boost=1.5,
    )

    # Fast baseline for testing
    presets["fast_baseline"] = ExperimentConfig(
        name="fast_baseline",
        backbone="resnet18",
        pretrained=False,
        batch_size=64,
        lr=3e-4,
        max_epochs=20,
        early_stop_patience=10,
        sequence_len=500,
        loss_type="cross_entropy",
        weighted_sampling=False,
    )

    # Aggressive rare class focus
    presets["rare_class_focus"] = ExperimentConfig(
        name="rare_class_focus",
        backbone="resnet18",
        pretrained=False,
        batch_size=64,
        lr=3e-4,
        sequence_len=800,
        loss_type="label_smoothing",
        label_smoothing=0.1,
        weighted_sampling=True,
        sampling_strategy="inverse_freq",
        rare_class_boost=2.0,
        class_weights={"K12": 2.0, "K14": 3.0},
    )

    return presets


# =============================================================================
# Data Utilities
# =============================================================================

class SpectrogramCache:
    """Disk-based spectrogram cache for faster training."""

    def __init__(self, cache_dir: Path, config: ExperimentConfig):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        config_str = f"{config.sr}_{config.n_fft}_{config.hop_length}_{config.n_freq_bins}_{config.fmin}_{config.fmax}_{config.freq_compression}_{config.sequence_len}"
        self.config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

    def get_cache_path(self, audio_path: str) -> Path:
        return self.cache_dir / f"{Path(audio_path).stem}_{self.config_hash}.npy"

    def get(self, audio_path: str) -> Optional[np.ndarray]:
        cache_path = self.get_cache_path(audio_path)
        if cache_path.exists():
            try:
                return np.load(cache_path)
            except:
                return None
        return None

    def put(self, audio_path: str, spectrogram: np.ndarray):
        try:
            np.save(self.get_cache_path(audio_path), spectrogram)
        except:
            pass


class AudioDataset(TorchDataset):
    """Audio dataset with spectrogram generation."""

    def __init__(self, file_list: List[str], config: ExperimentConfig,
                 classes: List[str], cache: Optional[SpectrogramCache] = None,
                 augment: bool = False):
        if not HAS_AUDIO_LIBS:
            raise ImportError("librosa and soundfile required. Install with: pip install librosa soundfile")

        self.file_list = file_list
        self.config = config
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.cache = cache
        self.augment = augment
        self.seq_samples = int(config.sequence_len * config.sr / 1000)

    def __len__(self):
        return len(self.file_list)

    def get_class(self, filepath: str) -> int:
        filename = Path(filepath).name
        class_name = filename.split("-", 1)[0]
        return self.class_to_idx.get(class_name, 0)

    def load_audio(self, filepath: str) -> np.ndarray:
        try:
            audio, sr = sf.read(filepath)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            if sr != self.config.sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.sr)

            if len(audio) < self.seq_samples:
                audio = np.pad(audio, (0, self.seq_samples - len(audio)))
            else:
                if self.augment:
                    start = np.random.randint(0, len(audio) - self.seq_samples + 1)
                else:
                    start = (len(audio) - self.seq_samples) // 2
                audio = audio[start:start + self.seq_samples]

            return audio.astype(np.float32)
        except:
            return np.zeros(self.seq_samples, dtype=np.float32)

    def compute_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        stft = librosa.stft(audio, n_fft=self.config.n_fft, hop_length=self.config.hop_length)
        magnitude = np.abs(stft)

        freqs = librosa.fft_frequencies(sr=self.config.sr, n_fft=self.config.n_fft)
        freq_mask = (freqs >= self.config.fmin) & (freqs <= self.config.fmax)
        magnitude = magnitude[freq_mask, :]

        if magnitude.shape[0] != self.config.n_freq_bins:
            zoom_factor = self.config.n_freq_bins / magnitude.shape[0]
            magnitude = zoom(magnitude, (zoom_factor, 1), order=1)

        magnitude = librosa.amplitude_to_db(magnitude, ref=np.max)
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)

        return magnitude.astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath = self.file_list[idx]
        label = self.get_class(filepath)

        spec = None
        if self.cache is not None and not self.augment:
            spec = self.cache.get(filepath)

        if spec is None:
            audio = self.load_audio(filepath)
            spec = self.compute_spectrogram(audio)
            if self.cache is not None and not self.augment:
                self.cache.put(filepath, spec)

        spec = torch.from_numpy(spec).unsqueeze(0)
        return spec, label


# =============================================================================
# Loss Functions
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class WeightedLoss(nn.Module):
    """Cross-entropy with per-sample weighting for specific classes."""

    def __init__(self, class_to_idx: Dict[str, int], class_weights: Dict[str, float],
                 label_smoothing: float = 0.0):
        super().__init__()
        self.class_to_idx = class_to_idx
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

        # Build weight tensor
        self.weight_map = {}
        for class_name, weight in class_weights.items():
            if class_name in class_to_idx:
                self.weight_map[class_to_idx[class_name]] = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = torch.ones_like(targets, dtype=torch.float32)
        for idx, weight in self.weight_map.items():
            weights[targets == idx] = weight

        ce_loss = F.cross_entropy(inputs, targets, reduction='none',
                                   label_smoothing=self.label_smoothing)
        return (ce_loss * weights).mean()


# =============================================================================
# Model Creation
# =============================================================================

def create_model(config: ExperimentConfig, num_classes: int, device: str) -> nn.Module:
    """Create model based on configuration."""
    import torchvision.models as models

    if config.backbone == "resnet18":
        model = models.resnet18(weights='IMAGENET1K_V1' if config.pretrained else None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif config.backbone == "resnet34":
        model = models.resnet34(weights='IMAGENET1K_V1' if config.pretrained else None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif config.backbone == "convnext_tiny":
        model = models.convnext_tiny(weights='IMAGENET1K_V1' if config.pretrained else None)
        original_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            1, 96, kernel_size=original_conv.kernel_size,
            stride=original_conv.stride, padding=original_conv.padding
        )
        if config.pretrained:
            with torch.no_grad():
                model.features[0][0].weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    else:
        raise ValueError(f"Unknown backbone: {config.backbone}")

    model = model.to(device)

    if config.use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            logging.warning(f"Model compilation failed: {e}")

    return model


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    """Unified trainer for all experiment types."""

    def __init__(self, config: ExperimentConfig, data_dir: Path, output_dir: Path,
                 device: str = "cuda"):
        self.config = config
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s|%(levelname)s|%(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "training.log"),
                logging.StreamHandler()
            ],
            force=True
        )
        self.logger = logging.getLogger()

        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.best_rare_class_acc = {}
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def prepare_data(self):
        """Prepare datasets and dataloaders."""
        # Find audio files
        audio_files = []
        for ext in ['*.wav', '*.WAV']:
            audio_files.extend(self.data_dir.rglob(ext))
        audio_files = [str(f) for f in audio_files]

        # Extract classes
        classes = set()
        for f in audio_files:
            class_name = Path(f).name.split("-", 1)[0]
            classes.add(class_name)
        self.classes = sorted(list(classes))
        self.num_classes = len(self.classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.logger.info(f"Found {len(audio_files)} audio files with {self.num_classes} classes")
        self.logger.info(f"Classes: {self.classes}")

        # Split by tape (ensures same tape stays in same split)
        tape_to_files = {}
        for f in audio_files:
            parts = Path(f).stem.split("_")
            tape = parts[3] if len(parts) >= 4 else "unknown"
            if tape not in tape_to_files:
                tape_to_files[tape] = []
            tape_to_files[tape].append(f)

        tapes = list(tape_to_files.keys())
        np.random.seed(self.config.random_seed)
        np.random.shuffle(tapes)

        n_tapes = len(tapes)
        train_end = int(self.config.train_split * n_tapes)
        val_end = int((self.config.train_split + self.config.val_split) * n_tapes)

        train_files, val_files, test_files = [], [], []
        for i, tape in enumerate(tapes):
            if i < train_end:
                train_files.extend(tape_to_files[tape])
            elif i < val_end:
                val_files.extend(tape_to_files[tape])
            else:
                test_files.extend(tape_to_files[tape])

        self.logger.info(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

        # Count samples per class
        self.samples_per_class = {c: 0 for c in self.classes}
        for f in train_files:
            class_name = Path(f).name.split("-", 1)[0]
            if class_name in self.samples_per_class:
                self.samples_per_class[class_name] += 1

        self.logger.info(f"Samples per class: {self.samples_per_class}")

        # Create cache
        cache = None
        if self.config.use_cache:
            cache_dir = self.data_dir / ".spectrogram_cache"
            cache = SpectrogramCache(cache_dir, self.config)

        # Create datasets
        self.train_dataset = AudioDataset(train_files, self.config, self.classes, cache, augment=True)
        self.val_dataset = AudioDataset(val_files, self.config, self.classes, cache, augment=False)
        self.test_dataset = AudioDataset(test_files, self.config, self.classes, cache, augment=False)

        # Weighted sampling
        if self.config.weighted_sampling:
            sample_weights = self._compute_sample_weights(train_files)
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        loader_kwargs = {
            'batch_size': self.config.batch_size,
            'num_workers': self.config.num_workers,
            'pin_memory': self.config.pin_memory,
        }
        if self.config.num_workers > 0:
            loader_kwargs['prefetch_factor'] = self.config.prefetch_factor
            loader_kwargs['persistent_workers'] = self.config.persistent_workers

        self.train_loader = DataLoader(self.train_dataset, sampler=sampler,
                                        shuffle=shuffle if sampler is None else False, **loader_kwargs)
        self.val_loader = DataLoader(self.val_dataset, shuffle=False, **loader_kwargs)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, **loader_kwargs)

    def _compute_sample_weights(self, files: List[str]) -> List[float]:
        """Compute sample weights for weighted sampling."""
        class_counts = {c: 0 for c in self.classes}
        sample_classes = []

        for f in files:
            class_name = Path(f).name.split("-", 1)[0]
            if class_name in class_counts:
                class_counts[class_name] += 1
                sample_classes.append(class_name)

        total = sum(class_counts.values())

        if self.config.sampling_strategy == "inverse_freq":
            class_weights = {c: total / max(class_counts[c], 1) for c in self.classes}
        elif self.config.sampling_strategy == "sqrt_inverse_freq":
            class_weights = {c: math.sqrt(total / max(class_counts[c], 1)) for c in self.classes}
        else:
            class_weights = {c: 1.0 for c in self.classes}

        # Boost rare classes
        for rare_class in self.config.rare_classes:
            if rare_class in class_weights:
                class_weights[rare_class] *= self.config.rare_class_boost

        # Normalize
        max_w = max(class_weights.values())
        class_weights = {c: w / max_w for c, w in class_weights.items()}

        return [class_weights[c] for c in sample_classes]

    def setup_training(self):
        """Setup model, optimizer, scheduler, and loss."""
        self.model = create_model(self.config, self.num_classes, self.device)

        # Loss function
        if self.config.loss_type == "focal":
            self.criterion = FocalLoss(gamma=self.config.focal_gamma)
        elif self.config.loss_type == "weighted" and self.config.class_weights:
            self.criterion = WeightedLoss(
                self.class_to_idx, self.config.class_weights,
                label_smoothing=self.config.label_smoothing
            )
        elif self.config.loss_type == "label_smoothing":
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )

        # Scheduler
        total_steps = len(self.train_loader) * self.config.max_epochs
        if self.config.scheduler == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.lr * 10,
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy='cos'
            )
            self.step_scheduler_per_batch = True
        else:
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', patience=8, factor=0.5)
            self.step_scheduler_per_batch = False

        # Mixed precision
        if self.config.mixed_precision and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        self.logger.info(f"Model: {self.config.backbone}, pretrained={self.config.pretrained}")
        self.logger.info(f"Loss: {self.config.loss_type}, LR: {self.config.lr}")
        self.logger.info(f"Scheduler: {self.config.scheduler}")

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Train E{epoch}", leave=False)
        for inputs, targets in pbar:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.scaler:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

            if self.step_scheduler_per_batch:
                self.scheduler.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})

        return total_loss / total if total > 0 else 0, correct / total if total > 0 else 0

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, float, Dict[str, float]]:
        """Evaluate on a dataset."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        class_correct = {c: 0 for c in self.classes}
        class_total = {c: 0 for c in self.classes}

        for inputs, targets in tqdm(loader, desc="Eval", leave=False):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if self.scaler:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = self.model(inputs)
                    loss = F.cross_entropy(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, targets)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            for pred, target in zip(predicted, targets):
                class_name = self.classes[target.item()]
                class_total[class_name] += 1
                if pred == target:
                    class_correct[class_name] += 1

        per_class_acc = {c: class_correct[c] / class_total[c] if class_total[c] > 0 else 0
                        for c in self.classes}

        return total_loss / total if total > 0 else 0, correct / total if total > 0 else 0, per_class_acc

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': asdict(self.config),
            'classes': self.classes
        }

        torch.save(checkpoint, self.output_dir / "latest.pt")
        if is_best:
            torch.save(checkpoint, self.output_dir / "best.pt")

    def plot_results(self, test_acc: float, test_per_class: Dict[str, float]):
        """Generate training plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss curve
        axes[0, 0].plot(self.history['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(self.history['val_loss'], label='Val', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy curve
        axes[0, 1].plot([a * 100 for a in self.history['train_acc']], label='Train', linewidth=2)
        axes[0, 1].plot([a * 100 for a in self.history['val_acc']], label='Val', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training & Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Per-class accuracy
        classes = list(test_per_class.keys())
        accuracies = [test_per_class[c] * 100 for c in classes]
        colors = ['red' if c in self.config.rare_classes else 'steelblue' for c in classes]
        bars = axes[1, 0].bar(classes, accuracies, color=colors, edgecolor='black', alpha=0.8)
        axes[1, 0].axhline(y=90, color='gray', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].set_title(f'Per-Class Test Accuracy (Overall: {test_acc*100:.2f}%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, axis='y', alpha=0.3)

        for bar, acc in zip(bars, accuracies):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)

        # Summary
        summary = f"""
Configuration: {self.config.name}
Backbone: {self.config.backbone}
Sequence: {self.config.sequence_len}ms
Loss: {self.config.loss_type}
Sampling: {self.config.sampling_strategy}

Results:
  Best Val Acc: {self.best_val_acc * 100:.2f}%
  Test Accuracy: {test_acc * 100:.2f}%
  Best Epoch: {self.best_epoch}

Rare Classes:
"""
        for c in self.config.rare_classes:
            if c in test_per_class:
                summary += f"  {c}: {test_per_class[c]*100:.1f}%\n"

        axes[1, 1].text(0.05, 0.95, summary, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Experiment Summary')

        plt.tight_layout()
        plt.savefig(self.output_dir / "results.png", dpi=150, bbox_inches='tight')
        plt.close()

    def run(self) -> Dict:
        """Run the full training experiment."""
        self.logger.info("=" * 60)
        self.logger.info(f"EXPERIMENT: {self.config.name}")
        self.logger.info("=" * 60)
        self.logger.info(f"Config: {json.dumps(asdict(self.config), indent=2, default=str)}")

        self.prepare_data()
        self.setup_training()

        no_improve = 0
        start_time = time.time()

        try:
            for epoch in range(self.config.max_epochs):
                epoch_start = time.time()

                train_loss, train_acc = self.train_epoch(epoch)
                val_loss, val_acc, per_class_acc = self.evaluate(self.val_loader)

                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                # Track rare class accuracy
                for c in self.config.rare_classes:
                    if c in per_class_acc:
                        if per_class_acc[c] > self.best_rare_class_acc.get(c, 0):
                            self.best_rare_class_acc[c] = per_class_acc[c]

                # Step scheduler (if per-epoch)
                if not self.step_scheduler_per_batch:
                    self.scheduler.step(val_acc)

                current_lr = self.optimizer.param_groups[0]['lr']

                # Save best
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_epoch = epoch
                    no_improve = 0
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    no_improve += 1

                epoch_time = time.time() - epoch_start

                rare_info = " | ".join([f"{c}: {per_class_acc.get(c, 0)*100:.1f}%"
                                       for c in self.config.rare_classes])
                self.logger.info(
                    f"E{epoch:03d} | Train: {train_loss:.4f}/{train_acc*100:.2f}% | "
                    f"Val: {val_loss:.4f}/{val_acc*100:.2f}% | LR: {current_lr:.2e} | "
                    f"{rare_info} | {epoch_time:.1f}s"
                )

                if no_improve >= self.config.early_stop_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break

        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")

        # Load best model and evaluate on test
        best_ckpt = torch.load(self.output_dir / "best.pt", map_location=self.device)
        self.model.load_state_dict(best_ckpt['model_state_dict'])
        test_loss, test_acc, test_per_class = self.evaluate(self.test_loader)

        total_time = time.time() - start_time

        # Plot results
        self.plot_results(test_acc, test_per_class)

        # Save results
        results = {
            'config': asdict(self.config),
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'test_acc': test_acc,
            'test_per_class': test_per_class,
            'best_rare_class_acc': self.best_rare_class_acc,
            'total_time_minutes': total_time / 60,
        }

        with open(self.output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info("\n" + "=" * 60)
        self.logger.info("FINAL RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"Test Accuracy: {test_acc*100:.2f}%")
        self.logger.info(f"Best Val Accuracy: {self.best_val_acc*100:.2f}%")
        for c in self.config.rare_classes:
            if c in test_per_class:
                self.logger.info(f"{c}: {test_per_class[c]*100:.1f}%")

        return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified ANIMAL-SPOT Training Experiment")

    # Required arguments
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to training data directory")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: ./experiments/<timestamp>)")

    # Preset or custom config
    parser.add_argument("--preset", type=str, default=None,
                       help="Use predefined config preset")
    parser.add_argument("--list_presets", action="store_true",
                       help="List available presets and exit")

    # Model
    parser.add_argument("--backbone", type=str, default="resnet18",
                       choices=["resnet18", "resnet34", "convnext_tiny"])
    parser.add_argument("--pretrained", action="store_true")

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--sequence_len", type=int, default=1000)

    # Loss
    parser.add_argument("--loss_type", type=str, default="label_smoothing",
                       choices=["cross_entropy", "label_smoothing", "focal"])

    # Sampling
    parser.add_argument("--no_weighted_sampling", action="store_true")
    parser.add_argument("--rare_class_boost", type=float, default=1.5)

    # Other
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--name", type=str, default="experiment")

    args = parser.parse_args()

    # List presets
    if args.list_presets:
        presets = get_presets()
        print("\nAvailable presets:")
        print("-" * 40)
        for name, config in presets.items():
            print(f"  {name}:")
            print(f"    backbone: {config.backbone}")
            print(f"    sequence_len: {config.sequence_len}ms")
            print(f"    loss: {config.loss_type}")
            print()
        return

    # Create config
    if args.preset:
        presets = get_presets()
        if args.preset not in presets:
            print(f"Unknown preset: {args.preset}")
            print(f"Available: {list(presets.keys())}")
            return
        config = presets[args.preset]
    else:
        config = ExperimentConfig(
            name=args.name,
            backbone=args.backbone,
            pretrained=args.pretrained,
            batch_size=args.batch_size,
            lr=args.lr,
            max_epochs=args.max_epochs,
            sequence_len=args.sequence_len,
            loss_type=args.loss_type,
            weighted_sampling=not args.no_weighted_sampling,
            rare_class_boost=args.rare_class_boost,
        )

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("./experiments") / f"{config.name}_{timestamp}"

    # Run
    print("=" * 60)
    print(f"EXPERIMENT: {config.name}")
    print("=" * 60)
    print(f"Data: {args.data_dir}")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    trainer = Trainer(config, Path(args.data_dir), output_dir, args.device)
    results = trainer.run()

    print(f"\nResults saved to: {output_dir}")
    return results


if __name__ == "__main__":
    main()
