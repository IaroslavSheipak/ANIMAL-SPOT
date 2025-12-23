"""
Module: modern_utils.py
Modern utilities and helper functions for ANIMAL-SPOT.

This module provides updated, type-hinted utilities that work with
modern Python (3.10+) and PyTorch (2.0+).

Authors: CETACEANS Project
Last Access: December 2024
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Optional, Union, Dict, List, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
import time

import torch
import torch.nn as nn
import numpy as np


# ============================================
# Configuration Classes
# ============================================

@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sr: int = 44100
    n_fft: int = 1024
    hop_length: int = 512
    n_freq_bins: int = 256
    fmin: int = 500
    fmax: int = 10000
    freq_compression: str = "linear"
    preemphasis: float = 0.98
    min_level_db: float = -100
    ref_level_db: float = 20

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AudioConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    learning_rate: float = 1e-4
    max_epochs: int = 150
    early_stopping_patience: int = 20
    lr_patience: int = 8
    lr_decay_factor: float = 0.5
    num_workers: int = 4
    val_interval: int = 2
    weighted_sampling: bool = False
    sampling_strategy: str = "inverse_freq"
    augmentation: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelConfig:
    """Model configuration."""
    backbone: str = "convnext"  # "resnet" or "convnext"
    pretrained: bool = False
    num_classes: int = 2
    resnet_size: int = 18  # Only for resnet backbone

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================
# Timing Utilities
# ============================================

@contextmanager
def timer(name: str = "Operation", logger: Optional[logging.Logger] = None):
    """Context manager for timing operations."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    msg = f"{name} took {elapsed:.2f}s"
    if logger:
        logger.info(msg)
    else:
        print(msg)


class ProgressTimer:
    """Track progress with ETA estimation."""

    def __init__(self, total: int, name: str = "Progress"):
        self.total = total
        self.name = name
        self.current = 0
        self.start_time = time.perf_counter()

    def update(self, n: int = 1) -> str:
        self.current += n
        elapsed = time.perf_counter() - self.start_time
        if self.current > 0:
            eta = elapsed / self.current * (self.total - self.current)
            progress = self.current / self.total * 100
            return f"{self.name}: {progress:.1f}% ({self.current}/{self.total}) ETA: {eta:.0f}s"
        return f"{self.name}: 0%"


# ============================================
# Path Utilities
# ============================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
    """Find the most recent checkpoint file."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("*.checkpoint"))
    if not checkpoints:
        return None

    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def find_audio_files(
    directory: Union[str, Path],
    extensions: List[str] = [".wav", ".mp3", ".flac"],
    recursive: bool = True
) -> List[Path]:
    """Find all audio files in directory."""
    directory = Path(directory)
    files = []
    pattern = "**/*" if recursive else "*"

    for ext in extensions:
        files.extend(directory.glob(f"{pattern}{ext}"))
        files.extend(directory.glob(f"{pattern}{ext.upper()}"))

    return sorted(files)


# ============================================
# Model Utilities
# ============================================

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in megabytes."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


def freeze_layers(model: nn.Module, layer_names: List[str]) -> None:
    """Freeze specific layers by name."""
    for name, param in model.named_parameters():
        if any(ln in name for ln in layer_names):
            param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = True


# ============================================
# Device Utilities
# ============================================

def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available device."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    return torch.device("cpu")


def get_device_info() -> Dict[str, Any]:
    """Get information about available devices."""
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["gpu_memory_gb"] = [
            torch.cuda.get_device_properties(i).total_memory / 1024**3
            for i in range(torch.cuda.device_count())
        ]

    return info


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================
# Checkpoint Utilities
# ============================================

def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    best_metric: float = 0.0,
    extra_info: Optional[Dict[str, Any]] = None
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_metric": best_metric,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if extra_info is not None:
        checkpoint["extra_info"] = extra_info

    torch.save(checkpoint, path)


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Load training checkpoint."""
    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


# ============================================
# Logging Utilities
# ============================================

def setup_logger(
    name: str,
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Set up a logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ============================================
# Data Utilities
# ============================================

def compute_class_weights(
    class_counts: Dict[int, int],
    strategy: str = "inverse_freq",
    normalize: bool = True
) -> Dict[int, float]:
    """
    Compute class weights for imbalanced datasets.

    Strategies:
    - inverse_freq: 1 / class_count
    - sqrt_inverse_freq: 1 / sqrt(class_count)
    - effective_num: (1-beta) / (1 - beta^n), beta=0.9999
    """
    import math

    total = sum(class_counts.values())
    num_classes = len(class_counts)

    if strategy == "inverse_freq":
        weights = {c: total / (num_classes * count) for c, count in class_counts.items()}
    elif strategy == "sqrt_inverse_freq":
        weights = {c: math.sqrt(total / count) for c, count in class_counts.items()}
    elif strategy == "effective_num":
        beta = 0.9999
        weights = {c: (1 - beta) / (1 - math.pow(beta, count)) for c, count in class_counts.items()}
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if normalize:
        max_weight = max(weights.values())
        weights = {c: w / max_weight for c, w in weights.items()}

    return weights


def train_val_test_split(
    items: List[Any],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42
) -> Tuple[List[Any], List[Any], List[Any]]:
    """Split items into train/val/test sets."""
    import random
    random.seed(seed)

    items = list(items)
    random.shuffle(items)

    n = len(items)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    return items[:train_end], items[train_end:val_end], items[val_end:]


# ============================================
# Reproducibility
# ============================================

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
