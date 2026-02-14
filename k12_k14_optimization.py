#!/usr/bin/env python
"""
K12/K14 Optimization Experiments
================================
Specifically designed to maximize performance on rare classes K12 and K14.

Key insights:
- K12: 273 training samples (1.9% of data)
- K14: 70 training samples (0.5% of data) - MOST RARE
- Long sequence (1000ms) achieved K12=94.9%, K14=83.3%
- But overall accuracy dropped to 91.48%

Strategy: Combine long sequence benefits with techniques that maintain overall accuracy
"""

import os
import sys
import json
import logging
import time
import math
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from scipy.ndimage import zoom

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SCRIPT_DIR = Path(__file__).parent.absolute()


@dataclass
class K12K14Config:
    """Configuration optimized for K12/K14."""
    name: str

    # Model
    backbone: str = "resnet18"
    pretrained: bool = False

    # Training
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 100
    early_stop_patience: int = 25

    # Audio - KEY: sequence length affects rare class performance
    sr: int = 44100
    sequence_len: int = 500
    n_fft: int = 1024
    hop_length: int = 172
    n_freq_bins: int = 256
    fmin: int = 500
    fmax: int = 10000
    freq_compression: str = "linear"

    # Loss
    loss_type: str = "label_smoothing"
    label_smoothing: float = 0.1
    class_weight_boost: Dict = None  # Extra weight for K12/K14

    # Sampling - critical for rare classes
    weighted_sampling: bool = True
    sampling_strategy: str = "sqrt_inverse_freq"
    k12_k14_boost: float = 1.0  # Extra boost for K12/K14 sampling

    # Training
    use_compile: bool = True
    scheduler: str = "onecycle"
    grad_clip: float = 1.0

    # Data loading
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 4
    persistent_workers: bool = True
    use_cache: bool = True


class SpectrogramCache:
    def __init__(self, cache_dir: Path, config: K12K14Config):
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
    def __init__(self, file_list: List[str], config: K12K14Config,
                 classes: List[str], cache: Optional[SpectrogramCache] = None,
                 augment: bool = False):
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


class K12K14WeightedLoss(nn.Module):
    """Cross-entropy with extra weight for K12 and K14."""
    def __init__(self, class_to_idx: Dict, k12_weight: float = 2.0, k14_weight: float = 3.0,
                 label_smoothing: float = 0.1):
        super().__init__()
        self.class_to_idx = class_to_idx
        self.k12_idx = class_to_idx.get('K12', -1)
        self.k14_idx = class_to_idx.get('K14', -1)
        self.k12_weight = k12_weight
        self.k14_weight = k14_weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Create per-sample weights
        weights = torch.ones_like(targets, dtype=torch.float32)
        weights[targets == self.k12_idx] = self.k12_weight
        weights[targets == self.k14_idx] = self.k14_weight

        # Compute loss per sample
        ce_loss = F.cross_entropy(inputs, targets, reduction='none',
                                   label_smoothing=self.label_smoothing)
        weighted_loss = (ce_loss * weights).mean()
        return weighted_loss


def create_model(config: K12K14Config, num_classes: int, device: str) -> nn.Module:
    import torchvision.models as models

    if config.backbone == "convnext_tiny":
        model = models.convnext_tiny(weights='IMAGENET1K_V1' if config.pretrained else None)
        # Adapt first conv layer for 1-channel input
        old_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(1, old_conv.out_channels,
                                          kernel_size=old_conv.kernel_size,
                                          stride=old_conv.stride,
                                          padding=old_conv.padding,
                                          bias=old_conv.bias is not None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    else:
        if config.backbone == "resnet34":
            model = models.resnet34(weights='IMAGENET1K_V1' if config.pretrained else None)
        else:
            model = models.resnet18(weights='IMAGENET1K_V1' if config.pretrained else None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    model = model.to(device)

    if config.use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except:
            pass

    return model


class K12K14Trainer:
    def __init__(self, config: K12K14Config, data_dir: Path, output_dir: Path, device: str = "cuda"):
        self.config = config
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

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
        self.best_k12_acc = 0.0
        self.best_k14_acc = 0.0
        self.best_epoch = 0
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def prepare_data(self):
        audio_files = []
        for ext in ['*.wav', '*.WAV']:
            audio_files.extend(self.data_dir.rglob(ext))
        audio_files = [str(f) for f in audio_files]

        classes = set()
        for f in audio_files:
            class_name = Path(f).name.split("-", 1)[0]
            classes.add(class_name)
        self.classes = sorted(list(classes))
        self.num_classes = len(self.classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.logger.info(f"Found {len(audio_files)} files with {self.num_classes} classes")

        # Split by tape
        tape_to_files = {}
        for f in audio_files:
            parts = Path(f).stem.split("_")
            tape = parts[3] if len(parts) >= 4 else "unknown"
            if tape not in tape_to_files:
                tape_to_files[tape] = []
            tape_to_files[tape].append(f)

        tapes = list(tape_to_files.keys())
        np.random.seed(42)
        np.random.shuffle(tapes)

        n_tapes = len(tapes)
        train_end = int(0.7 * n_tapes)
        val_end = int(0.85 * n_tapes)

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
        self.logger.info(f"K12: {self.samples_per_class.get('K12', 0)}, K14: {self.samples_per_class.get('K14', 0)}")

        # Create cache
        cache = None
        if self.config.use_cache:
            cache_dir = self.data_dir / ".spectrogram_cache"
            cache = SpectrogramCache(cache_dir, self.config)

        self.train_dataset = AudioDataset(train_files, self.config, self.classes, cache, augment=True)
        self.val_dataset = AudioDataset(val_files, self.config, self.classes, cache, augment=False)
        self.test_dataset = AudioDataset(test_files, self.config, self.classes, cache, augment=False)

        # Weighted sampling with BOOST for K12/K14
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
            'prefetch_factor': self.config.prefetch_factor if self.config.num_workers > 0 else None,
            'persistent_workers': self.config.persistent_workers if self.config.num_workers > 0 else False,
        }

        self.train_loader = DataLoader(self.train_dataset, sampler=sampler,
                                        shuffle=shuffle if sampler is None else False, **loader_kwargs)
        self.val_loader = DataLoader(self.val_dataset, shuffle=False, **loader_kwargs)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, **loader_kwargs)

    def _compute_sample_weights(self, files: List[str]) -> List[float]:
        class_counts = {c: 0 for c in self.classes}
        sample_classes = []

        for f in files:
            class_name = Path(f).name.split("-", 1)[0]
            if class_name in class_counts:
                class_counts[class_name] += 1
                sample_classes.append(class_name)

        total = sum(class_counts.values())

        # Base weights using sqrt_inverse_freq or inverse_freq
        if self.config.sampling_strategy == "inverse_freq":
            class_weights = {c: total / max(class_counts[c], 1) for c in self.classes}
        else:  # sqrt_inverse_freq
            class_weights = {c: math.sqrt(total / max(class_counts[c], 1)) for c in self.classes}

        # BOOST for K12 and K14
        boost = self.config.k12_k14_boost
        if 'K12' in class_weights:
            class_weights['K12'] *= boost
        if 'K14' in class_weights:
            class_weights['K14'] *= boost

        # Normalize
        max_w = max(class_weights.values())
        class_weights = {c: w / max_w for c, w in class_weights.items()}

        self.logger.info(f"Sampling weights (K12/K14 boosted by {boost}x): K12={class_weights.get('K12', 0):.2f}, K14={class_weights.get('K14', 0):.2f}")

        return [class_weights[c] for c in sample_classes]

    def setup_training(self):
        self.model = create_model(self.config, self.num_classes, self.device)

        # Use weighted loss for K12/K14
        if self.config.class_weight_boost:
            k12_w = self.config.class_weight_boost.get('K12', 1.0)
            k14_w = self.config.class_weight_boost.get('K14', 1.0)
            self.criterion = K12K14WeightedLoss(
                self.class_to_idx,
                k12_weight=k12_w,
                k14_weight=k14_w,
                label_smoothing=self.config.label_smoothing
            )
            self.logger.info(f"Using K12/K14 weighted loss: K12={k12_w}x, K14={k14_w}x")
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )

        total_steps = len(self.train_loader) * self.config.max_epochs
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.lr * 10,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )

        self.scaler = torch.amp.GradScaler('cuda')

        self.logger.info(f"Model: {self.config.backbone}")
        self.logger.info(f"Sequence length: {self.config.sequence_len}ms")

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Train E{epoch}", leave=False)
        for inputs, targets in pbar:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

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
            self.scheduler.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})

        return total_loss / total if total > 0 else 0, correct / total if total > 0 else 0

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        class_correct = {c: 0 for c in self.classes}
        class_total = {c: 0 for c in self.classes}

        for inputs, targets in tqdm(loader, desc="Eval", leave=False):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, targets, label_smoothing=self.config.label_smoothing)

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

    def save_checkpoint(self, epoch: int, per_class_acc: Dict, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_k12_acc': self.best_k12_acc,
            'best_k14_acc': self.best_k14_acc,
            'config': asdict(self.config),
            'classes': self.classes
        }

        torch.save(checkpoint, self.output_dir / "latest.pt")
        if is_best:
            torch.save(checkpoint, self.output_dir / "best.pt")

    def run(self) -> Dict:
        self.logger.info("=" * 60)
        self.logger.info(f"K12/K14 OPTIMIZATION: {self.config.name}")
        self.logger.info("=" * 60)

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

                k12_acc = per_class_acc.get('K12', 0)
                k14_acc = per_class_acc.get('K14', 0)
                current_lr = self.optimizer.param_groups[0]['lr']

                # Track best K12/K14
                if k12_acc > self.best_k12_acc:
                    self.best_k12_acc = k12_acc
                if k14_acc > self.best_k14_acc:
                    self.best_k14_acc = k14_acc

                # Save best overall
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_epoch = epoch
                    no_improve = 0
                    self.save_checkpoint(epoch, per_class_acc, is_best=True)
                else:
                    no_improve += 1

                epoch_time = time.time() - epoch_start

                self.logger.info(
                    f"E{epoch:03d} | Train: {train_loss:.4f}/{train_acc*100:.2f}% | "
                    f"Val: {val_loss:.4f}/{val_acc*100:.2f}% | "
                    f"K12: {k12_acc*100:.1f}% | K14: {k14_acc*100:.1f}% | "
                    f"LR: {current_lr:.2e} | {epoch_time:.1f}s"
                )

                if no_improve >= self.config.early_stop_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break

        except KeyboardInterrupt:
            self.logger.info("Training interrupted")

        # Final test
        best_ckpt = torch.load(self.output_dir / "best.pt", map_location=self.device)
        self.model.load_state_dict(best_ckpt['model_state_dict'])
        test_loss, test_acc, test_per_class = self.evaluate(self.test_loader)

        total_time = time.time() - start_time

        results = {
            'config': asdict(self.config),
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'test_acc': test_acc,
            'test_per_class': test_per_class,
            'best_k12_acc': self.best_k12_acc,
            'best_k14_acc': self.best_k14_acc,
            'test_k12': test_per_class.get('K12', 0),
            'test_k14': test_per_class.get('K14', 0),
            'total_time_minutes': total_time / 60,
        }

        with open(self.output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info("\n" + "=" * 60)
        self.logger.info("FINAL RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"Test Accuracy: {test_acc*100:.2f}%")
        self.logger.info(f"K12: {test_per_class.get('K12',0)*100:.1f}%")
        self.logger.info(f"K14: {test_per_class.get('K14',0)*100:.1f}%")
        self.logger.info(f"Best K12 seen: {self.best_k12_acc*100:.1f}%")
        self.logger.info(f"Best K14 seen: {self.best_k14_acc*100:.1f}%")

        return results


def get_k12_k14_experiments() -> List[K12K14Config]:
    """Experiments specifically targeting K12/K14 improvement."""
    configs = []

    # Exp 1: Long sequence (proven to help K12/K14) + label smoothing
    configs.append(K12K14Config(
        name="exp1_long_seq_label_smooth",
        sequence_len=800,  # Longer but not too long
        loss_type="label_smoothing",
        label_smoothing=0.1,
        weighted_sampling=True,
        sampling_strategy="sqrt_inverse_freq",
        k12_k14_boost=1.5,  # 1.5x boost for K12/K14 sampling
    ))

    # Exp 2: Very aggressive K12/K14 sampling + loss weighting
    configs.append(K12K14Config(
        name="exp2_aggressive_k12_k14",
        sequence_len=500,
        loss_type="label_smoothing",
        label_smoothing=0.1,
        weighted_sampling=True,
        sampling_strategy="inverse_freq",  # More aggressive
        k12_k14_boost=2.0,  # 2x boost
        class_weight_boost={'K12': 2.0, 'K14': 3.0},  # Extra loss weight
    ))

    # Exp 3: Maximum sequence length for best rare class performance
    configs.append(K12K14Config(
        name="exp3_max_seq_balanced",
        sequence_len=1000,  # Same as original long_seq that got K12=94.9%
        loss_type="label_smoothing",
        label_smoothing=0.1,
        weighted_sampling=True,
        sampling_strategy="sqrt_inverse_freq",
        k12_k14_boost=1.5,
    ))

    # Exp 4: ResNet34 with long sequence (more capacity)
    configs.append(K12K14Config(
        name="exp4_resnet34_long_seq",
        backbone="resnet34",
        sequence_len=800,
        batch_size=48,  # Slightly smaller for larger model
        loss_type="label_smoothing",
        label_smoothing=0.1,
        weighted_sampling=True,
        sampling_strategy="sqrt_inverse_freq",
        k12_k14_boost=1.5,
    ))

    # Exp 5: Extreme K14 focus (most rare class)
    configs.append(K12K14Config(
        name="exp5_k14_extreme_focus",
        sequence_len=800,
        loss_type="label_smoothing",
        label_smoothing=0.1,
        weighted_sampling=True,
        sampling_strategy="inverse_freq",
        k12_k14_boost=3.0,  # 3x boost!
        class_weight_boost={'K12': 2.5, 'K14': 4.0},
    ))

    # Exp 6: ConvNeXt with all optimizations (potential new champion!)
    # Hypothesis: ConvNeXt's larger receptive fields (7x7) + 1000ms sequence
    # could capture even more acoustic patterns than ResNet18
    # Update: pretrained=False based on Nov results (scratch > pretrained for spectrograms)
    configs.append(K12K14Config(
        name="exp6_convnext_ultimate",
        backbone="convnext_tiny",
        pretrained=False,  # Nov showed scratch > pretrained for spectrograms
        sequence_len=1000,  # Same as exp3 champion
        batch_size=32,  # Smaller due to larger model + long sequence
        lr=3e-4,  # Same as ResNet18 since not pretrained
        weight_decay=0.01,  # ConvNeXt prefers higher weight decay
        loss_type="label_smoothing",
        label_smoothing=0.1,
        weighted_sampling=True,
        sampling_strategy="sqrt_inverse_freq",
        k12_k14_boost=1.5,
    ))

    # Exp 7: ResNet18 with 1200ms sequence
    configs.append(K12K14Config(
        name="exp7_resnet18_1200ms",
        backbone="resnet18",
        pretrained=False,
        sequence_len=1200,
        batch_size=64,
        lr=3e-4,
        weight_decay=1e-4,
        loss_type="label_smoothing",
        label_smoothing=0.1,
        weighted_sampling=True,
        sampling_strategy="sqrt_inverse_freq",
        k12_k14_boost=1.5,
    ))

    # Exp 8: ResNet18 with 1500ms sequence
    configs.append(K12K14Config(
        name="exp8_resnet18_1500ms",
        backbone="resnet18",
        pretrained=False,
        sequence_len=1500,
        batch_size=48,
        lr=3e-4,
        weight_decay=1e-4,
        loss_type="label_smoothing",
        label_smoothing=0.1,
        weighted_sampling=True,
        sampling_strategy="sqrt_inverse_freq",
        k12_k14_boost=1.5,
    ))

    # Exp 9: ConvNeXt with 1200ms, batch_size=64
    configs.append(K12K14Config(
        name="exp9_convnext_1200ms_bs64",
        backbone="convnext_tiny",
        pretrained=False,
        sequence_len=1200,
        batch_size=64,
        lr=3e-4,
        weight_decay=0.01,
        loss_type="label_smoothing",
        label_smoothing=0.1,
        weighted_sampling=True,
        sampling_strategy="sqrt_inverse_freq",
        k12_k14_boost=1.5,
    ))

    # Exp 10: Raw ANIMAL-SPOT with Olga's original parameters
    # n_fft=4096, hop_length=441 from DefaultSpecDatasetOps in audiodataset.py
    # sequence_len=1280ms from main.py default
    # No weighted sampling, no label smoothing, no K12/K14 boost (vanilla)
    configs.append(K12K14Config(
        name="exp10_raw_animal_spot",
        backbone="resnet18",
        pretrained=False,
        sequence_len=1280,
        batch_size=64,
        lr=3e-4,
        weight_decay=1e-4,
        n_fft=4096,
        hop_length=441,
        n_freq_bins=256,
        fmin=500,
        fmax=10000,
        freq_compression="linear",
        loss_type="cross_entropy",
        label_smoothing=0.0,
        weighted_sampling=False,
        k12_k14_boost=1.0,
    ))

    return configs


def main():
    data_dir = Path("/mnt/c/Users/Iaroslav/CETACEANS/new_training_data")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(f"/mnt/c/Users/Iaroslav/CETACEANS/k12_k14_optimization_{timestamp}")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    configs = get_k12_k14_experiments()

    print("=" * 60)
    print("K12/K14 OPTIMIZATION EXPERIMENTS")
    print("=" * 60)
    print(f"Running {len(configs)} experiments")
    print(f"Output: {base_output_dir}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_results = []
    best_k12 = 0
    best_k14 = 0
    best_overall = 0
    best_config = None

    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}/{len(configs)}: {config.name}")
        print(f"{'='*60}")

        try:
            output_dir = base_output_dir / config.name
            trainer = K12K14Trainer(config, data_dir, output_dir, "cuda")
            results = trainer.run()
            all_results.append(results)

            # Track best
            if results['test_k12'] > best_k12:
                best_k12 = results['test_k12']
            if results['test_k14'] > best_k14:
                best_k14 = results['test_k14']
            if results['test_acc'] > best_overall:
                best_overall = results['test_acc']
                best_config = config.name

            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    for r in sorted(all_results, key=lambda x: x['test_acc'], reverse=True):
        print(f"{r['config']['name']}: Overall={r['test_acc']*100:.2f}%, K12={r['test_k12']*100:.1f}%, K14={r['test_k14']*100:.1f}%")

    print(f"\nBest K12: {best_k12*100:.1f}%")
    print(f"Best K14: {best_k14*100:.1f}%")
    print(f"Best Overall: {best_overall*100:.2f}% ({best_config})")

    # Save summary
    summary = {
        'timestamp': timestamp,
        'best_k12': best_k12,
        'best_k14': best_k14,
        'best_overall': best_overall,
        'best_config': best_config,
        'all_results': all_results
    }

    with open(base_output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {base_output_dir}")
    return base_output_dir, all_results


if __name__ == "__main__":
    main()
