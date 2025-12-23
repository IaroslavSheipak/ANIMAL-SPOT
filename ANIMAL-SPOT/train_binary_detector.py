#!/usr/bin/env python3
"""
Module: train_binary_detector.py
Train a binary detection model (call vs. noise) for the two-stage pipeline.

This model is used as Stage 1 to filter out noise before multi-class classification.

Usage:
    python train_binary_detector.py --data_dir /path/to/data \
        --noise_class NOISE --output_dir /path/to/output

The data directory should contain subdirectories for each class.
One subdirectory should be named 'NOISE' (or specified with --noise_class).
All other classes will be merged into a single 'CALL' class.

Authors: CETACEANS Project
Last Access: December 2024
"""

import os
import sys
import json
import math
import pathlib
import argparse
import platform
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler

from data.audiodataset import (
    get_audio_files_from_dir,
    DatabaseCsvSplit,
    DefaultSpecDatasetOps,
    Dataset,
)

from trainer import Trainer
from utils.logging import Logger
from collections import OrderedDict
from models.residual_encoder import DefaultEncoderOpts as DefaultResNetEncoderOpts
from models.residual_encoder import ResidualEncoder
from models.convnext_encoder import DefaultEncoderOpts as DefaultConvNextEncoderOpts
from models.convnext_encoder import ConvNextEncoder
from models.classifier import Classifier, DefaultClassifierOpts
import utils.metrics as m


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser(description="Train binary detection model for two-stage pipeline")

# Required paths
parser.add_argument("--data_dir", type=str, required=True, help="Directory with audio data")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model and logs")
parser.add_argument("--noise_class", type=str, default="NOISE", help="Name of the noise class directory")
parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for spectrograms")
parser.add_argument("--noise_dir", type=str, default=None, help="Additional noise files for augmentation")

# Training parameters
parser.add_argument("--backbone", type=str, default="resnet", choices=["resnet", "convnext"])
parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights")
parser.add_argument("--max_epochs", type=int, default=100, help="Maximum training epochs")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--num_workers", type=int, default=4, help="Data loading workers")
parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")

# Audio parameters
parser.add_argument("--sequence_len", type=int, default=800, help="Sequence length in ms")
parser.add_argument("--sr", type=int, default=44100, help="Sample rate")
parser.add_argument("--n_fft", type=int, default=1024, help="FFT size")
parser.add_argument("--hop_length", type=int, default=172, help="Hop length")
parser.add_argument("--n_freq_bins", type=int, default=256, help="Number of frequency bins")
parser.add_argument("--fmin", type=int, default=500, help="Minimum frequency")
parser.add_argument("--fmax", type=int, default=10000, help="Maximum frequency")

# Balancing
parser.add_argument("--balance_classes", action="store_true", help="Balance call/noise classes")

parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")

ARGS = parser.parse_args()
ARGS.cuda = torch.cuda.is_available() and not ARGS.no_cuda
ARGS.device = torch.device("cuda") if ARGS.cuda else torch.device("cpu")

# Create output directories
os.makedirs(ARGS.output_dir, exist_ok=True)
os.makedirs(os.path.join(ARGS.output_dir, "model"), exist_ok=True)
os.makedirs(os.path.join(ARGS.output_dir, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(ARGS.output_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(ARGS.output_dir, "summaries"), exist_ok=True)

log = Logger("BINARY_TRAIN", ARGS.debug, os.path.join(ARGS.output_dir, "logs"))


def prepare_binary_dataset(data_dir, noise_class, output_dir):
    """
    Prepare a binary dataset by creating symlinks.
    All non-noise classes are merged into 'CALL' class.
    """
    binary_data_dir = os.path.join(output_dir, "binary_data")

    # Create directories
    call_dir = os.path.join(binary_data_dir, "CALL")
    noise_dir = os.path.join(binary_data_dir, "NOISE")
    os.makedirs(call_dir, exist_ok=True)
    os.makedirs(noise_dir, exist_ok=True)

    # Find all class directories
    data_path = pathlib.Path(data_dir)
    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]

    call_count = 0
    noise_count = 0

    for class_dir in class_dirs:
        class_name = class_dir.name.upper()

        # Get all audio files in this class
        audio_files = list(class_dir.glob("*.wav")) + list(class_dir.glob("*.WAV"))

        for audio_file in audio_files:
            # Determine target directory
            if class_name == noise_class.upper():
                target_dir = noise_dir
                noise_count += 1
                prefix = "NOISE"
            else:
                target_dir = call_dir
                call_count += 1
                prefix = "CALL"

            # Create new filename with binary class prefix
            # Format: CALL-original_filename or NOISE-original_filename
            original_name = audio_file.name
            # Remove existing class prefix if present
            if "-" in original_name:
                parts = original_name.split("-", 1)
                new_name = f"{prefix}-{parts[1]}"
            else:
                new_name = f"{prefix}-{original_name}"

            # Copy file (or create symlink)
            target_path = os.path.join(target_dir, new_name)
            if not os.path.exists(target_path):
                try:
                    # Try symlink first (faster, saves space)
                    os.symlink(str(audio_file.absolute()), target_path)
                except OSError:
                    # Fall back to copy
                    shutil.copy2(str(audio_file), target_path)

    log.info(f"Prepared binary dataset:")
    log.info(f"  CALL samples: {call_count}")
    log.info(f"  NOISE samples: {noise_count}")

    return binary_data_dir


def save_model(model, encoder, encoderOpts, classifier, classifierOpts, dataOpts,
               path, class_dist_dict, backbone_type="resnet"):
    """Save the trained model."""
    model = model.cpu()
    encoder = encoder.cpu()
    classifier = classifier.cpu()

    save_dict = {
        "backbone": backbone_type,
        "encoderOpts": encoderOpts,
        "classifierOpts": classifierOpts,
        "dataOpts": dataOpts,
        "encoderState": encoder.state_dict(),
        "classifierState": classifier.state_dict(),
        "classes": class_dist_dict,
        "model_type": "binary_detector"
    }

    torch.save(save_dict, path)
    log.info(f"Model saved to: {path}")


def get_classes(database):
    """Get classes from filenames."""
    classes = set()
    for file in database:
        if platform.system() == "Windows":
            class_name = file.split("\\")[-1].split("-", 1)[0]
        else:
            class_name = file.split("/")[-1].split("-", 1)[0]
        classes.add(class_name)
    return sorted(list(classes))


if __name__ == "__main__":
    log.info("="*60)
    log.info("Training Binary Detection Model for Two-Stage Pipeline")
    log.info("="*60)
    log.info(f"Device: {ARGS.device}")
    log.info(f"Data directory: {ARGS.data_dir}")
    log.info(f"Noise class: {ARGS.noise_class}")

    # Prepare binary dataset
    log.info("Preparing binary dataset...")
    binary_data_dir = prepare_binary_dataset(ARGS.data_dir, ARGS.noise_class, ARGS.output_dir)

    # Set up encoder options
    if ARGS.backbone == "convnext":
        encoderOpts = DefaultConvNextEncoderOpts.copy()
        encoderOpts["pretrained"] = ARGS.pretrained
        log.info("Using ConvNeXt backbone")
    else:
        encoderOpts = DefaultResNetEncoderOpts.copy()
        encoderOpts["pretrained"] = ARGS.pretrained
        encoderOpts["resnet_size"] = 18
        log.info("Using ResNet-18 backbone")

    classifierOpts = DefaultClassifierOpts.copy()
    classifierOpts["num_classes"] = 2  # Binary: CALL vs NOISE

    dataOpts = DefaultSpecDatasetOps.copy()
    dataOpts["sr"] = ARGS.sr
    dataOpts["n_fft"] = ARGS.n_fft
    dataOpts["hop_length"] = ARGS.hop_length
    dataOpts["n_freq_bins"] = ARGS.n_freq_bins
    dataOpts["fmin"] = ARGS.fmin
    dataOpts["fmax"] = ARGS.fmax

    # Compute sequence length in frames
    sequence_len = int(float(ARGS.sequence_len) / 1000 * dataOpts["sr"] / dataOpts["hop_length"])
    log.info(f"Sequence length: {ARGS.sequence_len}ms = {sequence_len} frames")

    # Set up data
    split_fracs = {"train": 0.7, "val": 0.15, "test": 0.15}
    input_data = DatabaseCsvSplit(split_fracs, working_dir=binary_data_dir, split_per_dir=True)
    audio_files = list(get_audio_files_from_dir(binary_data_dir))
    log.info(f"Found {len(audio_files)} audio files")

    if ARGS.noise_dir:
        noise_files = [str(p) for p in pathlib.Path(ARGS.noise_dir).glob("*.wav")]
    else:
        noise_files = []

    classes = get_classes(database=audio_files)
    log.info(f"Classes: {classes}")

    if len(classes) != 2:
        log.error(f"Expected 2 classes (CALL, NOISE), found: {classes}")
        sys.exit(1)

    # Create datasets
    datasets = {
        split: Dataset(
            file_names=input_data.load(split, audio_files),
            working_dir=binary_data_dir,
            cache_dir=ARGS.cache_dir,
            sr=dataOpts["sr"],
            n_fft=dataOpts["n_fft"],
            hop_length=dataOpts["hop_length"],
            n_freq_bins=dataOpts["n_freq_bins"],
            freq_compression="linear",
            f_min=dataOpts["fmin"],
            f_max=dataOpts["fmax"],
            seq_len=sequence_len,
            augmentation=(split == "train"),
            noise_files=noise_files,
            dataset_name=split,
            classes=classes,
            min_max_normalize=True
        )
        for split in split_fracs.keys()
    }

    # Create dataloaders with optional balancing
    dataloaders = {}
    for split in split_fracs.keys():
        if split == "train" and ARGS.balance_classes:
            # Compute sample weights
            class_counts = {}
            for idx in range(len(datasets[split])):
                _, meta = datasets[split][idx]
                label = meta['call'].item() if hasattr(meta['call'], 'item') else meta['call']
                class_counts[label] = class_counts.get(label, 0) + 1

            log.info(f"Class distribution in training: {class_counts}")

            num_samples = len(datasets[split])
            class_weights = {c: num_samples / count for c, count in class_counts.items()}

            sample_weights = []
            for idx in range(len(datasets[split])):
                _, meta = datasets[split][idx]
                label = meta['call'].item() if hasattr(meta['call'], 'item') else meta['call']
                sample_weights.append(class_weights[label])

            sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(sample_weights),
                num_samples=len(datasets[split]),
                replacement=True
            )
            dataloaders[split] = torch.utils.data.DataLoader(
                datasets[split],
                batch_size=ARGS.batch_size,
                sampler=sampler,
                num_workers=ARGS.num_workers,
                drop_last=True,
                pin_memory=True,
            )
        else:
            dataloaders[split] = torch.utils.data.DataLoader(
                datasets[split],
                batch_size=ARGS.batch_size,
                shuffle=(split == "train"),
                num_workers=ARGS.num_workers,
                drop_last=False if split in ["val", "test"] else True,
                pin_memory=True,
            )

    # Create model
    if ARGS.backbone == "convnext":
        encoder = ConvNextEncoder(encoderOpts)
        encoder_out_ch = 768
    else:
        encoder = ResidualEncoder(encoderOpts)
        encoder_out_ch = 512 * encoder.block_type.expansion

    classifierOpts["input_channels"] = encoder_out_ch
    classifier = Classifier(classifierOpts)

    model = nn.Sequential(
        OrderedDict([("encoder", encoder), ("classifier", classifier)])
    )

    # Binary classification metrics
    metrics = {
        "tp": m.TruePositives(ARGS.device),
        "tn": m.TrueNegatives(ARGS.device),
        "fp": m.FalsePositives(ARGS.device),
        "fn": m.FalseNegatives(ARGS.device),
        "accuracy": m.Accuracy(ARGS.device),
        "f1": m.F1Score(ARGS.device),
        "precision": m.Precision(ARGS.device),
        "recall": m.Recall(ARGS.device),
    }

    # Set up trainer
    trainer = Trainer(
        model=model,
        logger=log,
        prefix="binary_detector",
        checkpoint_dir=os.path.join(ARGS.output_dir, "checkpoints"),
        summary_dir=os.path.join(ARGS.output_dir, "summaries"),
        n_summaries=4,
        start_scratch=True,
    )

    optimizer = optim.Adam(model.parameters(), lr=ARGS.lr, betas=(0.9, 0.999))

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=4,
        factor=0.5,
        threshold=1e-3,
        threshold_mode="abs",
    )

    loss_fn = nn.CrossEntropyLoss()

    # Train
    log.info("Starting training...")
    model = trainer.fit(
        dataloaders["train"],
        dataloaders["val"],
        dataloaders["test"],
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        n_epochs=ARGS.max_epochs,
        val_interval=2,
        patience_early_stopping=20,
        device=ARGS.device,
        metrics=metrics,
        val_metric="accuracy",
        val_metric_mode="max",
    )

    # Save model
    model_path = os.path.join(ARGS.output_dir, "model", "binary_detector.pk")
    class_dist_dict = datasets["train"].class_dist_dict
    save_model(
        model, model.encoder, encoderOpts,
        model.classifier, classifierOpts, dataOpts,
        model_path, class_dist_dict, ARGS.backbone
    )

    log.info("Training complete!")
    log.close()
