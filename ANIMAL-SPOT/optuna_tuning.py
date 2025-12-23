#!/usr/bin/env python3
"""
Module: optuna_tuning.py
Hyperparameter tuning for ANIMAL-SPOT using Optuna.

This script provides automated hyperparameter optimization for the ANIMAL-SPOT
framework. It searches for optimal values of learning rate, batch size,
audio processing parameters, and network architecture settings.

Usage:
    python optuna_tuning.py --data_dir /path/to/data --output_dir /path/to/output \
        --n_trials 100 --timeout 3600

Authors: CETACEANS Project
Last Access: December 2024
"""

import os
import json
import math
import pathlib
import argparse
import platform
import optuna
from optuna.trial import TrialState
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


parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for ANIMAL-SPOT")

# Required paths
parser.add_argument("--data_dir", type=str, required=True, help="Path to training data")
parser.add_argument("--output_dir", type=str, required=True, help="Directory for output (models, logs, etc.)")
parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for spectrograms")
parser.add_argument("--noise_dir", type=str, default=None, help="Directory with noise files for augmentation")

# Optuna settings
parser.add_argument("--n_trials", type=int, default=100, help="Number of trials for optimization")
parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds for the study")
parser.add_argument("--study_name", type=str, default="animal_spot_tuning", help="Name of the Optuna study")
parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g., sqlite:///study.db)")
parser.add_argument("--load_if_exists", action="store_true", help="Load existing study if available")

# Fixed parameters (not tuned)
parser.add_argument("--num_classes", type=int, required=True, help="Number of classes")
parser.add_argument("--max_epochs", type=int, default=50, help="Maximum epochs per trial (keep low for fast tuning)")
parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")

# Tuning ranges (can be customized)
parser.add_argument("--lr_min", type=float, default=1e-6, help="Minimum learning rate")
parser.add_argument("--lr_max", type=float, default=1e-3, help="Maximum learning rate")
parser.add_argument("--batch_size_options", type=str, default="8,16,32", help="Batch size options (comma-separated)")
parser.add_argument("--sequence_len_min", type=int, default=300, help="Minimum sequence length (ms)")
parser.add_argument("--sequence_len_max", type=int, default=1500, help="Maximum sequence length (ms)")

ARGS = parser.parse_args()
ARGS.cuda = torch.cuda.is_available() and not ARGS.no_cuda
ARGS.device = torch.device("cuda") if ARGS.cuda else torch.device("cpu")

# Create output directories
os.makedirs(ARGS.output_dir, exist_ok=True)
os.makedirs(os.path.join(ARGS.output_dir, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(ARGS.output_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(ARGS.output_dir, "summaries"), exist_ok=True)

log = Logger("OPTUNA", True, os.path.join(ARGS.output_dir, "logs"))


def get_classes(database):
    """Identify all classes from file names."""
    classes = set()
    for file in database:
        if platform.system() == "Windows":
            class_name = file.split("\\")[-1].split("-", 1)[0]
        else:
            class_name = file.split("/")[-1].split("-", 1)[0]
        classes.add(class_name)
    return sorted(list(classes))


def objective(trial):
    """
    Optuna objective function for hyperparameter optimization.
    Returns the best validation accuracy achieved during training.
    """
    log.info(f"\n{'='*60}")
    log.info(f"Starting Trial {trial.number}")
    log.info(f"{'='*60}")

    # Sample hyperparameters
    backbone = trial.suggest_categorical("backbone", ["resnet", "convnext"])
    pretrained = trial.suggest_categorical("pretrained", [True, False])

    lr = trial.suggest_float("lr", ARGS.lr_min, ARGS.lr_max, log=True)
    batch_size_options = [int(x) for x in ARGS.batch_size_options.split(",")]
    batch_size = trial.suggest_categorical("batch_size", batch_size_options)

    # Audio processing parameters
    sequence_len = trial.suggest_int("sequence_len", ARGS.sequence_len_min, ARGS.sequence_len_max, step=100)
    n_fft = trial.suggest_categorical("n_fft", [512, 1024, 2048, 4096])
    n_freq_bins = trial.suggest_categorical("n_freq_bins", [128, 256, 512])
    freq_compression = trial.suggest_categorical("freq_compression", ["linear", "mel"])

    # Sampling and regularization
    weighted_sampling = trial.suggest_categorical("weighted_sampling", [True, False])
    augmentation = trial.suggest_categorical("augmentation", [True, False])

    # Optimizer parameters
    beta1 = trial.suggest_float("beta1", 0.5, 0.95)

    # Log trial parameters
    log.info(f"Trial {trial.number} parameters:")
    log.info(f"  backbone: {backbone}, pretrained: {pretrained}")
    log.info(f"  lr: {lr:.2e}, batch_size: {batch_size}")
    log.info(f"  sequence_len: {sequence_len}ms, n_fft: {n_fft}")
    log.info(f"  n_freq_bins: {n_freq_bins}, freq_compression: {freq_compression}")
    log.info(f"  weighted_sampling: {weighted_sampling}, augmentation: {augmentation}")

    try:
        # Set up encoder options
        if backbone == "convnext":
            encoderOpts = DefaultConvNextEncoderOpts.copy()
            encoderOpts["pretrained"] = pretrained
        else:
            encoderOpts = DefaultResNetEncoderOpts.copy()
            encoderOpts["pretrained"] = pretrained
            encoderOpts["resnet_size"] = 18

        classifierOpts = DefaultClassifierOpts.copy()
        classifierOpts["num_classes"] = ARGS.num_classes

        dataOpts = DefaultSpecDatasetOps.copy()
        dataOpts["sr"] = 44100
        dataOpts["n_fft"] = n_fft
        dataOpts["n_freq_bins"] = n_freq_bins
        dataOpts["freq_compression"] = freq_compression

        # Compute hop_length to get ~128 time bins
        hop_length = int(sequence_len / 1000 * dataOpts["sr"] / 128)
        dataOpts["hop_length"] = hop_length

        # Compute actual sequence length in frames
        seq_len_frames = int(float(sequence_len) / 1000 * dataOpts["sr"] / dataOpts["hop_length"])

        # Set up data
        split_fracs = {"train": .7, "val": .15, "test": .15}
        input_data = DatabaseCsvSplit(split_fracs, working_dir=ARGS.data_dir, split_per_dir=True)
        audio_files = list(get_audio_files_from_dir(ARGS.data_dir))

        if ARGS.noise_dir:
            noise_files = [str(p) for p in pathlib.Path(ARGS.noise_dir).glob("*.wav")]
        else:
            noise_files = []

        classes = get_classes(database=audio_files)

        if ARGS.num_classes != len(classes):
            raise ValueError(f"num_classes ({ARGS.num_classes}) doesn't match found classes ({len(classes)})")

        # Create datasets
        datasets = {
            split: Dataset(
                file_names=input_data.load(split, audio_files),
                working_dir=ARGS.data_dir,
                cache_dir=ARGS.cache_dir,
                sr=dataOpts["sr"],
                n_fft=dataOpts["n_fft"],
                hop_length=dataOpts["hop_length"],
                n_freq_bins=dataOpts["n_freq_bins"],
                freq_compression=freq_compression,
                f_min=dataOpts["fmin"],
                f_max=dataOpts["fmax"],
                seq_len=seq_len_frames,
                augmentation=augmentation if split == "train" else False,
                noise_files=noise_files,
                dataset_name=split,
                classes=classes,
                min_max_normalize=True
            )
            for split in split_fracs.keys()
        }

        # Create dataloaders
        dataloaders = {}
        for split in split_fracs.keys():
            if split == "train" and weighted_sampling:
                # Compute sample weights
                class_counts = {}
                for idx in range(len(datasets[split])):
                    _, meta = datasets[split][idx]
                    label = meta['call'].item() if hasattr(meta['call'], 'item') else meta['call']
                    class_counts[label] = class_counts.get(label, 0) + 1

                num_samples = len(datasets[split])
                class_weights = {c: num_samples / count for c, count in class_counts.items()}
                max_weight = max(class_weights.values())
                class_weights = {c: w / max_weight for c, w in class_weights.items()}

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
                    batch_size=batch_size,
                    sampler=sampler,
                    num_workers=ARGS.num_workers,
                    drop_last=True,
                    pin_memory=True,
                )
            else:
                dataloaders[split] = torch.utils.data.DataLoader(
                    datasets[split],
                    batch_size=batch_size,
                    shuffle=(split == "train"),
                    num_workers=ARGS.num_workers,
                    drop_last=False if split in ["val", "test"] else True,
                    pin_memory=True,
                )

        # Create model
        if backbone == "convnext":
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

        # Set up training
        prefix = f"trial_{trial.number}"

        if ARGS.num_classes == 2:
            metrics = {
                "accuracy": m.Accuracy(ARGS.device),
                "f1": m.F1Score(ARGS.device),
            }
        else:
            metrics = {
                "accuracy": m.Accuracy(ARGS.device),
                "per_class_accuracy": m.PerClassAccuracy(ARGS.num_classes, ARGS.device)
            }

        trainer = Trainer(
            model=model,
            logger=log,
            prefix=prefix,
            checkpoint_dir=os.path.join(ARGS.output_dir, "checkpoints"),
            summary_dir=os.path.join(ARGS.output_dir, "summaries"),
            n_summaries=4,
            start_scratch=True,  # Always start fresh for each trial
        )

        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))

        lr_patience = max(1, math.ceil(8 / 2))  # epochs_per_eval = 2
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=lr_patience,
            factor=0.5,
            threshold=1e-3,
            threshold_mode="abs",
        )

        loss_fn = nn.CrossEntropyLoss()

        # Train with pruning callback
        class OptunaCallback:
            def __init__(self, trial):
                self.trial = trial
                self.best_accuracy = 0.0

            def __call__(self, epoch, val_accuracy):
                self.best_accuracy = max(self.best_accuracy, val_accuracy)
                self.trial.report(val_accuracy, epoch)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()
                return self.best_accuracy

        callback = OptunaCallback(trial)

        # Modified training with early return for Optuna
        model = model.to(ARGS.device)
        best_accuracy = 0.0

        for epoch in range(ARGS.max_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            for batch_data in dataloaders["train"]:
                inputs = batch_data[0].to(ARGS.device)
                targets = batch_data[1]['call'].to(ARGS.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation phase
            if (epoch + 1) % 2 == 0:  # epochs_per_eval = 2
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch_data in dataloaders["val"]:
                        inputs = batch_data[0].to(ARGS.device)
                        targets = batch_data[1]['call'].to(ARGS.device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs, 1)
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()

                val_accuracy = correct / total if total > 0 else 0.0
                best_accuracy = callback(epoch, val_accuracy)
                lr_scheduler.step(val_accuracy)

                log.info(f"Trial {trial.number} - Epoch {epoch+1}: val_accuracy={val_accuracy:.4f}, best={best_accuracy:.4f}")

        log.info(f"Trial {trial.number} completed with best accuracy: {best_accuracy:.4f}")
        return best_accuracy

    except Exception as e:
        log.error(f"Trial {trial.number} failed with error: {str(e)}")
        raise optuna.TrialPruned()


def main():
    """Main function to run Optuna hyperparameter optimization."""
    log.info("Starting Optuna hyperparameter optimization for ANIMAL-SPOT")
    log.info(f"Device: {ARGS.device}")
    log.info(f"Data directory: {ARGS.data_dir}")
    log.info(f"Output directory: {ARGS.output_dir}")
    log.info(f"Number of trials: {ARGS.n_trials}")
    log.info(f"Timeout: {ARGS.timeout}s" if ARGS.timeout else "No timeout")

    # Create or load study
    if ARGS.storage:
        study = optuna.create_study(
            study_name=ARGS.study_name,
            storage=ARGS.storage,
            direction="maximize",
            load_if_exists=ARGS.load_if_exists,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
    else:
        study = optuna.create_study(
            study_name=ARGS.study_name,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )

    # Run optimization
    study.optimize(
        objective,
        n_trials=ARGS.n_trials,
        timeout=ARGS.timeout,
        show_progress_bar=True,
        gc_after_trial=True  # Help with memory management
    )

    # Print results
    log.info("\n" + "="*60)
    log.info("OPTIMIZATION COMPLETE")
    log.info("="*60)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    log.info(f"Number of finished trials: {len(study.trials)}")
    log.info(f"  Completed: {len(complete_trials)}")
    log.info(f"  Pruned: {len(pruned_trials)}")

    if len(complete_trials) > 0:
        log.info("\nBest trial:")
        trial = study.best_trial
        log.info(f"  Value (accuracy): {trial.value:.4f}")
        log.info("  Params:")
        for key, value in trial.params.items():
            log.info(f"    {key}: {value}")

        # Save best parameters
        best_params_path = os.path.join(ARGS.output_dir, "best_params.json")
        with open(best_params_path, "w") as f:
            json.dump(trial.params, f, indent=2)
        log.info(f"\nBest parameters saved to: {best_params_path}")

        # Generate config file from best parameters
        config_path = os.path.join(ARGS.output_dir, "best_config.conf")
        with open(config_path, "w") as f:
            f.write("# Best hyperparameters found by Optuna\n")
            f.write(f"# Best validation accuracy: {trial.value:.4f}\n\n")
            for key, value in trial.params.items():
                if isinstance(value, bool):
                    f.write(f"{key}={'true' if value else 'false'}\n")
                else:
                    f.write(f"{key}={value}\n")
        log.info(f"Config file saved to: {config_path}")

    log.close()


if __name__ == "__main__":
    main()
