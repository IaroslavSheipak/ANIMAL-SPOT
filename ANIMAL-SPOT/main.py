#!/usr/bin/env python3

"""
Module: main.py
Authors: Christian Bergler, Hendrik Schroeter
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 27.12.2022
"""

import os
import json
import math
import pathlib
import argparse
import platform
import utils.metrics as m

import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR

# CUDA optimizations for faster training
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Auto-tune cuDNN for faster convolutions
    # Use new TF32 API (PyTorch 2.0+) to avoid deprecation warnings
    # 'high' enables TF32 for better performance on Ampere+ GPUs
    torch.set_float32_matmul_precision('high')

from data.audiodataset import (
    get_audio_files_from_dir,
    get_broken_audio_files,
    DatabaseCsvSplit,
    DefaultSpecDatasetOps,
    Dataset,
)

from trainer import Trainer
from utils.logging import Logger
from collections import OrderedDict
from models.residual_encoder import DefaultEncoderOpts
from models.residual_encoder import ResidualEncoder as Encoder
from models.convnext_encoder import DefaultEncoderOpts as DefaultConvNextEncoderOpts
from models.convnext_encoder import ConvNextEncoder
from models.classifier import Classifier, DefaultClassifierOpts

parser = argparse.ArgumentParser()

"""
Convert string to boolean.
"""
def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

parser.add_argument(
    "-d",
    "--debug",
    dest="debug",
    action="store_true",
    help="Log additional training and model information.",
)

""" Directory parameters """
parser.add_argument(
    "--data_dir",
    type=str,
    help="The path to the dataset directory.",
)

parser.add_argument(
    "--cache_dir",
    type=str,
    default=None,
    help="The path to the dataset directory.",
)

parser.add_argument(
    "--model_dir",
    type=str,
    help="The directory where the model will be stored.",
)

parser.add_argument(
    "--checkpoint_dir",
    type=str,
    help="The directory where the checkpoints will be stored.",
)

parser.add_argument(
    "--log_dir", type=str, default=None, help="The directory to store the logs."
)

parser.add_argument(
    "--summary_dir",
    type=str,
    help="The directory to store the tensorboard summaries.",
)

parser.add_argument(
    "--noise_dir",
    type=str,
    default=None,
    help="Path to a directory with noise files used for data augmentation.",
)

""" Training parameters """
parser.add_argument(
    "--start_from_scratch",
    dest="start_scratch",
    action="store_true",
    help="Start taining from scratch, i.e. do not use checkpoint to restore.",
)

parser.add_argument(
    "--max_train_epochs", type=int, default=500, help="The number of epochs to train for the classifier."
)

parser.add_argument(
    "--jit_save",
    dest="jit_save",
    action="store_true",
    help="Save model via torch.jit save functionality.",
)

parser.add_argument(
    "--epochs_per_eval",
    type=int,
    default=2,
    help="The number of batches to run in between evaluations.",
)

parser.add_argument(
    "--batch_size", type=int, default=1, help="The number of images per batch."
)

parser.add_argument(
    "--num_workers", type=int, default=4, help="Number of workers used in data-loading"
)

parser.add_argument(
    "--no_cuda",
    dest="cuda",
    action="store_false",
    help="Do not use cuda to train model.",
)

parser.add_argument(
    "--lr",
    "--learning_rate",
    type=float,
    default=1e-5,
    help="Initial learning rate. Will get multiplied by the batch size.",
)

parser.add_argument(
    "--beta1", type=float, default=0.5, help="beta1 for the adam optimizer."
)

parser.add_argument(
    "--lr_patience_epochs",
    type=int,
    default=8,
    help="Decay the learning rate after N/epochs_per_eval epochs without any improvements on the validation set.",
)

parser.add_argument(
    "--lr_decay_factor",
    type=float,
    default=0.5,
    help="Decay factor to apply to the learning rate.",
)

parser.add_argument(
    "--early_stopping_patience_epochs",
    metavar="N",
    type=int,
    default=20,
    help="Early stopping (stop training) after N/epochs_per_eval epochs without any improvements on the validation set.",
)

""" Class balancing parameters """
parser.add_argument(
    "--weighted_sampling",
    action="store_true",
    help="Enable weighted sampling to balance class distribution during training.",
)

parser.add_argument(
    "--sampling_strategy",
    type=str,
    default="sqrt_inverse_freq",
    choices=["inverse_freq", "sqrt_inverse_freq", "effective_num"],
    help="Strategy for computing sample weights: inverse_freq, sqrt_inverse_freq (default), effective_num.",
)

parser.add_argument(
    "--rare_class_boost",
    type=float,
    default=1.5,
    help="Boost factor for rare classes (classes with < 5%% of samples). Default: 1.5",
)

""" Loss function parameters """
parser.add_argument(
    "--label_smoothing",
    type=float,
    default=0.0,
    help="Label smoothing factor (0.0 = no smoothing, 0.1 = recommended). Default: 0.0",
)

""" Scheduler parameters """
parser.add_argument(
    "--scheduler",
    type=str,
    default="plateau",
    choices=["plateau", "onecycle"],
    help="Learning rate scheduler: plateau (ReduceLROnPlateau) or onecycle (OneCycleLR). Default: plateau",
)

""" Input parameters """
parser.add_argument(
    "--filter_broken_audio", action="store_true", help="Filter files which are below a minimum loudness of 1e-3 (float32)."
)

parser.add_argument(
    "--sequence_len", type=int, default=1000, help="Sequence length in ms."
)

parser.add_argument(
    "--freq_compression",
    type=str,
    default="linear",
    help="Frequency compression to reduce GPU memory usage. "
    "Options: `'linear'` (default), '`mel`', `'mfcc'`",
)

parser.add_argument(
    "--sr",
    type=int,
    default=44100,
    help="Sampling Rate in order to resample input data.",
)

parser.add_argument(
    "--fmin",
    type=int,
    default=500,
    help="Minimum frequency covered within the spectrogram (lower frequencies are ignored).",
)

parser.add_argument(
    "--fmax",
    type=int,
    default=10000,
    help="Maximum frequency covered within the spectrogram (higher frequencies are ignored).",
)

parser.add_argument(
    "--n_freq_bins",
    type=int,
    default=256,
    help="Number of frequency bins after compression.",
)

parser.add_argument(
    "--n_fft",
    type=int,
    default=1024,
    help="FFT size.")

parser.add_argument(
    "--hop_length",
    type=int,
    default=172,
    help="FFT hop length.")

parser.add_argument(
    "--augmentation",
    action="store_true",
    help="Whether to augment the input data. Validation and test data will not be augmented.",
)

""" Network parameters """
parser.add_argument(
    "--resnet", dest="resnet_size", type=int, default=18, help="ResNet size"
)

parser.add_argument(
    "--backbone",
    type=str,
    default="resnet",
    choices=["resnet", "convnext"],
    help="Backbone architecture to use (resnet or convnext, default: resnet)"
)

parser.add_argument(
    "--pretrained",
    action="store_true",
    help="Use ImageNet pretrained weights for the backbone (transfer learning)"
)

parser.add_argument(
    "--conv_kernel_size", nargs="*", type=int, help="Initial convolution kernel size."
)

parser.add_argument(
    "--num_classes", type=int, default=2, help="Number of classes needed to be classified."
)

parser.add_argument(
    "--min_max_norm",
    dest="min_max_norm",
    action="store_true",
    help="activates min-max normalization instead of default 0/1-dB-normalization.",
)

parser.add_argument(
    "--max_pool",
    type=int,
    default=None,
    help="Use max pooling after the initial convolution layer.",
)

ARGS = parser.parse_args()
ARGS.cuda = torch.cuda.is_available() and ARGS.cuda
ARGS.device = torch.device("cuda") if ARGS.cuda else torch.device("cpu")

if ARGS.conv_kernel_size is not None and len(ARGS.conv_kernel_size):
    ARGS.conv_kernel_size = ARGS.conv_kernel_size[0]

log = Logger("TRAIN", ARGS.debug, ARGS.log_dir)

"""
Get audio all audio files from the given data directory except they are broken.
"""
def get_audio_files():
    if input_data.can_load_from_csv():
        audio_files = list(get_audio_files_from_dir(ARGS.data_dir))
        log.info("Found csv files in {}".format(ARGS.data_dir))
    else:
        log.debug("Searching for audio files in {}".format(ARGS.data_dir))
        if ARGS.filter_broken_audio:
            data_dir_ = pathlib.Path(ARGS.data_dir)
            audio_files = get_audio_files_from_dir(ARGS.data_dir)
            log.debug("Moving possibly broken audio files to bkp:")
            broken_files = get_broken_audio_files(audio_files, ARGS.data_dir)
            for f in broken_files:
                log.debug(f)
                bkp_dir = data_dir_.joinpath(f).parent.joinpath("bkp")
                bkp_dir.mkdir(exist_ok=True)
                f = pathlib.Path(f)
                data_dir_.joinpath(f).rename(bkp_dir.joinpath(f.name))
        audio_files = list(get_audio_files_from_dir(ARGS.data_dir))
        log.info("Found {} audio files for training.".format(len(audio_files)))
        if len(audio_files) == 0:
            log.close()
            exit(1)
    return audio_files

"""
Save the trained model and corresponding options.
"""
def save_model(model, encoder, encoderOpts, classifier, classifierOpts, dataOpts, path, class_dist_dict, use_jit=False, min_max=False, backbone="resnet"):
    model = model.cpu()
    encoder = encoder.cpu()
    classifier = classifier.cpu()
    encoder_state_dict = encoder.state_dict()
    classifier_state_dict = classifier.state_dict()

    save_dict = {
        "backbone": backbone,
        "encoderOpts": encoderOpts,
        "classifierOpts": classifierOpts,
        "dataOpts": dataOpts,
        "encoderState": encoder_state_dict,
        "classifierState": classifier_state_dict,
        "classes": class_dist_dict
    }

    if min_max:
        norm = "{min_max : " + str(True) + ", min_db : "+str(None)+", ref_db : "+str(None)+"}"
    else:
        norm = "{min_max : " + str(False) + ", min_db : "+str(dataOpts["min_level_db"])+", ref_db : "+str(dataOpts["ref_level_db"])+"}"

    transform_dict = {
    "load_audio" : "{sr: "+str(dataOpts["sr"])+", mono : "+str(True)+"}",
    "pre_emph" : "{pre_empf_factor : "+str(dataOpts["preemphases"])+"}",
    "spectrogram" : "{fft : "+str(dataOpts["n_fft"])+", hop : "+str(dataOpts["hop_length"])+", center : "+str(False)+"}",
    "freq_compression" : "{type : "+str(dataOpts["freq_compression"])+", bins : "+str(dataOpts["n_freq_bins"])+", fmin : "+str(dataOpts["fmin"])+", fmax : "+str(dataOpts["fmax"])+"}",
    "amplitude_to_decibel" : "{min_db : "+str(dataOpts["min_level_db"])+"}",
    "normalize" : norm,
    "class_info" : "{num_class : "+str(len(class_dist_dict))+", name_class : "+str(class_dist_dict.__str__())+"}",
    "seg_size" : "{size_ms : "+str(ARGS.sequence_len)+"}"
    }

    if not os.path.isdir(ARGS.model_dir):
        os.makedirs(ARGS.model_dir)
    if use_jit:
        example = torch.rand(1, 1, 128, 256)
        extra_files = {}
        extra_files['dataOpts'] = dataOpts.__str__()
        extra_files['encoderOpts'] = encoderOpts.__str__()
        extra_files['classifierOpts'] = classifierOpts.__str__()
        extra_files['encoderState'] = encoder_state_dict.__str__()
        extra_files['classifierState'] = classifier_state_dict.__str__()
        extra_files['transforms'] = transform_dict.__str__()
        extra_files['classes'] = class_dist_dict.__str__()
        model = torch.jit.trace(model, example)
        torch.jit.save(model, path, _extra_files=extra_files)
        log.debug("Model successfully saved via torch jit: " + str(path))
    else:
        torch.save(save_dict, path)
        log.debug("Model successfully saved via torch save: " + str(path))

"""
Identifing all classes according to the file namings (class-label_id_year_tape_start_end).
"""
def get_classes(database):
    classes = set()
    for file in database:
        if platform.system() == "Windows":
            class_name = file.split("\\")[-1].split("-", 1)[0]
        else:
            class_name = file.split("/")[-1].split("-", 1)[0]
        classes.add(class_name)
    return sorted(list(classes))


"""
Compute sample weights for weighted sampling based on class distribution.
Uses class_dist_dict from dataset to avoid loading audio files.
"""
def compute_sample_weights(dataset, strategy="sqrt_inverse_freq", rare_class_boost=1.5):
    # Get class distribution from dataset (already computed during init)
    class_dist_dict = dataset.class_dist_dict  # Maps class_name -> class_idx

    # Count samples per class from filenames (fast, no audio loading)
    class_counts = {idx: 0 for idx in class_dist_dict.values()}
    sample_labels = []

    for file_name in dataset.file_names:
        # Extract class name from filename (CLASS-label_id_year_tape_start_end.wav)
        if platform.system() == "Windows":
            basename = file_name.split("\\")[-1]
        else:
            basename = file_name.split("/")[-1]
        class_name = basename.split("-", 1)[0]
        class_idx = class_dist_dict.get(class_name, 0)
        sample_labels.append(class_idx)
        class_counts[class_idx] = class_counts.get(class_idx, 0) + 1

    total_samples = len(dataset)
    num_classes = len(class_counts)

    # Compute class weights based on strategy
    if strategy == "inverse_freq":
        class_weights = {c: total_samples / (num_classes * max(count, 1)) for c, count in class_counts.items()}
    elif strategy == "sqrt_inverse_freq":
        class_weights = {c: math.sqrt(total_samples / max(count, 1)) for c, count in class_counts.items()}
    elif strategy == "effective_num":
        beta = 0.9999
        class_weights = {c: (1 - beta) / (1 - math.pow(beta, max(count, 1))) for c, count in class_counts.items()}
    else:
        class_weights = {c: 1.0 for c in class_counts.keys()}

    # Apply rare class boost (for classes with < 5% of samples)
    threshold = total_samples * 0.05
    for c, count in class_counts.items():
        if count < threshold:
            class_weights[c] *= rare_class_boost

    # Normalize weights
    max_weight = max(class_weights.values()) if class_weights else 1.0
    class_weights = {c: w / max_weight for c, w in class_weights.items()}

    # Create sample weights
    sample_weights = [class_weights[label] for label in sample_labels]

    return sample_weights, class_counts, class_weights


"""
Main function to compute data preprocessing, network training, evaluation, and saving.
"""
if __name__ == "__main__":

    encoderOpts = DefaultEncoderOpts
    classifierOpts = DefaultClassifierOpts
    dataOpts = DefaultSpecDatasetOps

    for arg, value in vars(ARGS).items():
        if arg in encoderOpts and value is not None:
            encoderOpts[arg] = value
        if arg in classifierOpts and value is not None:
            classifierOpts[arg] = value
        if arg in dataOpts and value is not None:
            dataOpts[arg] = value

    ARGS.lr *= ARGS.batch_size

    patience_lr = math.ceil(ARGS.lr_patience_epochs / ARGS.epochs_per_eval)

    patience_lr = int(max(1, patience_lr))

    log.debug("dataOpts: " + json.dumps(dataOpts, indent=4))

    sequence_len = int(
        float(ARGS.sequence_len) / 1000 * dataOpts["sr"] / dataOpts["hop_length"]
    )
    log.debug("Number of spectrogram time-steps (input size = time-steps x frequency-bins) : {}".format(sequence_len))
    input_shape = (ARGS.batch_size, 1, dataOpts["n_freq_bins"], sequence_len)

    log.info("Setting up model with backbone: {}".format(ARGS.backbone))

    # Select encoder based on backbone argument
    if ARGS.backbone == "convnext":
        encoderOpts = DefaultConvNextEncoderOpts.copy()
        encoderOpts["input_channels"] = 1
        encoderOpts["pretrained"] = ARGS.pretrained
        encoder = ConvNextEncoder(encoderOpts)
        encoder_out_ch = encoder.output_channels  # 768 for ConvNeXt Tiny
        log.info("Using ConvNeXt Tiny encoder (pretrained={})".format(ARGS.pretrained))
    else:
        encoder = Encoder(encoderOpts)
        encoder_out_ch = 512 * encoder.block_type.expansion
        log.info("Using ResNet-{} encoder".format(ARGS.resnet_size))

    log.debug("Encoder: " + str(encoder))
    log.debug("Encoder output channels: {}".format(encoder_out_ch))

    classifierOpts["input_channels"] = encoder_out_ch
    classifier = Classifier(classifierOpts)
    log.debug("Classifier: " + str(classifier))

    split_fracs = {"train": .7, "val": .15, "test": .15}
    input_data = DatabaseCsvSplit(
        split_fracs, working_dir=ARGS.data_dir, split_per_dir=True
    )

    audio_files = get_audio_files()

    if ARGS.noise_dir:
        noise_files = [str(p) for p in pathlib.Path(ARGS.noise_dir).glob("*.wav")]
    else:
        noise_files = []

    classes = get_classes(database=audio_files)

    if classifierOpts["num_classes"] == len(classes):
        log.info("Model predict " + str(len(classes)) + " classes")
    else:
        raise Exception("amount of automatically identified classes do not match amount of chosen classes!")

    datasets = {
        split: Dataset(
            file_names=input_data.load(split, audio_files),
            working_dir=ARGS.data_dir,
            cache_dir=ARGS.cache_dir,
            sr=dataOpts["sr"],
            n_fft=dataOpts["n_fft"],
            hop_length=dataOpts["hop_length"],
            n_freq_bins=dataOpts["n_freq_bins"],
            freq_compression=ARGS.freq_compression,
            f_min=dataOpts["fmin"],
            f_max=dataOpts["fmax"],
            seq_len=sequence_len,
            augmentation=ARGS.augmentation if split == "train" else False,
            noise_files=noise_files,
            dataset_name=split,
            classes=classes,
            min_max_normalize=ARGS.min_max_norm
        )
        for split in split_fracs.keys()
    }

    # Create dataloaders with optional weighted sampling
    # Configure DataLoader kwargs for faster data loading
    loader_kwargs = {
        'batch_size': ARGS.batch_size,
        'num_workers': ARGS.num_workers,
        'pin_memory': True,
    }
    # Enable persistent workers and prefetching for faster data loading when using multiple workers
    if ARGS.num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 4

    dataloaders = {}
    for split in split_fracs.keys():
        if split == "train" and ARGS.weighted_sampling:
            log.info("Computing sample weights for weighted sampling...")
            sample_weights, class_counts, class_weights = compute_sample_weights(
                datasets[split],
                strategy=ARGS.sampling_strategy,
                rare_class_boost=ARGS.rare_class_boost
            )
            log.info(f"Class distribution: {class_counts}")
            log.info(f"Class weights: {class_weights}")

            sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(sample_weights),
                num_samples=len(datasets[split]),
                replacement=True
            )
            dataloaders[split] = torch.utils.data.DataLoader(
                datasets[split],
                sampler=sampler,
                drop_last=True,
                **loader_kwargs,
            )
        else:
            dataloaders[split] = torch.utils.data.DataLoader(
                datasets[split],
                shuffle=(split == "train"),
                drop_last=False if split in ["val", "test"] else True,
                **loader_kwargs,
            )

    model = nn.Sequential(
        OrderedDict([("encoder", encoder), ("classifier", classifier)])
    )

    if classifierOpts["num_classes"] == 2:
        prefix = "binary_classifier"
        metrics = {
            "tp": m.TruePositives(ARGS.device),
            "tn": m.TrueNegatives(ARGS.device),
            "fp": m.FalsePositives(ARGS.device),
            "fn": m.FalseNegatives(ARGS.device),
            "accuracy": m.Accuracy(ARGS.device),
            "f1": m.F1Score(ARGS.device),
            "precision": m.Precision(ARGS.device),
            "TPR": m.Recall(ARGS.device),
            "FPR": m.FPR(ARGS.device),
        }
    elif classifierOpts["num_classes"] > 2:
        prefix = "multi_class_" + str(classifierOpts["num_classes"]) + "_classifier"
        metrics = {
            "accuracy": m.Accuracy(ARGS.device),
            "per_class_accuracy": m.PerClassAccuracy(classifierOpts["num_classes"], ARGS.device)
        }
    else:
        raise Exception("not a valid number of classes")

    trainer = Trainer(
        model=model,
        logger=log,
        prefix=prefix,
        checkpoint_dir=ARGS.checkpoint_dir,
        summary_dir=ARGS.summary_dir,
        n_summaries=4,
        start_scratch=ARGS.start_scratch,
    )

    optimizer = optim.Adam(
        model.parameters(), lr=ARGS.lr, betas=(ARGS.beta1, 0.999)
    )

    metric_mode = "max"

    # Set up learning rate scheduler
    if ARGS.scheduler == "onecycle":
        steps_per_epoch = len(dataloaders["train"])
        total_steps = steps_per_epoch * ARGS.max_train_epochs
        # For OneCycleLR, use max_lr = 10x the base LR (before batch scaling)
        # ARGS.lr was already multiplied by batch_size, so divide it back
        base_lr = ARGS.lr / ARGS.batch_size
        max_lr = base_lr * 10  # 3e-4 * 10 = 3e-3
        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=10,
            final_div_factor=100
        )
        log.info(f"Using OneCycleLR scheduler (max_lr={max_lr}, total_steps={total_steps})")
    else:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=metric_mode,
            patience=patience_lr,
            factor=ARGS.lr_decay_factor,
            threshold=1e-3,
            threshold_mode="abs",
        )
        log.info("Using ReduceLROnPlateau scheduler")

    # Set up loss function with optional label smoothing
    if ARGS.label_smoothing > 0:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=ARGS.label_smoothing)
        log.info(f"Using CrossEntropyLoss with label_smoothing={ARGS.label_smoothing}")
    else:
        loss_fn = nn.CrossEntropyLoss()
        log.info("Using CrossEntropyLoss without label smoothing")

    model = trainer.fit(
        dataloaders["train"],
        dataloaders["val"],
        dataloaders["test"],
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        n_epochs=ARGS.max_train_epochs,
        val_interval=ARGS.epochs_per_eval,
        patience_early_stopping=ARGS.early_stopping_patience_epochs,
        device=ARGS.device,
        metrics=metrics,
        val_metric="accuracy",
        val_metric_mode=metric_mode,
        scheduler_step_per_batch=(ARGS.scheduler == "onecycle"),
    )

    encoder = model.encoder

    classifier = model.classifier

    path = os.path.join(ARGS.model_dir, "ANIMAL-SPOT.pk")

    class_dist_dict = datasets["train"].class_dist_dict

    save_model(model, encoder, encoderOpts, classifier, classifierOpts, dataOpts, path, class_dist_dict, use_jit=ARGS.jit_save, min_max=ARGS.min_max_norm, backbone=ARGS.backbone)

    log.close()
