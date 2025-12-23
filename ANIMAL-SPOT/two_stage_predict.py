#!/usr/bin/env python3
"""
Module: two_stage_predict.py
Two-stage prediction pipeline for ANIMAL-SPOT.

Stage 1: Binary detection (call vs. noise)
Stage 2: Multi-class classification (call type identification)

This approach reduces false positives by first filtering out noise segments
before attempting call type classification.

Usage:
    python two_stage_predict.py --config config_file.conf

Authors: CETACEANS Project
Last Access: December 2024
"""

import os
import sys
import json
import logging
import argparse
import pathlib
import platform
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import soundfile as sf

from data.audiodataset import DefaultSpecDatasetOps
from data.transforms import (
    Compose,
    LoadAudio,
    PreEmphasis,
    Spectrogram,
    FrequencyCompressionLinear,
    FrequencyCompressionMel,
    AmplitudeToDB,
    MinMaxNormalize,
    Normalize,
)

from models.residual_encoder import ResidualEncoder
from models.convnext_encoder import ConvNextEncoder
from models.classifier import Classifier


def setup_logger(log_level="info"):
    """Set up logging."""
    logger = logging.getLogger('two_stage_predict')
    handler = logging.StreamHandler()

    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    handler.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


class TwoStagePredictor:
    """
    Two-stage prediction pipeline.

    Stage 1: Binary detection model (call vs. noise)
    Stage 2: Multi-class classification model (call types)
    """

    def __init__(
        self,
        binary_model_path: str,
        multiclass_model_path: str,
        device: str = "cuda",
        binary_threshold: float = 0.5,
        multiclass_threshold: float = 0.85,
        logger=None
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.binary_threshold = binary_threshold
        self.multiclass_threshold = multiclass_threshold
        self.logger = logger or logging.getLogger('two_stage_predict')

        # Load models
        self.logger.info(f"Loading binary model from {binary_model_path}")
        self.binary_model, self.binary_opts, self.binary_classes = self._load_model(binary_model_path)

        self.logger.info(f"Loading multiclass model from {multiclass_model_path}")
        self.multiclass_model, self.multiclass_opts, self.multiclass_classes = self._load_model(multiclass_model_path)

        # Set up transforms based on multiclass model options
        self._setup_transforms()

        self.logger.info(f"Two-stage predictor initialized on {self.device}")
        self.logger.info(f"Binary classes: {self.binary_classes}")
        self.logger.info(f"Multiclass classes: {self.multiclass_classes}")

    def _load_model(self, model_path):
        """Load a saved ANIMAL-SPOT model."""
        checkpoint = torch.load(model_path, map_location=self.device)

        backbone_type = checkpoint.get("backbone", "resnet")
        encoder_opts = checkpoint["encoderOpts"]
        classifier_opts = checkpoint["classifierOpts"]
        data_opts = checkpoint.get("dataOpts", DefaultSpecDatasetOps)
        classes = checkpoint.get("classes", {})

        # Create encoder
        if backbone_type == "convnext":
            encoder = ConvNextEncoder(encoder_opts)
        else:
            encoder = ResidualEncoder(encoder_opts)

        # Create classifier
        classifier = Classifier(classifier_opts)

        # Load weights
        encoder.load_state_dict(checkpoint["encoderState"])
        classifier.load_state_dict(checkpoint["classifierState"])

        # Create full model
        model = nn.Sequential(
            OrderedDict([("encoder", encoder), ("classifier", classifier)])
        )
        model = model.to(self.device)
        model.eval()

        return model, data_opts, classes

    def _setup_transforms(self):
        """Set up audio transforms based on model options."""
        opts = self.multiclass_opts

        transforms = [
            LoadAudio(sr=opts.get("sr", 44100)),
            PreEmphasis(opts.get("preemphases", 0.98)),
            Spectrogram(
                n_fft=opts.get("n_fft", 1024),
                hop_length=opts.get("hop_length", 512),
                center=False
            ),
        ]

        # Frequency compression
        freq_compression = opts.get("freq_compression", "linear")
        if freq_compression == "mel":
            transforms.append(FrequencyCompressionMel(
                n_mels=opts.get("n_freq_bins", 256),
                sr=opts.get("sr", 44100),
                f_min=opts.get("fmin", 500),
                f_max=opts.get("fmax", 10000)
            ))
        else:
            transforms.append(FrequencyCompressionLinear(
                n_bins=opts.get("n_freq_bins", 256),
                f_min=opts.get("fmin", 500),
                f_max=opts.get("fmax", 10000),
                sr=opts.get("sr", 44100),
                n_fft=opts.get("n_fft", 1024)
            ))

        transforms.append(AmplitudeToDB(min_level_db=opts.get("min_level_db", -100)))
        transforms.append(MinMaxNormalize())

        self.transform = Compose(transforms)
        self.sr = opts.get("sr", 44100)
        self.hop_length = opts.get("hop_length", 512)

    def predict_file(
        self,
        audio_path: str,
        window_size: float = 0.8,
        hop_size: float = 0.1
    ):
        """
        Predict call types in an audio file using two-stage pipeline.

        Args:
            audio_path: Path to audio file
            window_size: Sliding window size in seconds
            hop_size: Hop size in seconds

        Returns:
            List of predictions: [(start_time, end_time, call_type, probability)]
        """
        self.logger.info(f"Processing file: {audio_path}")

        # Load audio
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # Convert to mono

        duration = len(audio) / sr
        self.logger.info(f"Audio duration: {duration:.2f}s, sample rate: {sr}")

        # Convert window/hop to samples
        window_samples = int(window_size * sr)
        hop_samples = int(hop_size * sr)

        # Stage 1: Binary detection
        self.logger.info("Stage 1: Binary detection (call vs. noise)")
        binary_predictions = []

        position = 0
        while position + window_samples <= len(audio):
            segment = audio[position:position + window_samples]
            start_time = position / sr
            end_time = (position + window_samples) / sr

            # Prepare input
            spec = self._audio_to_spectrogram(segment, sr)
            if spec is None:
                position += hop_samples
                continue

            # Predict
            with torch.no_grad():
                output = self.binary_model(spec)
                probs = torch.softmax(output, dim=1)

                # Find non-noise class probability
                noise_idx = self._get_noise_idx(self.binary_classes)
                if noise_idx is not None:
                    call_prob = 1.0 - probs[0, noise_idx].item()
                else:
                    call_prob = probs[0, 1].item()  # Assume class 1 is "call"

            if call_prob >= self.binary_threshold:
                binary_predictions.append({
                    'start': start_time,
                    'end': end_time,
                    'segment': segment,
                    'call_prob': call_prob
                })

            position += hop_samples

        self.logger.info(f"Stage 1 found {len(binary_predictions)} potential calls")

        # Stage 2: Multi-class classification
        self.logger.info("Stage 2: Multi-class classification")
        final_predictions = []

        for pred in binary_predictions:
            spec = self._audio_to_spectrogram(pred['segment'], sr)
            if spec is None:
                continue

            with torch.no_grad():
                output = self.multiclass_model(spec)
                probs = torch.softmax(output, dim=1)
                max_prob, predicted_class = torch.max(probs, dim=1)

                max_prob = max_prob.item()
                predicted_class = predicted_class.item()

            if max_prob >= self.multiclass_threshold:
                # Get class name
                class_name = self._get_class_name(predicted_class, self.multiclass_classes)

                # Skip if classified as noise
                if class_name.lower() == "noise":
                    continue

                final_predictions.append({
                    'start': pred['start'],
                    'end': pred['end'],
                    'call_type': class_name,
                    'probability': max_prob,
                    'binary_prob': pred['call_prob']
                })

        self.logger.info(f"Stage 2 classified {len(final_predictions)} calls")

        # Merge overlapping predictions
        final_predictions = self._merge_predictions(final_predictions, hop_size)

        return final_predictions

    def _audio_to_spectrogram(self, audio, sr):
        """Convert audio segment to spectrogram tensor."""
        try:
            # Create temporary file-like structure for transform
            spec = self.transform({
                'audio': audio,
                'sr': sr,
                'file_name': 'temp'
            })

            if spec is None:
                return None

            # Add batch dimension
            if isinstance(spec, dict):
                spec = spec.get('spectrogram', spec)

            if isinstance(spec, np.ndarray):
                spec = torch.from_numpy(spec)

            spec = spec.unsqueeze(0).unsqueeze(0).float().to(self.device)
            return spec

        except Exception as e:
            self.logger.warning(f"Failed to create spectrogram: {e}")
            return None

    def _get_noise_idx(self, classes):
        """Get index of noise class."""
        for name, idx in classes.items():
            if name.lower() == "noise":
                return idx
        return None

    def _get_class_name(self, idx, classes):
        """Get class name from index."""
        idx_to_name = {v: k for k, v in classes.items()}
        return idx_to_name.get(idx, f"class_{idx}")

    def _merge_predictions(self, predictions, max_gap):
        """Merge adjacent predictions of the same class."""
        if not predictions:
            return []

        # Sort by start time
        predictions = sorted(predictions, key=lambda x: x['start'])

        merged = [predictions[0]]
        for pred in predictions[1:]:
            prev = merged[-1]

            # Check if same class and close in time
            if (pred['call_type'] == prev['call_type'] and
                pred['start'] - prev['end'] < max_gap):
                # Merge: extend end time and average probability
                merged[-1] = {
                    'start': prev['start'],
                    'end': pred['end'],
                    'call_type': prev['call_type'],
                    'probability': (prev['probability'] + pred['probability']) / 2,
                    'binary_prob': (prev['binary_prob'] + pred['binary_prob']) / 2
                }
            else:
                merged.append(pred)

        return merged

    def save_annotations(self, predictions, output_path, audio_duration=None):
        """Save predictions as Raven-compatible annotation file."""
        with open(output_path, 'w') as f:
            # Header
            f.write("Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\t")
            f.write("Low Freq (Hz)\tHigh Freq (Hz)\tAnnotation\tProbability\tBinary Prob\n")

            fmin = self.multiclass_opts.get("fmin", 500)
            fmax = self.multiclass_opts.get("fmax", 10000)

            for i, pred in enumerate(predictions, 1):
                f.write(f"{i}\tSpectrogram_1\t1\t{pred['start']:.3f}\t{pred['end']:.3f}\t")
                f.write(f"{fmin}\t{fmax}\t{pred['call_type']}\t")
                f.write(f"{pred['probability']:.3f}\t{pred.get('binary_prob', 0):.3f}\n")

        self.logger.info(f"Saved {len(predictions)} annotations to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Two-stage prediction for ANIMAL-SPOT")

    parser.add_argument("--binary_model", type=str, required=True,
                        help="Path to binary detection model (.pk)")
    parser.add_argument("--multiclass_model", type=str, required=True,
                        help="Path to multi-class classification model (.pk)")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory with input audio files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for output annotations")

    parser.add_argument("--binary_threshold", type=float, default=0.5,
                        help="Threshold for binary detection (default: 0.5)")
    parser.add_argument("--multiclass_threshold", type=float, default=0.85,
                        help="Threshold for multi-class classification (default: 0.85)")

    parser.add_argument("--window_size", type=float, default=0.8,
                        help="Sliding window size in seconds (default: 0.8)")
    parser.add_argument("--hop_size", type=float, default=0.1,
                        help="Hop size in seconds (default: 0.1)")

    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--log_level", type=str, default="info",
                        choices=["debug", "info", "warning", "error"])

    args = parser.parse_args()

    # Setup
    logger = setup_logger(args.log_level)
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize predictor
    predictor = TwoStagePredictor(
        binary_model_path=args.binary_model,
        multiclass_model_path=args.multiclass_model,
        device=args.device,
        binary_threshold=args.binary_threshold,
        multiclass_threshold=args.multiclass_threshold,
        logger=logger
    )

    # Find audio files
    input_path = pathlib.Path(args.input_dir)
    audio_extensions = ['.wav', '.WAV', '.mp3', '.flac']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_path.glob(f"*{ext}"))

    logger.info(f"Found {len(audio_files)} audio files")

    # Process each file
    for audio_file in audio_files:
        try:
            predictions = predictor.predict_file(
                str(audio_file),
                window_size=args.window_size,
                hop_size=args.hop_size
            )

            # Save annotations
            output_file = pathlib.Path(args.output_dir) / f"{audio_file.stem}_two_stage.txt"
            predictor.save_annotations(predictions, str(output_file))

        except Exception as e:
            logger.error(f"Failed to process {audio_file}: {e}")
            continue

    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
