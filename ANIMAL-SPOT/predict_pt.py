#!/usr/bin/env python3
"""
Prediction script for raw audio files using our optimized .pt models.
Performs sliding window classification on full recordings.

Usage:
    python predict_raw_audio.py \
        --model /path/to/best.pt \
        --audio_dir /path/to/audio \
        --output_dir ./predictions
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import zoom
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Predict on raw audio files')
    parser.add_argument('--model', type=str, required=True, help='Path to .pt model checkpoint')
    parser.add_argument('--audio_dir', type=str, required=True, help='Directory with audio files or single file')
    parser.add_argument('--output_dir', type=str, default='./predictions', help='Output directory')
    parser.add_argument('--sequence_len', type=int, default=1000, help='Sequence length in ms')
    parser.add_argument('--hop_len', type=int, default=500, help='Hop length in ms for sliding window')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cuda/cpu). Default: cpu')
    return parser.parse_args()


def load_model(model_path: str, device: str):
    """Load our optimized .pt model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    config = checkpoint.get('config', {})
    classes = checkpoint['classes']
    num_classes = len(classes)

    # Create model (ResNet18 with 1-channel input)
    backbone = config.get('backbone', 'resnet18')
    if backbone == 'resnet18':
        model = models.resnet18(weights=None)
    elif backbone == 'resnet34':
        model = models.resnet34(weights=None)
    else:
        model = models.resnet18(weights=None)

    # Modify for 1-channel input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Load weights - handle torch.compile() prefix
    state_dict = checkpoint['model_state_dict']

    # Remove '_orig_mod.' prefix if present (from torch.compile)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[10:]] = v  # Remove '_orig_mod.' prefix
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    return model, config, classes


def compute_spectrogram(audio: np.ndarray, config: dict) -> np.ndarray:
    """Compute spectrogram matching training configuration."""
    sr = config.get('sr', 44100)
    n_fft = config.get('n_fft', 1024)
    hop_length = config.get('hop_length', 172)
    n_freq_bins = config.get('n_freq_bins', 256)
    fmin = config.get('fmin', 500)
    fmax = config.get('fmax', 10000)

    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    magnitude = magnitude[freq_mask, :]

    if magnitude.shape[0] != n_freq_bins:
        zoom_factor = n_freq_bins / magnitude.shape[0]
        magnitude = zoom(magnitude, (zoom_factor, 1), order=1)

    magnitude = librosa.amplitude_to_db(magnitude, ref=np.max)
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)

    return magnitude.astype(np.float32)


def sliding_window_predict(audio_path: str, model: nn.Module, config: dict,
                          classes: list, args, device: str) -> list:
    """Perform sliding window prediction on an audio file."""

    sr = config.get('sr', 44100)
    seq_samples = int(args.sequence_len * sr / 1000)
    hop_samples = int(args.hop_len * sr / 1000)

    # Load audio
    audio, file_sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

    audio = audio.astype(np.float32)
    n_samples = len(audio)

    # Generate windows
    windows = []
    start_times = []

    start = 0
    while start + seq_samples <= n_samples:
        window = audio[start:start + seq_samples]
        windows.append(window)
        start_times.append(start / sr)
        start += hop_samples

    # Handle last window if needed
    if start < n_samples and (n_samples - start) > seq_samples // 2:
        window = audio[-seq_samples:]
        windows.append(window)
        start_times.append((n_samples - seq_samples) / sr)

    if not windows:
        return []

    # Batch inference
    predictions = []

    for batch_start in range(0, len(windows), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(windows))
        batch_windows = windows[batch_start:batch_end]
        batch_times = start_times[batch_start:batch_end]

        # Compute spectrograms
        specs = []
        for w in batch_windows:
            spec = compute_spectrogram(w, config)
            specs.append(spec)

        # Stack and convert to tensor
        batch_tensor = torch.tensor(np.stack(specs)).unsqueeze(1).to(device)

        with torch.no_grad():
            outputs = model(batch_tensor)
            probs = F.softmax(outputs, dim=1)
            max_probs, pred_classes = probs.max(dim=1)

        for i in range(len(batch_windows)):
            t_start = batch_times[i]
            t_end = t_start + args.sequence_len / 1000
            pred_idx = pred_classes[i].item()
            prob = max_probs[i].item()
            pred_class = classes[pred_idx]

            # Get all class probabilities
            all_probs = {classes[j]: float(probs[i, j]) for j in range(len(classes))}

            predictions.append({
                'start': round(t_start, 3),
                'end': round(t_end, 3),
                'predicted_class': pred_class,
                'confidence': round(prob, 4),
                'all_probabilities': all_probs
            })

    return predictions


def merge_predictions(predictions: list, threshold: float, min_gap: float = 0.3) -> list:
    """Merge consecutive predictions of the same class (legacy)."""
    if not predictions:
        return []
    filtered = [p for p in predictions if p['confidence'] >= threshold and p['predicted_class'].upper() != 'NOISE']
    if not filtered:
        return []
    filtered.sort(key=lambda x: x['start'])
    merged = []
    current = filtered[0].copy()
    for pred in filtered[1:]:
        if (pred['predicted_class'] == current['predicted_class'] and
            pred['start'] - current['end'] <= min_gap):
            current['end'] = pred['end']
            current['confidence'] = max(current['confidence'], pred['confidence'])
        else:
            merged.append(current)
            current = pred.copy()
    merged.append(current)
    return merged


def resolve_overlaps(predictions: list, threshold: float, time_resolution: float = 0.05) -> list:
    """
    Resolve overlapping segments so that every time point belongs to exactly one label.
    For each time slice we assign the label of the segment with highest confidence covering it,
    then merge contiguous same-label intervals. Produces non-overlapping Raven annotations.
    """
    if not predictions:
        return []

    # Filter by threshold (include noise so we have full coverage)
    filtered = [p for p in predictions if p['confidence'] >= threshold]
    if not filtered:
        return []

    # Get time bounds
    t_min = min(p['start'] for p in filtered)
    t_max = max(p['end'] for p in filtered)

    # Discretize timeline and assign winning label per slice
    n_slices = max(1, int((t_max - t_min) / time_resolution) + 1)
    times = np.linspace(t_min, t_max, n_slices)

    labels = []
    confidences = []
    for i in range(len(times) - 1):
        t_center = (times[i] + times[i + 1]) / 2
        best_conf = -1.0
        best_class = 'noise'
        for p in filtered:
            if p['start'] <= t_center < p['end'] and p['confidence'] > best_conf:
                best_conf = p['confidence']
                best_class = p['predicted_class']
        labels.append(best_class)
        confidences.append(best_conf)

    # Merge contiguous same-label intervals
    result = []
    i = 0
    while i < len(labels):
        cls = labels[i]
        conf = confidences[i]
        t_start = times[i]
        t_end = times[i + 1]
        j = i + 1
        while j < len(labels) and labels[j] == cls:
            t_end = times[j + 1]
            conf = max(conf, confidences[j])
            j += 1
        result.append({
            'start': round(t_start, 3),
            'end': round(t_end, 3),
            'predicted_class': cls,
            'confidence': round(conf, 4)
        })
        i = j

    return result


def save_raven_format(predictions: list, output_path: str, audio_filename: str):
    """Save predictions in Raven selection table format."""
    with open(output_path, 'w') as f:
        f.write("Selection\tView\tChannel\tBegin time (s)\tEnd time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tSound type\tConfidence\n")
        for i, pred in enumerate(predictions, 1):
            f.write(f"{i}\tSpectrogram_1\t1\t{pred['start']}\t{pred['end']}\t500\t10000\t{pred['predicted_class']}\t{pred['confidence']}\n")


def main():
    args = parse_args()

    # Setup
    device = args.device if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model}")
    model, config, classes = load_model(args.model, device)
    print(f"Classes: {classes}")
    print(f"Config: sequence_len={config.get('sequence_len')}ms, sr={config.get('sr')}")

    # Find audio files
    audio_path = Path(args.audio_dir)
    if audio_path.is_file():
        audio_files = [audio_path]
    else:
        audio_files = list(audio_path.glob('*.wav')) + list(audio_path.glob('*.WAV'))

    print(f"Found {len(audio_files)} audio files")

    all_results = {}

    for audio_file in tqdm(audio_files, desc="Processing"):
        print(f"\nProcessing: {audio_file.name}")

        try:
            predictions = sliding_window_predict(
                str(audio_file), model, config, classes, args, device
            )

            # Resolve overlaps so labels are non-overlapping (each time point → one label)
            non_overlapping = resolve_overlaps(predictions, args.threshold)

            # Save results
            base_name = audio_file.stem

            # Save detailed JSON (include raw and non-overlapping)
            json_path = output_dir / f"{base_name}_predictions.json"
            with open(json_path, 'w') as f:
                json.dump({
                    'audio_file': str(audio_file),
                    'config': config,
                    'threshold': args.threshold,
                    'raw_predictions': predictions,
                    'merged_predictions': non_overlapping
                }, f, indent=2)

            # Save Raven format (non-overlapping segments only)
            raven_path = output_dir / f"{base_name}_predictions.txt"
            save_raven_format(non_overlapping, str(raven_path), audio_file.name)

            all_results[audio_file.name] = {
                'n_raw_predictions': len(predictions),
                'n_merged_predictions': len(non_overlapping),
                'classes_detected': list(set(p['predicted_class'] for p in non_overlapping))
            }

            print(f"  Found {len(non_overlapping)} call segments: {all_results[audio_file.name]['classes_detected']}")

        except Exception as e:
            print(f"  Error: {e}")
            all_results[audio_file.name] = {'error': str(e)}

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model': args.model,
            'config': config,
            'results': all_results
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
