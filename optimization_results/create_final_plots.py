#!/usr/bin/env python3
"""Generate final plots for best model (exp3) and all experiments."""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# All experiment results
experiments = {
    "exp1_800ms": {"acc": 99.10, "k12": 92.4, "k14": 75.0, "seq": 800},
    "exp2_aggressive": {"acc": 98.78, "k12": 87.3, "k14": 75.0, "seq": 800},
    "exp3_1000ms": {"acc": 99.49, "k12": 96.2, "k14": 83.3, "seq": 1000},
    "exp4_resnet34": {"acc": 99.00, "k12": 93.7, "k14": 58.3, "seq": 800},
    "exp5_extreme": {"acc": 99.04, "k12": 92.4, "k14": 66.7, "seq": 800},
    "exp6_convnext": {"acc": 99.20, "k12": 89.9, "k14": 75.0, "seq": 1000},
    "exp7_1200ms": {"acc": 99.23, "k12": 92.4, "k14": 83.3, "seq": 1200},
    "exp8_1500ms": {"acc": 99.16, "k12": 92.4, "k14": 75.0, "seq": 1500},
    "exp9_convnext_1200": {"acc": 99.04, "k12": 84.8, "k14": 83.3, "seq": 1200},
}

# Best model per-class results
best_per_class = {
    "K1": 99.5, "K3": 97.6, "K4": 98.1, "K5": 99.7, "K7": 100.0,
    "K10": 98.1, "K12": 96.2, "K13": 100.0, "K14": 83.3,
    "K17": 100.0, "K21": 93.9, "K27": 100.0, "noise": 99.9
}

# November baseline per-class (from convnext_stable)
nov_per_class = {
    "K1": 97.5, "K3": 96.4, "K4": 97.2, "K5": 98.7, "K7": 99.5,
    "K10": 96.2, "K12": 85.5, "K13": 90.0, "K14": 62.5,
    "K17": 100.0, "K21": 90.9, "K27": 100.0, "noise": 99.5
}

output_dir = Path(__file__).parent


def plot_per_class_comparison():
    """Compare best model vs November baseline per-class."""
    fig, ax = plt.subplots(figsize=(14, 6))

    classes = list(best_per_class.keys())
    x = np.arange(len(classes))
    width = 0.35

    nov_vals = [nov_per_class.get(c, 0) for c in classes]
    best_vals = [best_per_class[c] for c in classes]

    bars1 = ax.bar(x - width/2, nov_vals, width, label='November (98.01%)',
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, best_vals, width, label='Best Model (99.49%)',
                   color='#2ecc71', alpha=0.8)

    # Highlight rare classes
    for i, c in enumerate(classes):
        if c in ['K12', 'K14']:
            bars1[i].set_edgecolor('red')
            bars1[i].set_linewidth(2)
            bars2[i].set_edgecolor('red')
            bars2[i].set_linewidth(2)

    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Class')
    ax.set_title('Per-Class Accuracy: November Baseline vs Best Model (exp3)\nRare classes K12/K14 highlighted in red border')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(50, 102)

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'best_model_per_class.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: best_model_per_class.png")


def plot_all_experiments():
    """Plot all 9 experiments comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    names = list(experiments.keys())
    accs = [experiments[n]["acc"] for n in names]
    k12s = [experiments[n]["k12"] for n in names]
    k14s = [experiments[n]["k14"] for n in names]

    # Overall accuracy
    colors = ['#2ecc71' if n == 'exp3_1000ms' else '#3498db' for n in names]
    bars = ax1.barh(names, accs, color=colors, alpha=0.8)
    ax1.axvline(x=99.49, color='green', linestyle='--', linewidth=2, label='Best (99.49%)')
    ax1.axvline(x=98.01, color='gray', linestyle=':', linewidth=2, label='Nov baseline (98.01%)')
    ax1.set_xlabel('Test Accuracy (%)')
    ax1.set_title('Overall Accuracy - All Experiments')
    ax1.set_xlim(98.5, 99.6)
    ax1.legend(loc='lower right')

    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{acc:.2f}%', va='center', fontsize=9)

    # Rare class comparison
    x = np.arange(len(names))
    width = 0.35

    bars1 = ax2.bar(x - width/2, k12s, width, label='K12', color='#e74c3c', alpha=0.8)
    bars2 = ax2.bar(x + width/2, k14s, width, label='K14', color='#9b59b6', alpha=0.8)

    ax2.axhline(y=96.2, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=83.3, color='#9b59b6', linestyle='--', linewidth=1, alpha=0.5)

    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Rare Class Accuracy (K12 & K14)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.legend(loc='lower right')
    ax2.set_ylim(50, 105)

    plt.tight_layout()
    plt.savefig(output_dir / 'all_experiments_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: all_experiments_comparison.png")


def plot_sequence_length_analysis():
    """Analyze sequence length impact."""
    fig, ax = plt.subplots(figsize=(10, 6))

    seq_lengths = [800, 1000, 1200, 1500]
    # Best results for each sequence length
    seq_results = {
        800: {"acc": 99.10, "k12": 92.4, "k14": 75.0, "exp": "exp1"},
        1000: {"acc": 99.49, "k12": 96.2, "k14": 83.3, "exp": "exp3"},
        1200: {"acc": 99.23, "k12": 92.4, "k14": 83.3, "exp": "exp7"},
        1500: {"acc": 99.16, "k12": 92.4, "k14": 75.0, "exp": "exp8"},
    }

    accs = [seq_results[s]["acc"] for s in seq_lengths]
    k12s = [seq_results[s]["k12"] for s in seq_lengths]
    k14s = [seq_results[s]["k14"] for s in seq_lengths]

    ax.plot(seq_lengths, accs, 'o-', markersize=10, linewidth=2,
            label='Overall Accuracy', color='#2ecc71')
    ax.plot(seq_lengths, k12s, 's--', markersize=8, linewidth=2,
            label='K12 Accuracy', color='#e74c3c')
    ax.plot(seq_lengths, k14s, '^--', markersize=8, linewidth=2,
            label='K14 Accuracy', color='#9b59b6')

    # Mark optimal
    ax.scatter([1000], [99.49], s=200, c='gold', marker='*', zorder=5,
               label='Optimal (1000ms)')

    ax.set_xlabel('Sequence Length (ms)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Sequence Length Impact on Accuracy\n1000ms is optimal - longer sequences provide no benefit')
    ax.legend(loc='lower right')
    ax.set_xticks(seq_lengths)
    ax.set_ylim(70, 101)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'sequence_length_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: sequence_length_analysis.png")


def plot_architecture_comparison():
    """Compare ResNet18, ResNet34, ConvNeXt."""
    fig, ax = plt.subplots(figsize=(10, 6))

    archs = ['ResNet18\n(exp3)', 'ResNet34\n(exp4)', 'ConvNeXt\n(exp6)']
    accs = [99.49, 99.00, 99.20]
    k12s = [96.2, 93.7, 89.9]
    k14s = [83.3, 58.3, 75.0]

    x = np.arange(len(archs))
    width = 0.25

    bars1 = ax.bar(x - width, accs, width, label='Overall', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, k12s, width, label='K12', color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x + width, k14s, width, label='K14', color='#9b59b6', alpha=0.8)

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Architecture Comparison\nResNet18 is best overall and for rare classes')
    ax.set_xticks(x)
    ax.set_xticklabels(archs)
    ax.legend(loc='lower right')
    ax.set_ylim(50, 105)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'architecture_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: architecture_comparison.png")


def plot_improvement_summary():
    """Summary of improvements from November to December."""
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['Overall', 'K12', 'K14']
    nov_values = [98.01, 85.5, 62.5]
    dec_values = [99.49, 96.2, 83.3]
    improvements = [dec - nov for nov, dec in zip(nov_values, dec_values)]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, nov_values, width, label='November', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, dec_values, width, label='December (Best)', color='#2ecc71', alpha=0.8)

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Improvement Summary: November vs December Best Model')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='lower right')
    ax.set_ylim(50, 105)

    # Add improvement annotations
    for i, (bar1, bar2, imp) in enumerate(zip(bars1, bars2, improvements)):
        mid_x = (bar1.get_x() + bar2.get_x() + bar2.get_width()) / 2
        max_y = max(bar1.get_height(), bar2.get_height())
        ax.annotate(f'+{imp:.1f}%',
                    xy=(mid_x, max_y + 2),
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold', color='green')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: improvement_summary.png")


def create_summary_text():
    """Create text summary."""
    summary = """
CETACEAN CLASSIFICATION - FINAL RESULTS SUMMARY
================================================
Date: December 23, 2025
Total Experiments: 24 (15 November + 9 December)

BEST MODEL: exp3_max_seq_balanced
---------------------------------
Test Accuracy: 99.49%
K12 Accuracy:  96.2% (+10.7% vs November)
K14 Accuracy:  83.3% (+20.8% vs November)

Configuration:
- Backbone: ResNet18 (from scratch)
- Sequence Length: 1000ms
- Loss: Label Smoothing (0.1)
- Sampling: sqrt_inverse_freq with 1.5x K12/K14 boost
- Batch Size: 64
- Learning Rate: 0.0003
- Scheduler: OneCycle

KEY FINDINGS:
1. 1000ms sequence length is optimal (longer sequences did NOT help)
2. Label smoothing (0.1) beats focal loss and cross-entropy
3. sqrt_inverse_freq sampling with moderate boost (1.5x) is optimal
4. ResNet18 outperforms ResNet34 and ConvNeXt
5. Training from scratch beats ImageNet pretraining for spectrograms

WHAT DIDN'T WORK:
- Focal loss (noise class collapse)
- MixUp/SpecAugment (training instability)
- Aggressive class weighting (3x+)
- Longer sequences (1200ms, 1500ms)
- Larger models (ResNet34)

Per-Class Results (Best Model):
K1: 99.5%  K3: 97.6%  K4: 98.1%  K5: 99.7%  K7: 100.0%
K10: 98.1% K12: 96.2% K13: 100.0% K14: 83.3% K17: 100.0%
K21: 93.9% K27: 100.0% noise: 99.9%
"""

    with open(output_dir / 'FINAL_SUMMARY.txt', 'w') as f:
        f.write(summary)
    print("Created: FINAL_SUMMARY.txt")


if __name__ == "__main__":
    print("Generating final plots for best model...")
    plot_per_class_comparison()
    plot_all_experiments()
    plot_sequence_length_analysis()
    plot_architecture_comparison()
    plot_improvement_summary()
    create_summary_text()
    print("\nAll plots generated successfully!")
