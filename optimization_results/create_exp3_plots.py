#!/usr/bin/env python3
"""Generate specific plots for exp3 (best model)."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11

output_dir = Path(__file__).parent

# exp3 per-class results
exp3_per_class = {
    "K1": 99.5, "K3": 97.6, "K4": 98.1, "K5": 99.7, "K7": 100.0,
    "K10": 98.1, "K12": 96.2, "K13": 100.0, "K14": 83.3,
    "K17": 100.0, "K21": 93.9, "K27": 100.0, "noise": 99.9
}

def plot_exp3_per_class_bar():
    """Bar chart of exp3 per-class accuracy."""
    fig, ax = plt.subplots(figsize=(12, 6))

    classes = list(exp3_per_class.keys())
    values = list(exp3_per_class.values())

    colors = ['#e74c3c' if c in ['K12', 'K14'] else '#2ecc71' for c in classes]
    bars = ax.bar(classes, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.axhline(y=99.49, color='blue', linestyle='--', linewidth=2, label='Overall: 99.49%')
    ax.axhline(y=90, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_title('exp3_max_seq_balanced: Per-Class Test Accuracy\n(Best Model - 99.49% Overall)', fontsize=14, fontweight='bold')
    ax.set_ylim(80, 102)
    ax.legend(loc='lower right')

    # Add value labels
    for bar, val in zip(bars, values):
        ax.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add legend for colors
    ax.text(0.02, 0.98, 'Green = common classes\nRed = rare classes (K12, K14)',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'exp3_per_class_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: exp3_per_class_accuracy.png")


def plot_exp3_confusion_style():
    """Heatmap style visualization of exp3 results."""
    fig, ax = plt.subplots(figsize=(10, 8))

    classes = list(exp3_per_class.keys())
    values = np.array(list(exp3_per_class.values())).reshape(-1, 1)

    # Create heatmap
    im = ax.imshow(values, cmap='RdYlGn', aspect='auto', vmin=80, vmax=100)

    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xticks([0])
    ax.set_xticklabels(['Accuracy'])

    # Add text annotations
    for i, (cls, val) in enumerate(exp3_per_class.items()):
        color = 'white' if val < 90 else 'black'
        ax.text(0, i, f'{val:.1f}%', ha='center', va='center',
                fontsize=12, fontweight='bold', color=color)

    ax.set_title('exp3_max_seq_balanced: Per-Class Accuracy Heatmap\n(Best Model)',
                 fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Accuracy (%)', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'exp3_accuracy_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: exp3_accuracy_heatmap.png")


def plot_exp3_summary_card():
    """Create a summary card for exp3."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, 'BEST MODEL: exp3_max_seq_balanced',
            ha='center', va='top', fontsize=18, fontweight='bold',
            transform=ax.transAxes)

    # Main metrics
    metrics_text = """
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    TEST ACCURACY:  99.49%  (+1.48% vs November)

    RARE CLASS PERFORMANCE:
      • K12: 96.2%  (+10.7% improvement)
      • K14: 83.3%  (+20.8% improvement)

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    CONFIGURATION:
      • Backbone: ResNet18 (from scratch)
      • Sequence Length: 1000ms
      • Loss: Label Smoothing (0.1)
      • Sampling: sqrt_inverse_freq + 1.5x K12/K14 boost
      • Batch Size: 64
      • Learning Rate: 0.0003
      • Scheduler: OneCycle
      • Training Time: 11.2 minutes
      • Best Epoch: 9

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    PER-CLASS RESULTS:
      K1: 99.5%   K3: 97.6%   K4: 98.1%   K5: 99.7%
      K7: 100%    K10: 98.1%  K12: 96.2%  K13: 100%
      K14: 83.3%  K17: 100%   K21: 93.9%  K27: 100%
      noise: 99.9%

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """

    ax.text(0.5, 0.85, metrics_text, ha='center', va='top', fontsize=11,
            transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    plt.savefig(output_dir / 'exp3_summary_card.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: exp3_summary_card.png")


def plot_exp3_radar():
    """Radar/spider chart for exp3 per-class performance."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    classes = list(exp3_per_class.keys())
    values = list(exp3_per_class.values())

    # Close the polygon
    values_closed = values + [values[0]]
    angles = np.linspace(0, 2 * np.pi, len(classes), endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    ax.plot(angles_closed, values_closed, 'o-', linewidth=2, color='#2ecc71', markersize=8)
    ax.fill(angles_closed, values_closed, alpha=0.25, color='#2ecc71')

    # Mark rare classes
    for i, cls in enumerate(classes):
        if cls in ['K12', 'K14']:
            ax.plot(angles[i], values[i], 'o', markersize=12, color='#e74c3c', zorder=5)

    ax.set_xticks(angles)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylim(75, 102)
    ax.set_title('exp3_max_seq_balanced: Per-Class Accuracy\n(Red dots = rare classes K12, K14)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'exp3_radar_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: exp3_radar_chart.png")


if __name__ == "__main__":
    print("Generating exp3 (best model) specific plots...")
    plot_exp3_per_class_bar()
    plot_exp3_confusion_style()
    plot_exp3_summary_card()
    plot_exp3_radar()
    print("\nAll exp3 plots generated!")
