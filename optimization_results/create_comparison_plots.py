#!/usr/bin/env python3
"""
Create comparison plots for cetacean classification experiments.
Compares ResNet18 vs ResNet34 vs ConvNeXt and analyzes rare class performance.
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Results directories
ULTIMATE_RESULTS = Path("/mnt/c/Users/Iaroslav/CETACEANS/ultimate_results")
K12K14_RESULTS = Path("/mnt/c/Users/Iaroslav/CETACEANS/k12_k14_results")
OUTPUT_DIR = Path("/mnt/c/Users/Iaroslav/CETACEANS/comprehensive_results")

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)


def load_all_results():
    """Load all experiment results."""
    results = {}

    # Load ultimate results
    ultimate_run = ULTIMATE_RESULTS / "run_20251222_170746"
    if ultimate_run.exists():
        for exp_dir in sorted(ultimate_run.iterdir()):
            if exp_dir.is_dir():
                results_file = exp_dir / "results.json"
                if results_file.exists():
                    with open(results_file) as f:
                        results[exp_dir.name] = json.load(f)

    # Load ConvNeXt final results
    convnext_dir = ULTIMATE_RESULTS / "final_convnext_20251223_025532"
    if convnext_dir.exists():
        results_file = convnext_dir / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                results["final_convnext"] = json.load(f)

    # Load K12/K14 optimization results if available
    if K12K14_RESULTS.exists():
        for exp_dir in sorted(K12K14_RESULTS.iterdir()):
            if exp_dir.is_dir():
                results_file = exp_dir / "results.json"
                if results_file.exists():
                    with open(results_file) as f:
                        results[f"k12k14_{exp_dir.name}"] = json.load(f)

    return results


def plot_overall_comparison(results):
    """Plot overall test accuracy comparison."""
    # Filter experiments with valid test accuracy
    valid_results = {k: v for k, v in results.items()
                     if v.get("test_acc", 0) > 0.5}  # Filter collapsed experiments

    # Sort by test accuracy
    sorted_results = sorted(valid_results.items(),
                           key=lambda x: x[1]["test_acc"], reverse=True)

    fig, ax = plt.subplots(figsize=(14, 8))

    names = [r[0].replace("_", "\n") for r in sorted_results[:12]]  # Top 12
    accs = [r[1]["test_acc"] * 100 for r in sorted_results[:12]]

    # Color by backbone type
    colors = []
    for name, _ in sorted_results[:12]:
        if "convnext" in name.lower():
            colors.append("#2ecc71")  # Green for ConvNeXt
        elif "resnet34" in name.lower():
            colors.append("#e74c3c")  # Red for ResNet34
        elif "k12k14" in name.lower():
            colors.append("#9b59b6")  # Purple for K12/K14 optimization
        else:
            colors.append("#3498db")  # Blue for ResNet18

    bars = ax.barh(names, accs, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{acc:.2f}%', va='center', fontsize=10, fontweight='bold')

    # Add baseline reference line
    ax.axvline(x=97.79, color='red', linestyle='--', linewidth=2, label='Baseline (97.79%)')

    ax.set_xlabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Overall Test Accuracy Comparison\n(Cetacean Classification)', fontsize=14, fontweight='bold')
    ax.set_xlim(90, 101)
    ax.legend(loc='lower right')
    ax.invert_yaxis()

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='black', label='ResNet18'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='ResNet34'),
        Patch(facecolor='#2ecc71', edgecolor='black', label='ConvNeXt'),
        Patch(facecolor='#9b59b6', edgecolor='black', label='K12/K14 Optimization'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "overall_accuracy_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'overall_accuracy_comparison.png'}")


def plot_rare_class_comparison(results):
    """Plot K12 and K14 (rare class) performance comparison."""
    # Extract K12/K14 performance from all experiments
    exp_data = []
    for name, data in results.items():
        if "test_per_class" in data and data.get("test_acc", 0) > 0.5:
            per_class = data["test_per_class"]
            k12 = per_class.get("K12", 0) * 100
            k14 = per_class.get("K14", 0) * 100
            overall = data["test_acc"] * 100
            noise = per_class.get("noise", 0) * 100

            exp_data.append({
                "name": name,
                "K12": k12,
                "K14": k14,
                "overall": overall,
                "noise": noise,
                "backbone": "convnext" if "convnext" in name.lower() else
                           ("resnet34" if "resnet34" in name.lower() else "resnet18")
            })

    # Sort by K12+K14 average
    exp_data = sorted(exp_data, key=lambda x: (x["K12"] + x["K14"]) / 2, reverse=True)[:10]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # K12 Performance
    names_k12 = [d["name"].replace("_", "\n")[:30] for d in exp_data]
    k12_vals = [d["K12"] for d in exp_data]
    colors_k12 = ["#2ecc71" if d["backbone"] == "convnext" else
                  "#e74c3c" if d["backbone"] == "resnet34" else "#3498db"
                  for d in exp_data]

    bars1 = ax1.barh(names_k12, k12_vals, color=colors_k12, edgecolor='black')
    ax1.axvline(x=64.47, color='red', linestyle='--', linewidth=2, label='Baseline K12 (64.47%)')
    ax1.set_xlabel('K12 Accuracy (%)', fontsize=11)
    ax1.set_title('K12 (Rare Class) Performance\n273 samples, 1.9% of dataset', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 105)
    ax1.legend(loc='lower right')
    ax1.invert_yaxis()

    for bar, val in zip(bars1, k12_vals):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')

    # K14 Performance
    k14_vals = [d["K14"] for d in exp_data]

    bars2 = ax2.barh(names_k12, k14_vals, color=colors_k12, edgecolor='black')
    ax2.axvline(x=68.75, color='red', linestyle='--', linewidth=2, label='Baseline K14 (68.75%)')
    ax2.set_xlabel('K14 Accuracy (%)', fontsize=11)
    ax2.set_title('K14 (Most Rare Class) Performance\n70 samples, 0.5% of dataset', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 105)
    ax2.legend(loc='lower right')
    ax2.invert_yaxis()

    for bar, val in zip(bars2, k14_vals):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rare_class_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'rare_class_comparison.png'}")


def plot_architecture_comparison(results):
    """Compare ResNet18 vs ResNet34 vs ConvNeXt architectures."""
    # Group by architecture
    arch_results = {"ResNet18": [], "ResNet34": [], "ConvNeXt": []}

    for name, data in results.items():
        if data.get("test_acc", 0) > 0.5:  # Only valid experiments
            backbone = data.get("config", {}).get("backbone", "")
            if "convnext" in backbone:
                arch_results["ConvNeXt"].append((name, data))
            elif "resnet34" in backbone:
                arch_results["ResNet34"].append((name, data))
            else:
                arch_results["ResNet18"].append((name, data))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Best Overall Accuracy per Architecture
    ax1 = axes[0, 0]
    arch_best = {}
    for arch, exps in arch_results.items():
        if exps:
            best = max(exps, key=lambda x: x[1]["test_acc"])
            arch_best[arch] = best[1]["test_acc"] * 100
        else:
            arch_best[arch] = 0

    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    bars = ax1.bar(arch_best.keys(), arch_best.values(), color=colors, edgecolor='black', linewidth=2)
    ax1.axhline(y=97.79, color='red', linestyle='--', linewidth=2, label='Baseline (97.79%)')
    ax1.set_ylabel('Best Test Accuracy (%)', fontsize=11)
    ax1.set_title('Best Overall Accuracy by Architecture', fontsize=12, fontweight='bold')
    ax1.set_ylim(95, 100)
    ax1.legend()

    for bar, val in zip(bars, arch_best.values()):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 2. Best K12 per Architecture
    ax2 = axes[0, 1]
    arch_k12 = {}
    for arch, exps in arch_results.items():
        if exps:
            best = max(exps, key=lambda x: x[1].get("test_per_class", {}).get("K12", 0))
            arch_k12[arch] = best[1].get("test_per_class", {}).get("K12", 0) * 100
        else:
            arch_k12[arch] = 0

    bars = ax2.bar(arch_k12.keys(), arch_k12.values(), color=colors, edgecolor='black', linewidth=2)
    ax2.axhline(y=64.47, color='red', linestyle='--', linewidth=2, label='Baseline K12 (64.47%)')
    ax2.set_ylabel('Best K12 Accuracy (%)', fontsize=11)
    ax2.set_title('Best K12 (Rare Class) Accuracy by Architecture', fontsize=12, fontweight='bold')
    ax2.set_ylim(50, 100)
    ax2.legend()

    for bar, val in zip(bars, arch_k12.values()):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 3. Best K14 per Architecture
    ax3 = axes[1, 0]
    arch_k14 = {}
    for arch, exps in arch_results.items():
        if exps:
            best = max(exps, key=lambda x: x[1].get("test_per_class", {}).get("K14", 0))
            arch_k14[arch] = best[1].get("test_per_class", {}).get("K14", 0) * 100
        else:
            arch_k14[arch] = 0

    bars = ax3.bar(arch_k14.keys(), arch_k14.values(), color=colors, edgecolor='black', linewidth=2)
    ax3.axhline(y=68.75, color='red', linestyle='--', linewidth=2, label='Baseline K14 (68.75%)')
    ax3.set_ylabel('Best K14 Accuracy (%)', fontsize=11)
    ax3.set_title('Best K14 (Most Rare Class) Accuracy by Architecture', fontsize=12, fontweight='bold')
    ax3.set_ylim(50, 100)
    ax3.legend()

    for bar, val in zip(bars, arch_k14.values()):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 4. Training Time per Architecture (average)
    ax4 = axes[1, 1]
    arch_time = {}
    for arch, exps in arch_results.items():
        if exps:
            times = [e[1].get("total_time_minutes", 0) for e in exps if e[1].get("total_time_minutes", 0) > 0]
            arch_time[arch] = np.mean(times) if times else 0
        else:
            arch_time[arch] = 0

    bars = ax4.bar(arch_time.keys(), arch_time.values(), color=colors, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Average Training Time (minutes)', fontsize=11)
    ax4.set_title('Average Training Time by Architecture', fontsize=12, fontweight='bold')

    for bar, val in zip(bars, arch_time.values()):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f} min', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "architecture_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'architecture_comparison.png'}")


def plot_per_class_heatmap(results):
    """Create heatmap of per-class performance across experiments."""
    # Get all classes
    all_classes = ["K1", "K3", "K4", "K5", "K7", "K10", "K12", "K13", "K14", "K17", "K21", "K27", "noise"]

    # Filter valid experiments
    valid_results = {k: v for k, v in results.items()
                     if v.get("test_acc", 0) > 0.5 and "test_per_class" in v}

    # Sort by overall accuracy
    sorted_exps = sorted(valid_results.items(), key=lambda x: x[1]["test_acc"], reverse=True)[:10]

    # Build matrix
    exp_names = [e[0].replace("_", "\n")[:25] for e in sorted_exps]
    matrix = []
    for _, data in sorted_exps:
        row = [data["test_per_class"].get(c, 0) * 100 for c in all_classes]
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=50, vmax=100)

    # Labels
    ax.set_xticks(range(len(all_classes)))
    ax.set_xticklabels(all_classes, fontsize=10, rotation=45, ha='right')
    ax.set_yticks(range(len(exp_names)))
    ax.set_yticklabels(exp_names, fontsize=9)

    # Add values
    for i in range(len(exp_names)):
        for j in range(len(all_classes)):
            val = matrix[i, j]
            color = 'white' if val < 70 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                   fontsize=8, color=color, fontweight='bold')

    ax.set_title('Per-Class Test Accuracy Heatmap (%)\nTop 10 Experiments by Overall Accuracy',
                fontsize=12, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy (%)', fontsize=10)

    # Highlight rare classes
    ax.axvline(x=6.5, color='yellow', linewidth=3)  # Before K12
    ax.axvline(x=8.5, color='yellow', linewidth=3)  # After K14

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "per_class_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'per_class_heatmap.png'}")


def create_summary_report(results):
    """Create a summary report of all results."""
    report = []
    report.append("=" * 80)
    report.append("CETACEAN CLASSIFICATION - COMPREHENSIVE RESULTS SUMMARY")
    report.append("=" * 80)
    report.append("")

    # Find best experiments
    valid_results = {k: v for k, v in results.items() if v.get("test_acc", 0) > 0.5}

    # Best Overall
    best_overall = max(valid_results.items(), key=lambda x: x[1]["test_acc"])
    report.append("BEST OVERALL ACCURACY:")
    report.append(f"  Experiment: {best_overall[0]}")
    report.append(f"  Test Accuracy: {best_overall[1]['test_acc']*100:.2f}%")
    report.append(f"  Backbone: {best_overall[1].get('config', {}).get('backbone', 'N/A')}")
    report.append("")

    # Best K12
    best_k12 = max(valid_results.items(),
                   key=lambda x: x[1].get("test_per_class", {}).get("K12", 0))
    report.append("BEST K12 (RARE CLASS) ACCURACY:")
    report.append(f"  Experiment: {best_k12[0]}")
    report.append(f"  K12 Accuracy: {best_k12[1].get('test_per_class', {}).get('K12', 0)*100:.2f}%")
    report.append(f"  Overall Accuracy: {best_k12[1]['test_acc']*100:.2f}%")
    report.append("")

    # Best K14
    best_k14 = max(valid_results.items(),
                   key=lambda x: x[1].get("test_per_class", {}).get("K14", 0))
    report.append("BEST K14 (MOST RARE CLASS) ACCURACY:")
    report.append(f"  Experiment: {best_k14[0]}")
    report.append(f"  K14 Accuracy: {best_k14[1].get('test_per_class', {}).get('K14', 0)*100:.2f}%")
    report.append(f"  Overall Accuracy: {best_k14[1]['test_acc']*100:.2f}%")
    report.append("")

    # Comparison to baseline
    report.append("COMPARISON TO BASELINE (97.79%):")
    baseline = 97.79
    for name, data in sorted(valid_results.items(), key=lambda x: x[1]["test_acc"], reverse=True):
        acc = data["test_acc"] * 100
        diff = acc - baseline
        sign = "+" if diff >= 0 else ""
        report.append(f"  {name[:40]:<40} {acc:.2f}% ({sign}{diff:.2f}%)")

    report.append("")
    report.append("=" * 80)

    report_text = "\n".join(report)

    # Save report
    with open(OUTPUT_DIR / "summary_report.txt", "w") as f:
        f.write(report_text)

    print(f"Saved: {OUTPUT_DIR / 'summary_report.txt'}")
    print("\n" + report_text)

    return report_text


def main():
    print("Loading all experiment results...")
    results = load_all_results()
    print(f"Loaded {len(results)} experiments")

    print("\nCreating comparison plots...")

    # 1. Overall accuracy comparison
    plot_overall_comparison(results)

    # 2. Rare class (K12/K14) comparison
    plot_rare_class_comparison(results)

    # 3. Architecture comparison
    plot_architecture_comparison(results)

    # 4. Per-class heatmap
    plot_per_class_heatmap(results)

    # 5. Summary report
    create_summary_report(results)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
