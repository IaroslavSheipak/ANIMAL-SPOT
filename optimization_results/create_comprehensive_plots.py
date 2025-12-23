#!/usr/bin/env python3
"""
Comprehensive Training Results Comparison Plots
Integrates November 2025 baselines with December 2025 optimization experiments.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# November 2025 Baselines (from model_analysis_2025-11-09)
NOVEMBER_BASELINES = {
    "ResNet-18 Nov (91.95%)": {
        "overall": 91.95,
        "per_class": {
            "K1": 94.0, "K3": 52.9, "K4": 35.7, "K5": 99.0, "K7": 91.9,
            "K10": 76.7, "K12": 71.1, "K13": 76.9, "K14": 56.2,
            "K17": 98.0, "K21": 76.7, "K27": 0.0, "noise": 99.5
        }
    },
    "ConvNeXt Pre Nov (96.91%)": {
        "overall": 96.91,
        "per_class": {
            "K1": 99.0, "K3": 99.0, "K4": 84.6, "K5": 99.4, "K7": 95.4,
            "K10": 70.0, "K12": 81.6, "K13": 84.6, "K14": 62.5,
            "K17": 92.2, "K21": 76.7, "K27": 0.0, "noise": 99.9
        }
    },
    "ConvNeXt Stable Nov (98.01%)": {
        "overall": 98.01,
        "per_class": {
            "K1": 100.0, "K3": 96.2, "K4": 90.9, "K5": 99.4, "K7": 97.7,
            "K10": 93.3, "K12": 85.5, "K13": 76.9, "K14": 62.5,
            "K17": 100.0, "K21": 70.0, "K27": 71.4, "noise": 99.7
        }
    }
}

def load_december_results():
    """Load December 2025 experiment results."""
    results = {}
    base_path = Path(__file__).parent.parent

    # Run 20251222 experiments
    run_path = base_path / "ultimate_results" / "run_20251222_170746"
    if run_path.exists():
        for exp_dir in run_path.iterdir():
            if exp_dir.is_dir():
                results_file = exp_dir / "results.json"
                if results_file.exists():
                    with open(results_file) as f:
                        data = json.load(f)
                        results[exp_dir.name] = {
                            "overall": data["test_acc"] * 100,
                            "per_class": {k: v * 100 for k, v in data["test_per_class"].items()}
                        }

    # Final ConvNeXt
    convnext_path = base_path / "ultimate_results" / "final_convnext_20251223_025532" / "results.json"
    if convnext_path.exists():
        with open(convnext_path) as f:
            data = json.load(f)
            results["final_convnext_dec"] = {
                "overall": data["test_acc"] * 100,
                "per_class": {k: v * 100 for k, v in data["test_per_class"].items()}
            }

    # K12/K14 optimization experiments (NEW BEST!)
    k12_k14_path = base_path / "k12_k14_optimization_20251223_032653"
    if k12_k14_path.exists():
        for exp_dir in k12_k14_path.iterdir():
            if exp_dir.is_dir():
                results_file = exp_dir / "results.json"
                if results_file.exists():
                    with open(results_file) as f:
                        data = json.load(f)
                        results[f"k12k14_{exp_dir.name}"] = {
                            "overall": data["test_acc"] * 100,
                            "per_class": {k: v * 100 for k, v in data["test_per_class"].items()}
                        }

    return results

def create_overall_comparison(nov_data, dec_data, output_dir):
    """Create overall accuracy comparison chart."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Combine all results
    all_results = []

    # November baselines
    for name, data in nov_data.items():
        all_results.append((name, data["overall"], "November 2025"))

    # December experiments (select key ones)
    key_experiments = {
        "01_resnet18_fast_baseline": "ResNet18 Fast Dec",
        "06_resnet18_label_smooth": "ResNet18 LabelSmooth Dec",
        "k12k14_exp3_max_seq_balanced": "exp3 1000ms CHAMPION",
        "k12k14_exp1_long_seq_label_smooth": "exp1 800ms",
        "final_convnext_dec": "ConvNeXt Optimized Dec"
    }

    for exp_name, display_name in key_experiments.items():
        if exp_name in dec_data:
            acc = dec_data[exp_name]["overall"]
            all_results.append((f"{display_name} ({acc:.1f}%)", acc, "December 2025"))

    # Sort by accuracy
    all_results.sort(key=lambda x: x[1], reverse=True)

    names = [r[0] for r in all_results]
    accs = [r[1] for r in all_results]
    colors = ['#2ecc71' if r[2] == "December 2025" else '#3498db' for r in all_results]

    bars = ax.barh(range(len(names)), accs, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Overall Accuracy: November vs December 2025 Experiments', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 105)

    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accs)):
        ax.text(acc + 0.5, i, f'{acc:.2f}%', va='center', fontsize=10)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='December 2025 (New)'),
        Patch(facecolor='#3498db', label='November 2025 (Baseline)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    # Reference line
    ax.axvline(x=98.01, color='red', linestyle='--', alpha=0.7, label='Best Nov (98.01%)')
    ax.text(98.01, -0.5, 'Nov Best', color='red', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'overall_comparison_nov_vs_dec.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: overall_comparison_nov_vs_dec.png")

def create_rare_class_analysis(nov_data, dec_data, output_dir):
    """Analyze rare class (K12, K14, K21, K27) performance."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    rare_classes = ['K12', 'K14', 'K21', 'K27']

    for idx, cls in enumerate(rare_classes):
        ax = axes[idx // 2, idx % 2]

        results = []

        # November baselines
        for name, data in nov_data.items():
            short_name = name.split('(')[0].strip()
            results.append((short_name, data["per_class"].get(cls, 0), "#3498db"))

        # Best December experiments for this class
        dec_key = {
            "06_resnet18_label_smooth": "LabelSmooth Dec",
            "k12k14_exp3_max_seq_balanced": "exp3 CHAMPION",
            "k12k14_exp1_long_seq_label_smooth": "exp1 800ms",
            "final_convnext_dec": "ConvNeXt Dec"
        }
        for exp_name, display_name in dec_key.items():
            if exp_name in dec_data:
                results.append((display_name, dec_data[exp_name]["per_class"].get(cls, 0), "#2ecc71"))

        names = [r[0] for r in results]
        vals = [r[1] for r in results]
        colors = [r[2] for r in results]

        bars = ax.bar(range(len(names)), vals, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{cls} Class Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 110)

        # Value labels
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%',
                   ha='center', va='bottom', fontsize=9)

        # Highlight best
        best_idx = np.argmax(vals)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)

    plt.suptitle('Rare Class Performance: November vs December 2025', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'rare_class_nov_vs_dec.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: rare_class_nov_vs_dec.png")

def create_per_class_heatmap(nov_data, dec_data, output_dir):
    """Create per-class accuracy heatmap."""
    fig, ax = plt.subplots(figsize=(16, 10))

    classes = ['K1', 'K3', 'K4', 'K5', 'K7', 'K10', 'K12', 'K13', 'K14', 'K17', 'K21', 'K27', 'noise']

    # Select models to compare
    models = {}

    # November
    for name, data in nov_data.items():
        short = name.split('(')[0].strip()
        models[short] = data["per_class"]

    # December (key experiments)
    dec_select = {
        "01_resnet18_fast_baseline": "ResNet18 Fast Dec",
        "06_resnet18_label_smooth": "ResNet18 LS Dec",
        "k12k14_exp3_max_seq_balanced": "exp3 CHAMPION",
        "k12k14_exp1_long_seq_label_smooth": "exp1 800ms",
        "final_convnext_dec": "ConvNeXt Dec"
    }
    for exp, name in dec_select.items():
        if exp in dec_data:
            models[name] = dec_data[exp]["per_class"]

    # Build matrix
    model_names = list(models.keys())
    matrix = np.zeros((len(model_names), len(classes)))

    for i, model in enumerate(model_names):
        for j, cls in enumerate(classes):
            matrix[i, j] = models[model].get(cls, 0)

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)

    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(classes)):
            val = matrix[i, j]
            color = 'white' if val < 50 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label='Accuracy (%)')
    ax.set_title('Per-Class Accuracy Heatmap: All Models', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_heatmap_full.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: per_class_heatmap_full.png")

def create_improvement_chart(nov_data, dec_data, output_dir):
    """Show improvement from best November to best December."""
    fig, ax = plt.subplots(figsize=(12, 8))

    classes = ['K1', 'K3', 'K4', 'K5', 'K7', 'K10', 'K12', 'K13', 'K14', 'K17', 'K21', 'K27', 'noise']

    # Best November (ConvNeXt Stable)
    nov_best = nov_data["ConvNeXt Stable Nov (98.01%)"]["per_class"]

    # Best December (exp3 1000ms CHAMPION)
    dec_best = dec_data.get("k12k14_exp3_max_seq_balanced", {}).get("per_class", {})
    if not dec_best:
        dec_best = dec_data.get("06_resnet18_label_smooth", {}).get("per_class", {})

    nov_vals = [nov_best.get(c, 0) for c in classes]
    dec_vals = [dec_best.get(c, 0) for c in classes]
    improvements = [d - n for n, d in zip(nov_vals, dec_vals)]

    x = np.arange(len(classes))
    width = 0.35

    bars1 = ax.bar(x - width/2, nov_vals, width, label='ConvNeXt Stable Nov (98.01%)', color='#3498db')
    bars2 = ax.bar(x + width/2, dec_vals, width, label='exp3 1000ms CHAMPION (99.49%)', color='#2ecc71')

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Best November vs Best December: Per-Class Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 110)

    # Annotate improvements
    for i, (xi, imp) in enumerate(zip(x, improvements)):
        color = 'green' if imp > 0 else 'red' if imp < 0 else 'gray'
        sign = '+' if imp > 0 else ''
        ax.annotate(f'{sign}{imp:.1f}', xy=(xi, max(nov_vals[i], dec_vals[i]) + 2),
                   ha='center', fontsize=8, color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_nov_to_dec.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: improvement_nov_to_dec.png")

def generate_summary_report(nov_data, dec_data, output_dir):
    """Generate text summary report."""
    report = []
    report.append("=" * 80)
    report.append("CETACEAN CLASSIFICATION - COMPREHENSIVE RESULTS COMPARISON")
    report.append("November 2025 Baselines vs December 2025 Optimization")
    report.append("=" * 80)
    report.append("")

    # Best results
    report.append("BEST OVERALL RESULTS:")
    report.append(f"  November 2025 Best: ConvNeXt Stable = 98.01%")

    dec_best_name = max(dec_data.keys(), key=lambda k: dec_data[k]["overall"])
    dec_best_acc = dec_data[dec_best_name]["overall"]
    report.append(f"  December 2025 Best: {dec_best_name} = {dec_best_acc:.2f}%")
    report.append(f"  Improvement: +{dec_best_acc - 98.01:.2f}%")
    report.append("")

    # Key findings
    report.append("KEY FINDINGS:")
    report.append("  1. Label smoothing (0.1) achieves best overall accuracy")
    report.append("  2. ResNet18 outperforms ConvNeXt when properly optimized")
    report.append("  3. Focal loss causes noise class collapse - AVOID")
    report.append("  4. sqrt_inverse_freq weighted sampling effective for imbalance")
    report.append("  5. OneCycle scheduler outperforms plateau-based schedulers")
    report.append("")

    # Rare class comparison
    report.append("RARE CLASS PERFORMANCE (K12, K14, K21, K27):")
    report.append("-" * 60)

    nov_best_pc = nov_data["ConvNeXt Stable Nov (98.01%)"]["per_class"]
    dec_best_pc = dec_data.get("06_resnet18_label_smooth", {}).get("per_class", {})
    dec_long_pc = dec_data.get("11_resnet18_long_seq", {}).get("per_class", {})

    for cls in ['K12', 'K14', 'K21', 'K27']:
        nov_val = nov_best_pc.get(cls, 0)
        dec_val = dec_best_pc.get(cls, 0)
        long_val = dec_long_pc.get(cls, 0)
        best_dec = max(dec_val, long_val)
        report.append(f"  {cls}: Nov={nov_val:.1f}% -> Dec Best={best_dec:.1f}% ({'+' if best_dec > nov_val else ''}{best_dec - nov_val:.1f}%)")

    report.append("")
    report.append("=" * 80)

    with open(output_dir / 'comprehensive_summary.txt', 'w') as f:
        f.write('\n'.join(report))
    print("Created: comprehensive_summary.txt")

def main():
    output_dir = Path(__file__).parent

    print("Loading December 2025 results...")
    dec_data = load_december_results()
    print(f"Loaded {len(dec_data)} experiments")

    print("\nGenerating comparison plots...")
    create_overall_comparison(NOVEMBER_BASELINES, dec_data, output_dir)
    create_rare_class_analysis(NOVEMBER_BASELINES, dec_data, output_dir)
    create_per_class_heatmap(NOVEMBER_BASELINES, dec_data, output_dir)
    create_improvement_chart(NOVEMBER_BASELINES, dec_data, output_dir)
    generate_summary_report(NOVEMBER_BASELINES, dec_data, output_dir)

    print("\nAll plots generated successfully!")

if __name__ == "__main__":
    main()
