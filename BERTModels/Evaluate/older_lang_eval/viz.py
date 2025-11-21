"""
Cross-Language MLM Evaluation Visualization
Creates grouped bar charts comparing models across languages
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_language_results(file_path):
    """Extract language results (Top-1, Top-5, Top-10, Perplexity) from result file."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Extract model name
    model_match = re.search(r'CROSS-LANGUAGE MLM EVALUATION: (.+?)[\n\r]', text)
    model_name = model_match.group(1).strip() if model_match else "Unknown"

    languages = {}

    # Pattern to find each language section
    lang_pattern = r'^([A-Z]+)\s*\n-+\s*\n(.*?)(?=^[A-Z]+\s*\n-+|COMPARISON|$)'
    matches = re.finditer(lang_pattern, text, re.MULTILINE | re.DOTALL)

    for match in matches:
        lang_name = match.group(1).strip()
        lang_section = match.group(2)

        # Extract metrics
        top1_match = re.search(r'Top-1 Accuracy:\s+([\d.]+)', lang_section)
        top5_match = re.search(r'Top-5 Accuracy:\s+([\d.]+)', lang_section)
        top10_match = re.search(r'Top-10 Accuracy:\s+([\d.]+)', lang_section)
        ppl_match = re.search(r'Perplexity:\s+([\d.]+)', lang_section)

        if top1_match and top5_match and top10_match:
            languages[lang_name] = {
                'top1': float(top1_match.group(1)),
                'top5': float(top5_match.group(1)),
                'top10': float(top10_match.group(1)),
                'perplexity': float(ppl_match.group(1)) if ppl_match else 0.0,
            }

    return model_name, languages


def plot_single_model(model_name, languages, output_path=None):
    """Plot results for a single model across languages."""

    lang_names = sorted(languages.keys())
    metrics = ["Top-1 Accuracy", "Top-5 Accuracy", "Top-10 Accuracy"]

    # Data: rows = languages, cols = [Top-1, Top-5, Top-10]
    data = np.array([
        [languages[l]['top1'], languages[l]['top5'], languages[l]['top10']]
        for l in lang_names
    ])

    perplexities = [languages[l]['perplexity'] for l in lang_names]

    # Color mapping per language (consistent across plots)
    color_map = {
        "JAVA": "#FF6B6B",
        "PYTHON": "#4ECDC4",
        "JAVASCRIPT": "#45B7D1",
        "C++": "#FFA07A",
    }
    colors = [color_map.get(l, "#95E1D3") for l in lang_names]

    # --- Plot 1: Grouped bars for Top-K accuracies ---
    x = np.arange(len(metrics))
    n_langs = len(lang_names)
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, lang in enumerate(lang_names):
        offset = (i - n_langs / 2 + 0.5) * width
        bars = ax.bar(x + offset, data[i], width=width, label=lang, color=colors[i])

        # Add labels above bars
        for j, (bar, val) in enumerate(zip(bars, data[i])):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Accuracy", fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_title(f"{model_name} - Top-K Accuracies Across Languages",
                 fontweight='bold', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(str(output_path).replace('.png', '_accuracy.png'), dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'{model_name.lower()}_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()

    # --- Plot 2: Perplexity across languages ---
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(lang_names, perplexities, color=colors, edgecolor='black', linewidth=1.5)

    # Add labels on top
    for bar, val in zip(bars, perplexities):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05 * h,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight='bold')

    ax.set_ylabel("Perplexity", fontweight='bold')
    ax.set_title(f"{model_name} - Perplexity Across Languages",
                 fontweight='bold', fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(str(output_path).replace('.png', '_perplexity.png'), dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'{model_name.lower()}_perplexity.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_model_comparison(files_dict, output_dir=None):
    """Compare multiple models across a single language."""

    # Parse all files
    all_results = {}
    for model_name, file_path in files_dict.items():
        _, languages = parse_language_results(file_path)
        all_results[model_name] = languages

    # Get all languages
    all_languages = set()
    for langs in all_results.values():
        all_languages.update(langs.keys())
    all_languages = sorted(list(all_languages))

    # Color mapping per model (consistent)
    model_color_map = {
        "CodeBERT": "#2E86AB",
        "GraphCodeBERT": "#A23B72",
        "UniXcoder": "#F18F01",
    }

    metrics = ["Top-1 Accuracy", "Top-5 Accuracy", "Top-10 Accuracy"]

    # For each language, create a comparison plot
    for language in all_languages:
        # Collect data for this language across all models
        model_names = []
        top1_vals = []
        top5_vals = []
        top10_vals = []
        ppl_vals = []

        for model_name in sorted(all_results.keys()):
            if language in all_results[model_name]:
                model_names.append(model_name)
                lang_data = all_results[model_name][language]
                top1_vals.append(lang_data['top1'])
                top5_vals.append(lang_data['top5'])
                top10_vals.append(lang_data['top10'])
                ppl_vals.append(lang_data['perplexity'])

        if not model_names:
            continue

        data = np.array([top1_vals, top5_vals, top10_vals]).T  # shape (n_models, 3)
        colors = [model_color_map.get(m, "#95E1D3") for m in model_names]

        # --- Plot 1: Accuracy comparison ---
        x = np.arange(len(metrics))
        n_models = len(model_names)
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, model in enumerate(model_names):
            offset = (i - n_models / 2 + 0.5) * width
            bars = ax.bar(x + offset, data[i], width=width, label=model,
                          color=colors[i], edgecolor='black', linewidth=1)

            # Add labels
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                        f"{data[i, j]:.4f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylabel("Accuracy", fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.set_title(f"Model Comparison on {language} - Top-K Accuracies",
                     fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        output_file = f"model_comparison_{language.lower()}_accuracy.png"
        if output_dir:
            output_file = Path(output_dir) / output_file
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved {output_file}")
        plt.show()

        # --- Plot 2: Perplexity comparison ---
        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(model_names, ppl_vals, color=colors, edgecolor='black', linewidth=1.5)

        for bar, val in zip(bars, ppl_vals):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02 * max(ppl_vals),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight='bold')

        ax.set_ylabel("Perplexity", fontweight='bold')
        ax.set_title(f"Model Comparison on {language} - Perplexity",
                     fontweight='bold', fontsize=14)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        output_file = f"model_comparison_{language.lower()}_perplexity.png"
        if output_dir:
            output_file = Path(output_dir) / output_file
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved {output_file}")
        plt.show()


# ============================================================================
# MAIN USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Plot single model across languages
    print("=" * 80)
    print("PLOTTING SINGLE MODELS")
    print("=" * 80)

    model_files = {
        "CodeBERT": "results/cross_language_codebert/cross_language_summary.txt",
        "GraphCodeBERT": "results/cross_language/cross_language_summary.txt",
        "UniXcoder": "results/cross_language_unixcoder/cross_language_summary.txt",
    }

    for model_name, file_path in model_files.items():
        if Path(file_path).exists():
            print(f"\nProcessing {model_name}...")
            _, languages = parse_language_results(file_path)
            plot_single_model(model_name, languages)
        else:
            print(f"File not found: {file_path}")

    # Example 2: Compare models across languages
    print("\n" + "=" * 80)
    print("COMPARING MODELS ACROSS LANGUAGES")
    print("=" * 80)

    comparison_files = {
        "CodeBERT": "results/cross_language_codebert/cross_language_summary.txt",
        "GraphCodeBERT": "results/cross_language/cross_language_summary.txt",
        "UniXcoder": "results/cross_language_unixcoder/cross_language_summary.txt",
    }

    # Check if files exist
    existing_files = {k: v for k, v in comparison_files.items() if Path(v).exists()}

    if existing_files:
        plot_model_comparison(existing_files, output_dir="results")
    else:
        print("No result files found for comparison")