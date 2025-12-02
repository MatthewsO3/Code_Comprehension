"""
Cross-Language MLM Evaluation Visualization
Creates bar charts with languages grouped by accuracy metrics
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_language_results(file_path):
    """Extract language results (Top-1, Top-5, Top-10, Perplexity) from result file."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Parsing {file_path}...")

    # Extract model name
    model_match = re.search(r'CROSS-LANGUAGE MLM EVALUATION: (.+?)[\n\r]', text)
    model_name = model_match.group(1).strip() if model_match else "Unknown"
    print(f"Model: {model_name}")

    languages = {}

    # Split by language headers (look for uppercase words followed by dashes)
    sections = re.split(r'\n([A-Z\+]+)\s*\n-{5,}', text)

    print(f"Found {len(sections)} sections")

    # Process pairs of (language_name, language_section)
    for i in range(1, len(sections), 2):
        if i + 1 < len(sections):
            lang_name = sections[i].strip()
            lang_section = sections[i + 1]

            # Extract metrics from this section
            top1_match = re.search(r'Top-1 Accuracy:\s+([\d.]+)', lang_section)
            top5_match = re.search(r'Top-5 Accuracy:\s+([\d.]+)', lang_section)
            top10_match = re.search(r'Top-10 Accuracy:\s+([\d.]+)', lang_section)
            ppl_match = re.search(r'Perplexity:\s+([\d.]+)', lang_section)

            if top1_match:
                languages[lang_name] = {
                    'top1': float(top1_match.group(1)),
                    'top5': float(top5_match.group(1)) if top5_match else 0.0,
                    'top10': float(top10_match.group(1)) if top10_match else 0.0,
                    'perplexity': float(ppl_match.group(1)) if ppl_match else 0.0,
                }
                print(f"  Found {lang_name}: Top-1={languages[lang_name]['top1']:.4f}, Top-5={languages[lang_name]['top5']:.4f}, Top-10={languages[lang_name]['top10']:.4f}")

    print(f"Total languages found: {len(languages)}\n")
    return model_name, languages


def plot_single_model(model_name, languages, output_path=None):
    """
    Plot results for a single model across languages.
    X-axis: Accuracy metrics (Top-1, Top-5, Top-10)
    Groups: Languages with different colors
    """

    lang_names = sorted(languages.keys())
    metrics = ["Top-1 Accuracy", "Top-5 Accuracy", "Top-10 Accuracy"]

    # Data: rows = metrics, cols = languages
    data = np.array([
        [languages[l]['top1'] for l in lang_names],
        [languages[l]['top5'] for l in lang_names],
        [languages[l]['top10'] for l in lang_names],
    ])

    # Color mapping per language (consistent)
    color_map = {
        "JAVA": "#FF6B6B",
        "PYTHON": "#4ECDC4",
        "JAVASCRIPT": "#45B7D1",
        "C++": "#FFA07A",
    }
    colors = [color_map.get(l, "#95E1D3") for l in lang_names]

    # Plot: Metrics on X-axis, languages as grouped bars
    x = np.arange(len(metrics))
    n_langs = len(lang_names)
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, lang in enumerate(lang_names):
        offset = (i - n_langs/2 + 0.5) * width
        bars = ax.bar(x + offset, data[:, i], width=width, label=lang,
                     color=colors[i], edgecolor='black', linewidth=1.2)

        # Add labels above bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   f"{height:.4f}", ha="center", va="bottom", fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_title(f"{model_name}", fontsize=16, fontweight='bold', pad=20)
    ax.legend(title="Language", fontsize=10, title_fontsize=11, loc='upper right')
    ax.grid(axis="y", alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    plt.tight_layout()

    if output_path:
        out_file = str(output_path).replace('.txt', '.png').replace('summary', 'chart')
    else:
        out_file = f'{model_name.lower().replace(" ", "_")}_comparison.png'

    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {out_file}")
    plt.show()

    # --- Plot 2: Perplexity chart ---
    perplexities = [languages[l]['perplexity'] for l in lang_names]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(lang_names, perplexities, color=colors, edgecolor='black', linewidth=1.5)

    # Add labels on top
    for bar, val in zip(bars, perplexities):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05*max(perplexities),
               f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight='bold')

    ax.set_ylabel("Perplexity", fontsize=12, fontweight='bold')
    ax.set_xlabel("Language", fontsize=12, fontweight='bold')
    ax.set_title(f"{model_name} - Perplexity", fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis="y", alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()

    if output_path:
        out_file = str(output_path).replace('.txt', '_perplexity.png').replace('summary', 'chart')
    else:
        out_file = f'{model_name.lower().replace(" ", "_")}_perplexity.png'

    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {out_file}")
    plt.show()

def plot_model_comparison(all_models_data):
    """
    Compare models on each metric (Top-1, Top-5, Top-10, Perplexity)
    for each language.
    Produces one chart per (language, metric).
    """

    metrics = {
        "top1": "Top-1 Accuracy",
        "top5": "Top-5 Accuracy",
        "top10": "Top-10 Accuracy",
        "perplexity": "Perplexity",
    }

    # Collect all languages across models
    all_languages = set()
    for model, langs in all_models_data.items():
        all_languages.update(langs.keys())
    all_languages = sorted(all_languages)

    for lang in all_languages:

        for metric_key, metric_label in metrics.items():

            models = []
            values = []

            for model_name, lang_data in all_models_data.items():
                if lang in lang_data:
                    models.append(model_name)
                    values.append(lang_data[lang][metric_key])
                else:
                    # If missing, add zero so chart stays aligned
                    models.append(model_name)
                    values.append(0.0)

            # Create bar plot
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(models))

            bars = ax.bar(x, values, edgecolor='black', linewidth=1.5)

            # Add value labels above bars
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold"
                )

            ax.set_xticks(x)
            ax.set_xticklabels(models, fontsize=11, fontweight='bold')
            ax.set_title(f"{lang} — {metric_label}", fontsize=16, fontweight='bold', pad=20)

            # Metrics are percentages except perplexity
            if metric_key != "perplexity":
                ax.set_ylim(0, 1.1)
                ax.set_ylabel("Accuracy", fontsize=12, fontweight='bold')
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda y, _: f"{y:.0%}")
                )
            else:
                ax.set_ylabel("Perplexity", fontsize=12, fontweight='bold')

            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.set_axisbelow(True)
            plt.tight_layout()

            out_file = f"compare_{lang.lower()}_{metric_key}.png"
            plt.savefig(out_file, dpi=300, bbox_inches="tight")
            print(f"✓ Saved {out_file}")
            plt.close()

# ============================================================================
# MAIN USAGE
# ============================================================================

if __name__ == "__main__":
    # Define your result files
    result_files = {
        "CodeBERT": "/Users/czapmate/Desktop/szakdoga/GraphCodeBert_CPP/BERTModels/Evaluate/older_lang_eval/results/cross_language_codebert/cross_language_summary.txt",
        "GraphCodeBERT": "/Users/czapmate/Desktop/szakdoga/GraphCodeBert_CPP/BERTModels/Evaluate/older_lang_eval/results/cross_language/cross_language_summary.txt",
        "UniXcoder": "/Users/czapmate/Desktop/szakdoga/GraphCodeBert_CPP/BERTModels/Evaluate/older_lang_eval/results/cross_language_unixcoder/cross_language_summary.txt",
    }

    all_models_data = {}

    print("=" * 80)
    print("CROSS-LANGUAGE MODEL COMPARISON (PER-LANGUAGE, PER-METRIC)")
    print("=" * 80)

    for model_name, file_path in result_files.items():
        file_path = Path(file_path)

        if not file_path.exists():
            print(f"⚠ File not found: {file_path}")
            continue

        print(f"\nProcessing {model_name}...")

        parsed_model_name, lang_data = parse_language_results(file_path)

        if lang_data:
            all_models_data[model_name] = lang_data
        else:
            print(f"⚠ No languages found in {model_name}")

    # Now produce cross-model comparison charts
    if all_models_data:
        plot_model_comparison(all_models_data)

    print("\n" + "=" * 80)
    print("COMPARISON VISUALIZATION COMPLETE")
    print("=" * 80)