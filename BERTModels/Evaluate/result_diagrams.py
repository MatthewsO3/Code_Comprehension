import matplotlib.pyplot as plt
import re
from pathlib import Path

# File paths


import re
import matplotlib.pyplot as plt
import numpy as np


def parse_results(file_path):
    """Extract Top-1, Top-5, Top-10 accuracies and perplexity from a result file."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    top1 = float(re.search(r"Top-1 Accuracy:\s+([\d.]+)", text).group(1))
    top5 = float(re.search(r"Top-5 Accuracy:\s+([\d.]+)", text).group(1))
    top10 = float(re.search(r"Top-10 Accuracy:\s+([\d.]+)", text).group(1))
    ppl = float(re.search(r"Perplexity:\s+([\d.]+)", text).group(1))

    return top1, top5, top10, ppl



# Read results
#models = ["CodeBERT", "GraphCodeBERT", "UniXcoder"]
files = {
    "CodeBERT": "/Users/czapmate/Desktop/szakdoga/GraphCodeBert_CPP/BERTModels/Evaluate/results/codebert-cpp_results.txt",
    "GraphCodeBERT": "/Users/czapmate/Desktop/szakdoga/GraphCodeBert_CPP/BERTModels/Evaluate/results/graphcodebert_results.txt",
    "UniXcoder": "/Users/czapmate/Desktop/szakdoga/GraphCodeBert_CPP/BERTModels/Evaluate/results/unixcoder_results.txt"
}





models = list(files.keys())

results = {m: parse_results(files[m]) for m in models}
# Data arrays: rows = models, cols = [Top-1, Top-5, Top-10]
metrics = ["Top-1 Accuracy", "Top-5 Accuracy", "Top-10 Accuracy"]
data = np.array([[results[m][i] for i in range(3)] for m in models])  # shape (n_models, 3)
perplexities = [results[m][3] for m in models]

# --- Color mapping (fixed per model) ---
# Change these colors if you prefer others. They will stay consistent across plots.
color_map = {
    "CodeBERT": "green",      # CodeBERT always green
    "GraphCodeBERT": "lightblue",      # GraphCodeBERT always blue
    "UniXcoder": "orange"         # UniXcoder always purple
}
colors = [color_map[m] for m in models]

# --- Plot 1: grouped bars where each model has a consistent color ---
x = np.arange(len(metrics))  # three groups: Top-1, Top-5, Top-10
n_models = len(models)
width = 0.2

plt.figure(figsize=(10, 6))
for i, m in enumerate(models):
    plt.bar(x + (i - n_models/2 + 0.5) * width, data[i], width=width, label=m, color=color_map[m])

# add labels above bars
for i in range(n_models):
    for j in range(len(metrics)):
        xpos = x[j] + (i - n_models/2 + 0.5) * width
        plt.text(xpos, data[i, j] + 0.01, f"{data[i, j]:.4f}", ha="center", va="bottom", fontsize=9)

plt.xticks(x, metrics)
plt.ylabel("Accuracy")
plt.ylim(0.7,1 )
plt.title("Models Comparison on Top-K Accuracies")
plt.legend()
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig('model_comparison_accuracy.png', dpi=300, bbox_inches='tight')

plt.show()

# --- Plot 2: Perplexities with same model colors ---
plt.figure(figsize=(8, 5))
bars = plt.bar(models, perplexities, color=colors)

# labels on top
for bar, val in zip(bars, perplexities):
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h + 0.02*h, f"{val:.4f}", ha="center", va="bottom", fontsize=9)

plt.ylabel("Perplexity")
plt.title("Models Comparison on Perplexity")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

plt.ylim(0.5, 4.5)
plt.savefig('model_comparison_perplexity.png', dpi=300, bbox_inches='tight')

plt.show()



