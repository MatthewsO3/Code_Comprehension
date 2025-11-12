"""
Plot training metrics from training_history.json for UniXcoder using matplotlib.
Creates comprehensive visualizations of UniXcoder training.
Simpler than GraphCodeBERT (no MLM/Edge split) - only total loss tracking.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {
    'train': '#06b6d4',  # Cyan
    'val': '#ef4444',  # Red
    'lr': '#fbbf24'  # Amber
}


def load_training_history(filepath: str = 'training_history.json') -> dict:
    """Load training history from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_total_loss(history: dict, save_path: str = None):
    """Plot total loss (Train + Validation)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    epochs = history['epoch']
    train_loss = history['train_loss']
    val_loss = history['val_loss']

    ax.plot(epochs, train_loss, marker='o', linewidth=2.5, label='Train Loss',
            color=colors['train'], markersize=10)
    ax.plot(epochs, val_loss, marker='s', linewidth=2.5, label='Val Loss',
            color=colors['val'], markersize=10)

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('UniXcoder: Training & Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add value annotations
    for i, (e, t, v) in enumerate(zip(epochs, train_loss, val_loss)):
        ax.annotate(f'{t:.2f}', (e, t), textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=10, fontweight='bold', color=colors['train'])
        ax.annotate(f'{v:.2f}', (e, v), textcoords="offset points", xytext=(0, -15),
                    ha='center', fontsize=10, fontweight='bold', color=colors['val'])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_train_vs_val(history: dict, save_path: str = None):
    """Plot train vs validation comparison (bar chart)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = history['epoch']
    train_loss = history['train_loss']
    val_loss = history['val_loss']

    x = np.arange(len(epochs))
    width = 0.35

    bars1 = ax.bar(x - width / 2, train_loss, width, label='Train Loss',
                   color=colors['train'], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width / 2, val_loss, width, label='Val Loss',
                   color=colors['val'], alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('UniXcoder: Train vs Validation Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(epochs)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_batch_losses(history: dict, save_path: str = None):
    """Plot all batch losses to show training dynamics."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    train_batches = history['train_batch_losses']
    val_batches = history['val_batch_losses']

    # Train batch losses
    ax1.plot(train_batches, linewidth=1.5, color=colors['train'], alpha=0.7, label='Train Batch Losses')
    ax1.fill_between(range(len(train_batches)), train_batches, alpha=0.2, color=colors['train'])

    # Add moving average
    window = 5
    if len(train_batches) > window:
        moving_avg = np.convolve(train_batches, np.ones(window) / window, mode='valid')
        ax1.plot(range(window - 1, len(train_batches)), moving_avg, linewidth=2.5,
                 color='#3b82f6', alpha=0.8, label=f'Moving Avg (window={window})')

    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('UniXcoder: Training Batch Losses - All Iterations', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Validation batch losses
    ax2.plot(val_batches, linewidth=1.5, color=colors['val'], alpha=0.7, label='Val Batch Losses')
    ax2.fill_between(range(len(val_batches)), val_batches, alpha=0.2, color=colors['val'])

    # Add moving average for validation
    if len(val_batches) > window:
        moving_avg_val = np.convolve(val_batches, np.ones(window) / window, mode='valid')
        ax2.plot(range(window - 1, len(val_batches)), moving_avg_val, linewidth=2.5,
                 color='#dc2626', alpha=0.8, label=f'Moving Avg (window={window})')

    ax2.set_xlabel('Batch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('UniXcoder: Validation Batch Losses - All Iterations', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_learning_rate(history: dict, save_path: str = None):
    """Plot learning rate schedule."""
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = history['epoch']
    lr = history['learning_rate']
    lr_scaled = [lr_val * 1e6 for lr_val in lr]  # Scale to µ units

    ax.plot(epochs, lr_scaled, marker='o', linewidth=2.5, markersize=10,
            color=colors['lr'], label='Learning Rate')

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learning Rate (×10⁻⁶)', fontsize=12, fontweight='bold')
    ax.set_title('UniXcoder: Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Add value annotations
    for e, lr_val in zip(epochs, lr_scaled):
        ax.annotate(f'{lr_val:.2f}', (e, lr_val), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_all_metrics(history: dict, save_path: str = None):
    """Plot all metrics in one comprehensive figure."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    epochs = history['epoch']
    train_batches = history['train_batch_losses']
    val_batches = history['val_batch_losses']

    # 1. Total Loss
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(epochs, history['train_loss'], marker='o', linewidth=2.5,
             label='Train Loss', color=colors['train'], markersize=10)
    ax1.plot(epochs, history['val_loss'], marker='s', linewidth=2.5,
             label='Val Loss', color=colors['val'], markersize=10)
    ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax1.set_title('Total Loss (Train & Validation)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Train vs Val Bar Chart
    ax2 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(epochs))
    width = 0.35
    ax2.bar(x - width / 2, history['train_loss'], width, label='Train',
            color=colors['train'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.bar(x + width / 2, history['val_loss'], width, label='Val',
            color=colors['val'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax2.set_title('Train vs Validation Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(epochs)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Learning Rate
    ax3 = fig.add_subplot(gs[1, 1])
    lr_scaled = [lr_val * 1e6 for lr_val in history['learning_rate']]
    ax3.plot(epochs, lr_scaled, marker='o', linewidth=2.5, markersize=10,
             color=colors['lr'], label='Learning Rate')
    ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax3.set_ylabel('LR (×10⁻⁶)', fontsize=11, fontweight='bold')
    ax3.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Train Batch Losses
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(train_batches, linewidth=1, color=colors['train'], alpha=0.6)
    ax4.fill_between(range(len(train_batches)), train_batches, alpha=0.2, color=colors['train'])
    window = 5
    if len(train_batches) > window:
        moving_avg = np.convolve(train_batches, np.ones(window) / window, mode='valid')
        ax4.plot(range(window - 1, len(train_batches)), moving_avg, linewidth=2,
                 color='#3b82f6', label=f'Moving Avg ({window})')
    ax4.set_xlabel('Batch', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax4.set_title('Training Batch Losses', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    if len(train_batches) > window:
        ax4.legend(fontsize=9)

    # 5. Val Batch Losses
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(val_batches, linewidth=1, color=colors['val'], alpha=0.6)
    ax5.fill_between(range(len(val_batches)), val_batches, alpha=0.2, color=colors['val'])
    if len(val_batches) > window:
        moving_avg_val = np.convolve(val_batches, np.ones(window) / window, mode='valid')
        ax5.plot(range(window - 1, len(val_batches)), moving_avg_val, linewidth=2,
                 color='#dc2626', label=f'Moving Avg ({window})')
    ax5.set_xlabel('Batch', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax5.set_title('Validation Batch Losses', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    if len(val_batches) > window:
        ax5.legend(fontsize=9)

    fig.suptitle('UniXcoder Training: Comprehensive Metrics', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_batch_statistics(history: dict, save_path: str = None):
    """Plot batch loss statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    train_batches = history['train_batch_losses']
    val_batches = history['val_batch_losses']

    # Train batch histogram
    axes[0, 0].hist(train_batches, bins=15, color=colors['train'], alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(train_batches), color='#3b82f6', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(train_batches):.2f}')
    axes[0, 0].set_xlabel('Loss', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Train Batch Loss Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Val batch histogram
    axes[0, 1].hist(val_batches, bins=15, color=colors['val'], alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(np.mean(val_batches), color='#dc2626', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(val_batches):.2f}')
    axes[0, 1].set_xlabel('Loss', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Val Batch Loss Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Train batch statistics boxplot
    axes[1, 0].boxplot([train_batches], labels=['Train'], patch_artist=True,
                       boxprops=dict(facecolor=colors['train'], alpha=0.7),
                       medianprops=dict(color='black', linewidth=2))
    axes[1, 0].set_ylabel('Loss', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Train Batch Loss Statistics', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Val batch statistics boxplot
    axes[1, 1].boxplot([val_batches], labels=['Val'], patch_artist=True,
                       boxprops=dict(facecolor=colors['val'], alpha=0.7),
                       medianprops=dict(color='black', linewidth=2))
    axes[1, 1].set_ylabel('Loss', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Val Batch Loss Statistics', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('UniXcoder: Batch Loss Statistics', fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def print_summary(history: dict):
    """Print training summary statistics."""
    print("\n" + "=" * 70)
    print("UniXcoder Training Summary".center(70))
    print("=" * 70)

    train_loss = history['train_loss']
    val_loss = history['val_loss']
    best_val = history['best_val_loss']
    best_epoch = history['best_epoch']
    train_batches = history['train_batch_losses']
    val_batches = history['val_batch_losses']

    print(f"\nTraining Loss:")
    print(f"  Initial:  {train_loss[0]:.4f}")
    print(f"  Final:    {train_loss[-1]:.4f}")
    if train_loss[-1] < train_loss[0]:
        improvement = ((train_loss[0] - train_loss[-1]) / train_loss[0] * 100)
        print(f"  Improvement: {improvement:.1f}%")
    else:
        worsening = ((train_loss[-1] - train_loss[0]) / train_loss[0] * 100)
        print(f"  Worsening: {worsening:.1f}%")
    print(f"  Min:      {min(train_loss):.4f}")
    print(f"  Max:      {max(train_loss):.4f}")

    print(f"\nValidation Loss:")
    print(f"  Initial:  {val_loss[0]:.4f}")
    print(f"  Final:    {val_loss[-1]:.4f}")
    print(f"  Best:     {best_val:.4f} (Epoch {best_epoch})")
    if val_loss[-1] < val_loss[0]:
        improvement = ((val_loss[0] - val_loss[-1]) / val_loss[0] * 100)
        print(f"  Improvement: {improvement:.1f}%")
    else:
        worsening = ((val_loss[-1] - val_loss[0]) / val_loss[0] * 100)
        print(f"  Worsening: {worsening:.1f}%")
    print(f"  Min:      {min(val_loss):.4f}")
    print(f"  Max:      {max(val_loss):.4f}")

    print(f"\nBatch Losses Statistics:")
    print(f"  Train Batches:      {len(train_batches)}")
    print(f"    Mean: {np.mean(train_batches):.4f}")
    print(f"    Std:  {np.std(train_batches):.4f}")
    print(f"    Min:  {np.min(train_batches):.4f}")
    print(f"    Max:  {np.max(train_batches):.4f}")

    print(f"  Val Batches:        {len(val_batches)}")
    print(f"    Mean: {np.mean(val_batches):.4f}")
    print(f"    Std:  {np.std(val_batches):.4f}")
    print(f"    Min:  {np.min(val_batches):.4f}")
    print(f"    Max:  {np.max(val_batches):.4f}")

    print(f"\nLearning Rate Schedule:")
    print(f"  Epochs: {len(history['learning_rate'])}")
    for i, (epoch, lr) in enumerate(zip(history['epoch'], history['learning_rate'])):
        print(f"    Epoch {epoch}: {lr:.2e}")

    print("\n" + "=" * 70 + "\n")


def main():
    """Main function to generate all plots."""
    import argparse

    parser = argparse.ArgumentParser(description='Plot UniXcoder training metrics from training_history.json')
    parser.add_argument('--history_file', type=str, default='/Users/czapmate/Desktop/szakdoga/GraphCodeBert_CPP/BERTModels/UnixCoderCPP/unixcoder-cpp-mlm/training_history.json',
                        help='Path to training_history.json file')
    parser.add_argument('--output_dir', type=str, default='unix_training_plots',
                        help='Directory to save plots')
    parser.add_argument('--all', action='store_true', help='Generate all plots')
    parser.add_argument('--total', action='store_true', help='Plot total loss')
    parser.add_argument('--compare', action='store_true', help='Plot train vs val comparison')
    parser.add_argument('--batches', action='store_true', help='Plot batch losses')
    parser.add_argument('--lr', action='store_true', help='Plot learning rate')
    parser.add_argument('--stats', action='store_true', help='Plot batch statistics')
    parser.add_argument('--summary', action='store_true', help='Print summary only')

    args = parser.parse_args()

    # Load history
    if not Path(args.history_file).exists():
        print(f"Error: {args.history_file} not found")
        return

    history = load_training_history(args.history_file)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Print summary
    print_summary(history)

    # Generate plots
    if args.summary:
        return

    generate_all = args.all or not any([args.total, args.compare, args.batches, args.lr, args.stats])

    try:
        if args.total or generate_all:
            print("Generating total loss plot...")
            plot_total_loss(history, output_dir / 'unixcoder_total_loss.png')

        if args.compare or generate_all:
            print("Generating train vs val comparison...")
            plot_train_vs_val(history, output_dir / 'unixcoder_train_vs_val.png')

        if args.batches or generate_all:
            print("Generating batch losses plot...")
            plot_batch_losses(history, output_dir / 'unixcoder_batch_losses.png')

        if args.lr or generate_all:
            print("Generating learning rate plot...")
            plot_learning_rate(history, output_dir / 'unixcoder_learning_rate.png')

        if args.stats or generate_all:
            print("Generating batch statistics plot...")
            plot_batch_statistics(history, output_dir / 'unixcoder_batch_statistics.png')

        # Always generate the comprehensive figure
        if generate_all:
            print("Generating comprehensive metrics plot...")
            plot_all_metrics(history, output_dir / 'unixcoder_all_metrics.png')

        print(f"\n✓ All plots saved to: {output_dir}")
        print("✓ All files have 'unixcoder_' prefix\n")
    except Exception as e:
        print(f"Error generating plots: {e}")


if __name__ == "__main__":
    main()