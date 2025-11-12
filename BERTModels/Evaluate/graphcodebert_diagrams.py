"""
Plot training metrics from training_history.json using matplotlib.
Creates comprehensive visualizations of GraphCodeBERT training.
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
    'mlm': '#10b981',  # Green
    'edge': '#8b5cf6',  # Purple
    'lr': '#fbbf24'  # Amber
}


def load_training_history(filepath: str = 'BERTModels/GraphCodeBert/Models/graphcodebert-cpp-mlm-from-config/training_history.json') -> dict:
    """Load training history from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_total_loss(history: dict, save_path: str = None):
    """Plot total loss (Train + Validation)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    epochs = history['epoch']
    train_loss = history['train_total_loss']
    val_loss = history['val_total_loss']

    ax.plot(epochs, train_loss, marker='o', linewidth=2.5, label='Train Total',
            color=colors['train'], markersize=8)
    ax.plot(epochs, val_loss, marker='s', linewidth=2.5, label='Val Total',
            color=colors['val'], markersize=8)

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Total Loss (Train + Validation)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add value annotations
    for i, (e, t, v) in enumerate(zip(epochs, train_loss, val_loss)):
        ax.annotate(f'{t:.3f}', (e, t), textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=9, color=colors['train'])
        ax.annotate(f'{v:.3f}', (e, v), textcoords="offset points", xytext=(0, -15),
                    ha='center', fontsize=9, color=colors['val'])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_mlm_loss(history: dict, save_path: str = None):
    """Plot MLM loss component."""
    fig, ax = plt.subplots(figsize=(12, 6))

    epochs = history['epoch']
    train_mlm = history['train_mlm_loss']
    val_mlm = history['val_mlm_loss']

    ax.plot(epochs, train_mlm, marker='o', linewidth=2.5, label='Train MLM',
            color=colors['mlm'], markersize=8)
    ax.plot(epochs, val_mlm, marker='s', linewidth=2.5, label='Val MLM',
            color=colors['edge'], markersize=8)

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('MLM Loss', fontsize=12, fontweight='bold')
    ax.set_title('MLM Loss Component', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add value annotations
    for i, (e, t, v) in enumerate(zip(epochs, train_mlm, val_mlm)):
        ax.annotate(f'{t:.3f}', (e, t), textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=9, color=colors['mlm'])
        ax.annotate(f'{v:.3f}', (e, v), textcoords="offset points", xytext=(0, -15),
                    ha='center', fontsize=9, color=colors['edge'])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_edge_loss(history: dict, save_path: str = None):
    """Plot edge prediction loss component."""
    fig, ax = plt.subplots(figsize=(12, 6))

    epochs = history['epoch']
    train_edge = history['train_edge_loss']
    val_edge = history['val_edge_loss']

    ax.plot(epochs, train_edge, marker='o', linewidth=2.5, label='Train Edge',
            color=colors['train'], markersize=8)
    ax.plot(epochs, val_edge, marker='s', linewidth=2.5, label='Val Edge',
            color=colors['val'], markersize=8)

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Edge Loss', fontsize=12, fontweight='bold')
    ax.set_title('Edge Prediction Loss Component', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add value annotations
    for i, (e, t, v) in enumerate(zip(epochs, train_edge, val_edge)):
        ax.annotate(f'{t:.3f}', (e, t), textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=9, color=colors['train'])
        ax.annotate(f'{v:.3f}', (e, v), textcoords="offset points", xytext=(0, -15),
                    ha='center', fontsize=9, color=colors['val'])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_all_losses(history: dict, save_path: str = None):
    """Plot all loss components together."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    epochs = history['epoch']

    # Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_total_loss'], marker='o', linewidth=2.5,
            label='Train', color=colors['train'], markersize=8)
    ax.plot(epochs, history['val_total_loss'], marker='s', linewidth=2.5,
            label='Val', color=colors['val'], markersize=8)
    ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax.set_title('Total Loss', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # MLM Loss
    ax = axes[0, 1]
    ax.plot(epochs, history['train_mlm_loss'], marker='o', linewidth=2.5,
            label='Train', color=colors['mlm'], markersize=8)
    ax.plot(epochs, history['val_mlm_loss'], marker='s', linewidth=2.5,
            label='Val', color=colors['edge'], markersize=8)
    ax.set_ylabel('MLM Loss', fontsize=11, fontweight='bold')
    ax.set_title('MLM Loss Component', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Edge Loss
    ax = axes[1, 0]
    ax.plot(epochs, history['train_edge_loss'], marker='o', linewidth=2.5,
            label='Train', color=colors['train'], markersize=8)
    ax.plot(epochs, history['val_edge_loss'], marker='s', linewidth=2.5,
            label='Val', color=colors['val'], markersize=8)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Edge Loss', fontsize=11, fontweight='bold')
    ax.set_title('Edge Prediction Loss Component', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Learning Rate
    ax = axes[1, 1]
    lr_scaled = [lr * 1e6 for lr in history['learning_rate']]
    ax.plot(epochs, lr_scaled, marker='o', linewidth=2.5, color=colors['lr'],
            markersize=8, label='Learning Rate')
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Learning Rate (×10⁻⁶)', fontsize=11, fontweight='bold')
    ax.set_title('Learning Rate Warmup Schedule', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.suptitle('GraphCodeBERT Training: All Metrics', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_train_vs_val(history: dict, save_path: str = None):
    """Plot train vs validation comparison (bar chart)."""
    fig, ax = plt.subplots(figsize=(14, 7))

    epochs = history['epoch']
    train_loss = history['train_total_loss']
    val_loss = history['val_total_loss']

    x = np.arange(len(epochs))
    width = 0.35

    bars1 = ax.bar(x - width / 2, train_loss, width, label='Train Total',
                   color=colors['train'], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width / 2, val_loss, width, label='Val Total',
                   color=colors['val'], alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Train vs Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(epochs)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

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
    ax1.plot(train_batches, linewidth=1, color=colors['train'], alpha=0.7, label='Train Batch Losses')
    ax1.fill_between(range(len(train_batches)), train_batches, alpha=0.2, color=colors['train'])
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Batch Losses - All Iterations', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # Validation batch losses
    ax2.plot(val_batches, linewidth=1, color=colors['val'], alpha=0.7, label='Val Batch Losses')
    ax2.fill_between(range(len(val_batches)), val_batches, alpha=0.2, color=colors['val'])
    ax2.set_xlabel('Batch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Batch Losses - All Iterations', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_loss_decomposition(history: dict, save_path: str = None):
    """Plot how MLM and Edge loss contribute to total loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    epochs = history['epoch']

    # Training loss decomposition
    ax1.bar(epochs, history['train_mlm_loss'], label='MLM Loss',
            color=colors['mlm'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.bar(epochs, history['train_edge_loss'], bottom=history['train_mlm_loss'],
            label='Edge Loss', color=colors['train'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Loss Decomposition', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    # Validation loss decomposition
    ax2.bar(epochs, history['val_mlm_loss'], label='MLM Loss',
            color=colors['edge'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.bar(epochs, history['val_edge_loss'], bottom=history['val_mlm_loss'],
            label='Edge Loss', color=colors['val'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Loss Decomposition', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Loss Component Contributions', fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def print_summary(history: dict):
    """Print training summary statistics."""
    print("\n" + "=" * 70)
    print("GraphCodeBERT Training Summary".center(70))
    print("=" * 70)

    train_loss = history['train_total_loss']
    val_loss = history['val_total_loss']
    best_val = history['best_val_loss']
    best_epoch = history['best_epoch']

    print(f"\nTraining Loss:")
    print(f"  Initial: {train_loss[0]:.4f}")
    print(f"  Final:   {train_loss[-1]:.4f}")
    print(f"  Reduction: {((train_loss[0] - train_loss[-1]) / train_loss[0] * 100):.1f}%")
    print(f"  Min:     {min(train_loss):.4f}")

    print(f"\nValidation Loss:")
    print(f"  Initial: {val_loss[0]:.4f}")
    print(f"  Final:   {val_loss[-1]:.4f}")
    print(f"  Best:    {best_val:.4f} (Epoch {best_epoch})")
    print(f"  Reduction: {((val_loss[0] - val_loss[-1]) / val_loss[0] * 100):.1f}%")

    print(f"\nComponent Losses (Final Epoch):")
    print(f"  Train MLM Loss:   {history['train_mlm_loss'][-1]:.4f}")
    print(f"  Train Edge Loss:  {history['train_edge_loss'][-1]:.4f}")
    print(f"  Val MLM Loss:     {history['val_mlm_loss'][-1]:.4f}")
    print(f"  Val Edge Loss:    {history['val_edge_loss'][-1]:.4f}")

    print(f"\nLearning Rate Range:")
    print(f"  Min: {min(history['learning_rate']):.2e}")
    print(f"  Max: {max(history['learning_rate']):.2e}")

    print(f"\nDataset Metrics:")
    print(f"  Total Training Batches:   {len(history['train_batch_losses'])}")
    print(f"  Total Validation Batches: {len(history['val_batch_losses'])}")

    print("\n" + "=" * 70 + "\n")


def main():
    """Main function to generate all plots."""
    import argparse

    parser = argparse.ArgumentParser(description='Plot training metrics from training_history.json')
    parser.add_argument('--history_file', type=str, default='/Users/czapmate/Desktop/szakdoga/GraphCodeBert_CPP/BERTModels/GraphCodeBert/Models/graphcodebert-cpp-mlm-from-config/training_history.json',
                        help='Path to training_history.json file')
    parser.add_argument('--output_dir', type=str, default='graph_training_plots',
                        help='Directory to save plots')
    parser.add_argument('--all', action='store_true', help='Generate all plots')
    parser.add_argument('--total', action='store_true', help='Plot total loss')
    parser.add_argument('--mlm', action='store_true', help='Plot MLM loss')
    parser.add_argument('--edge', action='store_true', help='Plot edge loss')
    parser.add_argument('--compare', action='store_true', help='Plot train vs val comparison')
    parser.add_argument('--batches', action='store_true', help='Plot batch losses')
    parser.add_argument('--decomposition', action='store_true', help='Plot loss decomposition')
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

    generate_all = args.all or not any(
        [args.total, args.mlm, args.edge, args.compare, args.batches, args.decomposition])

    try:
        if args.total or generate_all:
            print("Generating total loss plot...")
            plot_total_loss(history, output_dir / 'total_loss.png')

        if args.mlm or generate_all:
            print("Generating MLM loss plot...")
            plot_mlm_loss(history, output_dir / 'mlm_loss.png')

        if args.edge or generate_all:
            print("Generating edge loss plot...")
            plot_edge_loss(history, output_dir / 'edge_loss.png')

        if args.compare or generate_all:
            print("Generating train vs val comparison...")
            plot_train_vs_val(history, output_dir / 'train_vs_val.png')

        if args.batches or generate_all:
            print("Generating batch losses plot...")
            plot_batch_losses(history, output_dir / 'batch_losses.png')

        if args.decomposition or generate_all:
            print("Generating loss decomposition plot...")
            plot_loss_decomposition(history, output_dir / 'loss_decomposition.png')

        # Always generate the comprehensive 4-plot figure
        if generate_all:
            print("Generating comprehensive metrics plot...")
            plot_all_losses(history, output_dir / 'all_metrics.png')

        print(f"\n✓ All plots saved to: {output_dir}")
    except Exception as e:
        print(f"Error generating plots: {e}")


if __name__ == "__main__":
    main()