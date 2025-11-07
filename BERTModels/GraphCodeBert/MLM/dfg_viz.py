"""
Visualization script for GraphCodeBERT model architecture and training metrics.
Generates:
- Model architecture diagram
- Layer structure with parameters
- Attention patterns
- DFG visualization
- Training loss curves (if TensorBoard logs available)
"""
import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional
from transformers import RobertaForMaskedLM, RobertaTokenizer
import numpy as np


class GraphCodeBERTVisualizer:
    def __init__(self, model_path: str = "microsoft/graphcodebert-base",
                 edge_classifier_path: Optional[str] = None):
        """
        Initialize visualizer with model path.

        Args:
            model_path: Path to saved model or HuggingFace model ID
            edge_classifier_path: Path to saved edge_classifier.pt weights
        """
        print(f"Loading model from {model_path}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "mps" if torch.backends.mps.is_available() else "cpu")

        self.roberta = RobertaForMaskedLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)

        self.edge_classifier = None
        if edge_classifier_path and os.path.exists(edge_classifier_path):
            print(f"Loading edge classifier from {edge_classifier_path}...")
            hidden_size = self.roberta.config.hidden_size
            self.edge_classifier = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, 1)
            )
            self.edge_classifier.load_state_dict(torch.load(edge_classifier_path, map_location=self.device))
            self.edge_classifier.to(self.device)

        print(f"Model loaded on device: {self.device}")

    def visualize_model_architecture(self, save_path: str = "model_architecture.png"):
        """Generate and save model architecture diagram."""
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')

        y_pos = 11

        # Title
        ax.text(5, y_pos, 'GraphCodeBERT Model Architecture',
                fontsize=16, fontweight='bold', ha='center')
        y_pos -= 0.8

        # Input layer
        self._draw_box(ax, 5, y_pos, 3, 0.6, "Input Layer", "lightblue")
        ax.text(5, y_pos - 0.9, "• input_ids\n• attention_mask\n• position_ids",
                fontsize=9, ha='center', va='top')
        y_pos -= 2.2

        # RoBERTa encoder
        num_layers = self.roberta.config.num_hidden_layers
        hidden_size = self.roberta.config.hidden_size

        self._draw_box(ax, 5, y_pos, 3.5, 0.7,
                      f"RoBERTa Encoder ({num_layers} layers)", "lightgreen")
        ax.text(5, y_pos - 1.1, f"Hidden size: {hidden_size}\nVocab size: {self.roberta.config.vocab_size}",
                fontsize=9, ha='center', va='top')
        y_pos -= 2.0

        # MLM head
        self._draw_box(ax, 2.5, y_pos, 2.5, 0.6, "MLM Head", "lightyellow")
        ax.text(2.5, y_pos - 0.9, f"Output: vocab_size\n({self.roberta.config.vocab_size})",
                fontsize=8, ha='center', va='top')

        # Edge prediction head
        if self.edge_classifier:
            self._draw_box(ax, 7.5, y_pos, 2.5, 0.6, "Edge Classifier", "lightcoral")
            ax.text(7.5, y_pos - 0.9, "Output: edge score\n(0 or 1)",
                    fontsize=8, ha='center', va='top')

        y_pos -= 1.8

        # Output
        ax.text(2.5, y_pos, "MLM Loss", fontsize=9, ha='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        if self.edge_classifier:
            ax.text(7.5, y_pos, "Edge Loss", fontsize=9, ha='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model architecture saved to {save_path}")
        plt.close()

    def visualize_layer_details(self, save_path: str = "layer_details.png"):
        """Generate detailed layer information visualization."""
        fig, ax = plt.subplots(figsize=(12, 14))
        ax.axis('off')

        y_pos = 0.95

        # Title
        ax.text(0.5, y_pos, 'GraphCodeBERT - Layer Details',
                fontsize=14, fontweight='bold', ha='center', transform=ax.transAxes)
        y_pos -= 0.05

        # Config info
        config_text = f"""
Configuration:
  • Model: microsoft/graphcodebert-base
  • Hidden size: {self.roberta.config.hidden_size}
  • Number of layers: {self.roberta.config.num_hidden_layers}
  • Attention heads: {self.roberta.config.num_attention_heads}
  • Intermediate size: {self.roberta.config.intermediate_size}
  • Vocab size: {self.roberta.config.vocab_size}
  • Max position embeddings: {self.roberta.config.max_position_embeddings}
        """
        ax.text(0.05, y_pos, config_text, fontsize=10, family='monospace',
                verticalalignment='top', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        y_pos -= 0.22

        # Encoder layers
        encoder_text = "RoBERTa Encoder Layers:\n"
        encoder_text += f"{'Layer':<8} {'Type':<20} {'Parameters':<15}\n"
        encoder_text += "─" * 50 + "\n"

        total_params = 0
        for i, layer in enumerate(self.roberta.roberta.encoder.layer[:3]):  # Show first 3
            attention_params = sum(p.numel() for p in layer.attention.parameters())
            output_params = sum(p.numel() for p in layer.output.parameters())
            intermediate_params = sum(p.numel() for p in layer.intermediate.parameters())
            layer_params = attention_params + output_params + intermediate_params
            total_params += layer_params
            encoder_text += f"{i:<8} {'Self-Attention':<20} {attention_params:<15,}\n"
            encoder_text += f"{'':8} {'Feed-Forward':<20} {intermediate_params + output_params:<15,}\n"

        encoder_text += "─" * 50 + "\n"
        encoder_text += f"{'Total params (first 3 layers):':<28} {total_params:,}\n"
        encoder_text += f"{'Full model params:':<28} {sum(p.numel() for p in self.roberta.parameters()):,}\n"

        ax.text(0.05, y_pos, encoder_text, fontsize=9, family='monospace',
                verticalalignment='top', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        y_pos -= 0.25

        # MLM head details
        mlm_text = "MLM Head:\n"
        mlm_text += f"{'Layer':<20} {'Type':<20} {'Parameters':<15}\n"
        mlm_text += "─" * 55 + "\n"
        mlm_text += f"{'lm_head':<20} {'Linear':<20} {self.roberta.lm_head.dense.in_features * self.roberta.lm_head.dense.out_features:,}\n"
        mlm_text += f"{'LayerNorm':<20} {'LayerNorm':<20} {sum(p.numel() for p in self.roberta.lm_head.layer_norm.parameters()):,}\n"
        mlm_text += f"{'Decoder':<20} {'Linear':<20} {self.roberta.lm_head.decoder.weight.numel():,}\n"

        ax.text(0.05, y_pos, mlm_text, fontsize=9, family='monospace',
                verticalalignment='top', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        y_pos -= 0.15

        # Edge classifier details
        if self.edge_classifier:
            edge_text = "Edge Classifier:\n"
            edge_text += f"{'Layer':<20} {'Type':<20} {'Parameters':<15}\n"
            edge_text += "─" * 55 + "\n"
            for i, layer in enumerate(self.edge_classifier):
                if isinstance(layer, nn.Linear):
                    params = layer.weight.numel() + layer.bias.numel()
                    edge_text += f"{'Linear_{i}':<20} {'Linear':<20} {params:,}\n"

            ax.text(0.05, y_pos, edge_text, fontsize=9, family='monospace',
                    verticalalignment='top', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Layer details saved to {save_path}")
        plt.close()

    def visualize_attention_mask(self, attention_mask: np.ndarray,
                                 save_path: str = "attention_mask.png"):
        """Visualize attention mask pattern."""
        fig, ax = plt.subplots(figsize=(10, 10))

        im = ax.imshow(attention_mask, cmap='Blues', aspect='auto')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Token Position')
        ax.set_title('Attention Mask Pattern\n(White = attention allowed, Blue = gradient of attention)')

        plt.colorbar(im, ax=ax, label='Attention')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention mask saved to {save_path}")
        plt.close()

    def visualize_dfg_structure(self, nodes: List, edges: List,
                               save_path: str = "dfg_structure.png"):
        """Visualize dataflow graph structure."""
        fig, ax = plt.subplots(figsize=(10, 8))

        if not nodes or not edges:
            ax.text(0.5, 0.5, 'No DFG nodes/edges to visualize',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return

        # Simple circular layout for nodes
        num_nodes = len(nodes)
        angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
        positions = {i: (np.cos(angles[i]), np.sin(angles[i])) for i in range(num_nodes)}

        # Draw edges
        for src, dst in edges:
            x_vals = [positions[src][0], positions[dst][0]]
            y_vals = [positions[src][1], positions[dst][1]]
            ax.plot(x_vals, y_vals, 'k-', alpha=0.3, linewidth=1)
            # Arrow
            ax.annotate('', xy=positions[dst], xytext=positions[src],
                       arrowprops=dict(arrowstyle='->', lw=1, color='gray', alpha=0.5))

        # Draw nodes
        for i, (var, pos) in enumerate(nodes):
            x, y = positions[i]
            circle = plt.Circle((x, y), 0.08, color='lightblue', ec='blue', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y - 0.15, f'{i}\n{var[:10]}', ha='center', fontsize=8)

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Dataflow Graph (DFG) Structure', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"DFG structure saved to {save_path}")
        plt.close()

    def _draw_box(self, ax, x, y, width, height, text, color):
        """Helper to draw boxes in architecture diagram."""
        rect = mpatches.FancyBboxPatch((x - width/2, y - height/2), width, height,
                                       boxstyle="round,pad=0.1",
                                       edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')

    def print_model_summary(self):
        """Print detailed model summary."""
        print("\n" + "=" * 70)
        print("GraphCodeBERT Model Summary".center(70))
        print("=" * 70)

        print("\n[RoBERTa Configuration]")
        for key in ['hidden_size', 'num_hidden_layers', 'num_attention_heads',
                   'intermediate_size', 'vocab_size', 'max_position_embeddings']:
            print(f"  {key:<25} {getattr(self.roberta.config, key)}")

        print("\n[Model Parameters]")
        total_params = sum(p.numel() for p in self.roberta.parameters())
        trainable_params = sum(p.numel() for p in self.roberta.parameters() if p.requires_grad)
        print(f"  Total parameters:        {total_params:,}")
        print(f"  Trainable parameters:    {trainable_params:,}")

        if self.edge_classifier:
            edge_params = sum(p.numel() for p in self.edge_classifier.parameters())
            print(f"  Edge classifier params:  {edge_params:,}")
            print(f"  Total (with edge clf):   {total_params + edge_params:,}")

        print("\n[Model Architecture]")
        print(self.roberta)

        if self.edge_classifier:
            print("\n[Edge Classifier Architecture]")
            print(self.edge_classifier)

        print("\n" + "=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize GraphCodeBERT model')
    parser.add_argument('--model_path', type=str, default="microsoft/graphcodebert-base",
                       help='Path to model or HuggingFace model ID')
    parser.add_argument('--edge_classifier', type=str, default=None,
                       help='Path to edge_classifier.pt weights')
    parser.add_argument('--output_dir', type=str, default="./visualizations",
                       help='Output directory for visualizations')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)

    visualizer = GraphCodeBERTVisualizer(args.model_path, args.edge_classifier)
    visualizer.print_model_summary()

    print("\nGenerating visualizations...")
    visualizer.visualize_model_architecture(f"{args.output_dir}/01_architecture.png")
    visualizer.visualize_layer_details(f"{args.output_dir}/02_layer_details.png")

    # Example attention mask
    attention_mask = np.ones((512, 512), dtype=bool)
    visualizer.visualize_attention_mask(attention_mask, f"{args.output_dir}/03_attention_mask_example.png")

    # Example DFG
    example_nodes = [("var1", 5), ("var2", 10), ("var3", 15), ("var4", 20)]
    example_edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
    visualizer.visualize_dfg_structure(example_nodes, example_edges,
                                       f"{args.output_dir}/04_dfg_structure_example.png")

    print(f"\nAll visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()