"""
Visualize GraphCodeBERT model architecture - both simplified and complex versions
Saves visualizations as PNG/PDF files
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Import the model class
import torch.nn as nn
from transformers import RobertaForMaskedLM, RobertaTokenizer


class GraphCodeBERTWithEdgePrediction(nn.Module):
    """Load from document"""
    def __init__(self, base_model_name: str = "microsoft/graphcodebert-base"):
        super().__init__()
        self.roberta_mlm = RobertaForMaskedLM.from_pretrained(base_model_name)
        hidden_size = self.roberta_mlm.config.hidden_size
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask, position_ids, labels=None,
                edge_batch_idx=None, edge_node1_pos=None, edge_node2_pos=None, edge_labels=None):
        mlm_outputs = self.roberta_mlm(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, labels=labels, output_hidden_states=True
        )
        mlm_loss = mlm_outputs.loss if labels is not None else None

        edge_loss = None
        if (edge_batch_idx is not None and len(edge_batch_idx) > 0 and
            edge_node1_pos is not None and edge_node2_pos is not None and edge_labels is not None):
            hidden_states = mlm_outputs.hidden_states[-1]
            batch_size, seq_len, hidden_size = hidden_states.shape

            node1_repr = hidden_states[edge_batch_idx, edge_node1_pos]
            node2_repr = hidden_states[edge_batch_idx, edge_node2_pos]
            edge_repr = torch.cat([node1_repr, node2_repr], dim=-1)
            edge_logits = self.edge_classifier(edge_repr).squeeze(-1)
            edge_loss = nn.functional.binary_cross_entropy_with_logits(edge_logits, edge_labels)

        if mlm_loss is not None and edge_loss is not None:
            total_loss = mlm_loss + edge_loss
        elif mlm_loss is not None:
            total_loss = mlm_loss
        else:
            total_loss = edge_loss

        return total_loss, mlm_loss, edge_loss

    def save_pretrained(self, save_directory):
        self.roberta_mlm.save_pretrained(save_directory)
        torch.save(self.edge_classifier.state_dict(), f"{save_directory}/edge_classifier.pt")


def install_dependencies():
    """Install required visualization libraries"""
    import subprocess
    try:
        import torchviz
        import torchsummary
    except ImportError:
        print("Installing visualization dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                             "torchviz", "torch-summary", "graphviz"])


def visualize_simplified_architecture(output_dir="visualizations"):
    """
    Create a simplified, hand-drawn style architecture diagram
    Shows main components: Input -> RoBERTa -> MLM Head + Edge Classifier
    """
    Path(output_dir).mkdir(exist_ok=True)

    try:
        from graphviz import Digraph
    except ImportError:
        print("graphviz not installed. Skipping simplified diagram.")
        return

    dot = Digraph(comment='GraphCodeBERT Simplified Architecture',
                  format='png', engine='dot')
    dot.attr(rankdir='TB', size='12,8')
    dot.attr('node', shape='box', style='rounded,filled',
             fillcolor='lightblue', fontsize='12', fontname='Arial')

    # Input layer
    dot.node('input', 'Input\n(input_ids, attention_mask,\nposition_ids)',
             fillcolor='lightgreen', shape='box')

    # RoBERTa backbone
    dot.node('roberta', 'RoBERTa-base\n(12 Transformer Blocks)\nHidden: 768 dim',
             fillcolor='lightyellow')

    # MLM head
    dot.node('mlm_head', 'MLM Head\n(Masked Language Model)\nPredicts masked tokens',
             fillcolor='lightcoral')

    # Edge Classifier
    dot.node('edge_input', 'Edge Pairs\n(node1, node2 embeddings)',
             fillcolor='lightcyan')
    dot.node('edge_fc1', 'Linear + Tanh\n(768*2 -> 768)',
             fillcolor='plum')
    dot.node('edge_dropout', 'Dropout (0.1)', fillcolor='plum')
    dot.node('edge_fc2', 'Linear + Sigmoid\n(768 -> 1)',
             fillcolor='plum')
    dot.node('edge_output', 'Edge Prediction\n(Binary Classification)',
             fillcolor='lightcoral')

    # Loss computation
    dot.node('mlm_loss', 'MLM Loss\n(Cross-Entropy)', fillcolor='salmon')
    dot.node('edge_loss', 'Edge Loss\n(Binary Cross-Entropy)', fillcolor='salmon')
    dot.node('total_loss', 'Total Loss\n(MLM Loss + Edge Loss)',
             fillcolor='red', fontcolor='white')

    # Connections
    dot.edge('input', 'roberta')
    dot.edge('roberta', 'mlm_head')
    dot.edge('roberta', 'edge_input')
    dot.edge('edge_input', 'edge_fc1')
    dot.edge('edge_fc1', 'edge_dropout')
    dot.edge('edge_dropout', 'edge_fc2')
    dot.edge('edge_fc2', 'edge_output')

    dot.edge('mlm_head', 'mlm_loss')
    dot.edge('edge_output', 'edge_loss')
    dot.edge('mlm_loss', 'total_loss')
    dot.edge('edge_loss', 'total_loss')

    output_path = f"{output_dir}/graphcodebert_simplified"
    dot.render(output_path, cleanup=True)
    print(f"✓ Simplified architecture saved to {output_path}.png")


def visualize_detailed_architecture(output_dir="visualizations"):
    """
    Create a detailed architecture diagram showing layer breakdown
    Includes transformer block details and full edge classifier
    """
    Path(output_dir).mkdir(exist_ok=True)

    try:
        from graphviz import Digraph
    except ImportError:
        print("graphviz not installed. Skipping detailed diagram.")
        return

    dot = Digraph(comment='GraphCodeBERT Detailed Architecture',
                  format='png', engine='dot')
    dot.attr(rankdir='TB', size='16,12')
    dot.attr('node', shape='box', style='rounded,filled',
             fontsize='10', fontname='Arial')

    # Input layer
    dot.node('input', 'Inputs', fillcolor='lightgreen')
    dot.node('input_ids', 'input_ids\n(batch, 512)', fillcolor='lightgreen')
    dot.node('attn_mask', 'attention_mask\n(batch, 512, 512)', fillcolor='lightgreen')
    dot.node('pos_ids', 'position_ids\n(batch, 512)', fillcolor='lightgreen')
    dot.node('dfg_info', 'dfg_info\n(dataflow graph)', fillcolor='lightgreen')

    dot.edge('input', 'input_ids')
    dot.edge('input', 'attn_mask')
    dot.edge('input', 'pos_ids')
    dot.edge('input', 'dfg_info')

    # RoBERTa embedding + blocks
    dot.node('embeddings', 'Token Embeddings\n+ Position Embeddings\n-> (batch, 512, 768)',
             fillcolor='lightyellow')

    # Simplified transformer blocks
    dot.node('transformer_blocks', 'Transformer Blocks × 12\n' +
             ''.join([f'Block {i+1}: MultiHeadAttention (12 heads) + FFN\n'
                     for i in range(3)]) + '...\nOutput: (batch, 512, 768)',
             fillcolor='lightyellow')

    dot.edge('input_ids', 'embeddings')
    dot.edge('pos_ids', 'embeddings')
    dot.edge('embeddings', 'transformer_blocks')

    # MLM head
    dot.node('mlm_input', 'Hidden States\n(batch, 512, 768)', fillcolor='lightcyan')
    dot.node('mlm_dense', 'Dense Layer\n(768 -> 768)', fillcolor='lightcoral')
    dot.node('mlm_activation', 'Activation (GELU)', fillcolor='lightcoral')
    dot.node('mlm_norm', 'LayerNorm', fillcolor='lightcoral')
    dot.node('mlm_output', 'Output Projection\n(768 -> vocab_size)\nSoftmax',
             fillcolor='lightcoral')

    dot.edge('transformer_blocks', 'mlm_input')
    dot.edge('mlm_input', 'mlm_dense')
    dot.edge('mlm_dense', 'mlm_activation')
    dot.edge('mlm_activation', 'mlm_norm')
    dot.edge('mlm_norm', 'mlm_output')

    # Edge Classifier detailed
    dot.node('edge_gather', 'Gather Node Embeddings\nnode1: (batch, 768)\nnode2: (batch, 768)',
             fillcolor='lightcyan')
    dot.node('edge_concat', 'Concatenate\n(batch, 1536)', fillcolor='plum')
    dot.node('edge_fc1_detail', 'Linear Layer\n(1536 -> 768)', fillcolor='plum')
    dot.node('edge_activation', 'Activation (Tanh)', fillcolor='plum')
    dot.node('edge_dropout_detail', 'Dropout (p=0.1)', fillcolor='plum')
    dot.node('edge_fc2_detail', 'Linear Layer\n(768 -> 1)', fillcolor='plum')
    dot.node('edge_sigmoid', 'Sigmoid', fillcolor='plum')

    dot.edge('transformer_blocks', 'edge_gather')
    dot.edge('dfg_info', 'edge_gather')
    dot.edge('edge_gather', 'edge_concat')
    dot.edge('edge_concat', 'edge_fc1_detail')
    dot.edge('edge_fc1_detail', 'edge_activation')
    dot.edge('edge_activation', 'edge_dropout_detail')
    dot.edge('edge_dropout_detail', 'edge_fc2_detail')
    dot.edge('edge_fc2_detail', 'edge_sigmoid')

    # Loss computation
    dot.node('mlm_loss', 'MLM Loss\nCross-Entropy', fillcolor='salmon')
    dot.node('edge_loss', 'Edge Loss\nBinary Cross-Entropy', fillcolor='salmon')
    dot.node('total_loss', 'Total Loss\nMLM + Edge', fillcolor='red', fontcolor='white')
    dot.node('backprop', 'Backpropagation &\nOptimizer Step', fillcolor='darkred', fontcolor='white')

    dot.edge('mlm_output', 'mlm_loss')
    dot.edge('edge_sigmoid', 'edge_loss')
    dot.edge('mlm_loss', 'total_loss')
    dot.edge('edge_loss', 'total_loss')
    dot.edge('total_loss', 'backprop')

    output_path = f"{output_dir}/graphcodebert_detailed"
    dot.render(output_path, cleanup=True)
    print(f"✓ Detailed architecture saved to {output_path}.png")


def print_model_summary(model):
    """Print comprehensive model summary to console"""
    print("\n" + "="*80)
    print("GRAPHCODEBERT WITH EDGE PREDICTION - MODEL SUMMARY")
    print("="*80 + "\n")

    # Overall statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Device: {next(model.parameters()).device}\n")

    # RoBERTa component
    print("-" * 80)
    print("RoBERTa MLM Component:")
    print("-" * 80)
    print(f"Base Model: microsoft/graphcodebert-base")
    print(f"Architecture: 12-layer Transformer")
    print(f"Hidden Size: 768")
    print(f"Attention Heads: 12")
    print(f"FFN Dimension: 3072")
    print(f"Vocab Size: {model.roberta_mlm.config.vocab_size}")

    roberta_params = sum(p.numel() for p in model.roberta_mlm.parameters())
    print(f"Total RoBERTa Parameters: {roberta_params:,}\n")

    # Edge Classifier component
    print("-" * 80)
    print("Edge Classifier Component:")
    print("-" * 80)
    print(model.edge_classifier)

    edge_params = sum(p.numel() for p in model.edge_classifier.parameters())
    print(f"Total Edge Classifier Parameters: {edge_params:,}\n")

    # Layer breakdown
    print("-" * 80)
    print("Edge Classifier Layer Breakdown:")
    print("-" * 80)
    for i, (name, module) in enumerate(model.edge_classifier.named_modules()):
        if isinstance(module, (nn.Linear, nn.Dropout, nn.Tanh)):
            param_count = sum(p.numel() for p in module.parameters())
            print(f"  {name}: {module} - {param_count:,} params")

    print("\n" + "="*80 + "\n")


def load_and_visualize(model_path=None):
    """Load model and create visualizations"""
    print("Loading GraphCodeBERT model...")

    try:
        model = GraphCodeBERTWithEdgePrediction("microsoft/graphcodebert-base")

        if model_path and Path(model_path).exists():
            print(f"Loading weights from {model_path}...")
            # Load RoBERTa weights
            roberta_path = Path(model_path)
            if (roberta_path / "pytorch_model.bin").exists():
                model.roberta_mlm = RobertaForMaskedLM.from_pretrained(str(roberta_path))

            # Load edge classifier weights
            edge_classifier_path = Path(model_path) / "edge_classifier.pt"
            if edge_classifier_path.exists():
                model.edge_classifier.load_state_dict(torch.load(edge_classifier_path, map_location='cpu'))
                print(f"✓ Loaded edge classifier from {edge_classifier_path}")

            print(f"✓ Model loaded from {model_path}")
        else:
            print("✓ Model initialized with pretrained weights (no checkpoint provided)")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    # Print summary
    print_model_summary(model)

    # Create visualizations
    print("Creating visualizations...")
    visualize_simplified_architecture()
    visualize_detailed_architecture()

    print("\n✓ All visualizations complete!")
    print("  - visualizations/graphcodebert_simplified.png")
    print("  - visualizations/graphcodebert_detailed.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize GraphCodeBERT Architecture')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to saved model checkpoint')
    args = parser.parse_args()

    install_dependencies()
    load_and_visualize(args.model_path)