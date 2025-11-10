"""
Visualize UniXcoder model architecture with detailed layer information.
Highlights the lm_head component and all other layers.
"""
import torch
from transformers import RobertaForMaskedLM, RobertaTokenizer
from torch.nn import Module
import json
from pathlib import Path


def get_model_architecture(model: Module, indent=0, parent_name=""):
    """
    Recursively traverse model architecture and return detailed structure.
    """
    architecture = []
    indent_str = "  " * indent

    for name, module in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        module_type = module.__class__.__name__

        # Get parameter count
        param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)

        # Special highlighting for lm_head
        highlight = "⭐ LM_HEAD ⭐" if "lm_head" in name.lower() else ""

        # Get shape info for common layer types
        shape_info = ""
        if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
            shape_info = f"({module.in_features} → {module.out_features})"
        elif hasattr(module, 'num_features'):
            shape_info = f"(features={module.num_features})"
        elif hasattr(module, 'num_heads'):
            shape_info = f"(heads={module.num_heads})"

        line = f"{indent_str}├─ {name}: {module_type} {shape_info} [{param_count:,} params] {highlight}"
        architecture.append(line)

        # Recursively get child modules
        children = get_model_architecture(module, indent + 1, full_name)
        architecture.extend(children)

    return architecture


def print_architecture(model: Module):
    """Print complete model architecture."""
    print("\n" + "=" * 100)
    print("UNIXCODER MODEL ARCHITECTURE - ROBERTA FOR MASKED LM")
    print("=" * 100 + "\n")

    print(f"Model Type: {model.__class__.__name__}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    architecture = get_model_architecture(model)
    for line in architecture:
        print(line)

    print("\n" + "=" * 100 + "\n")


def analyze_lm_head(model: Module):
    """Deep dive into the lm_head structure."""
    if hasattr(model, 'lm_head'):
        lm_head = model.lm_head

        print("=" * 100)
        print("⭐ LM_HEAD DETAILED ANALYSIS ⭐")
        print("=" * 100 + "\n")

        print(f"LM Head Type: {lm_head.__class__.__name__}")
        print(f"LM Head Total Parameters: {sum(p.numel() for p in lm_head.parameters()):,}\n")

        print("LM Head Sub-layers:")
        for name, module in lm_head.named_modules():
            if name:  # Skip the root module
                param_count = sum(p.numel() for p in module.parameters())
                print(f"  {name}: {module.__class__.__name__}")

                if hasattr(module, 'weight'):
                    print(f"    - Weight shape: {module.weight.shape}")
                if hasattr(module, 'bias') and module.bias is not None:
                    print(f"    - Bias shape: {module.bias.shape}")
                print(f"    - Parameters: {param_count:,}\n")
    else:
        print("⚠️  No lm_head found in model!")


def get_layer_summary(model: Module):
    """Generate a summary of layer types and counts."""
    layer_types = {}

    for module in model.modules():
        module_type = module.__class__.__name__
        if module_type not in ['RobertaForMaskedLM', 'RobertaModel']:
            layer_types[module_type] = layer_types.get(module_type, 0) + 1

    print("\n" + "=" * 100)
    print("LAYER TYPE SUMMARY")
    print("=" * 100 + "\n")

    for layer_type in sorted(layer_types.keys()):
        print(f"  {layer_type}: {layer_types[layer_type]}")

    print("\n" + "=" * 100 + "\n")


def visualize_tensor_flow(model: Module):
    """Show the tensor flow through the model."""
    print("\n" + "=" * 100)
    print("TENSOR FLOW (Expected Shapes)")
    print("=" * 100 + "\n")

    print("Input: [batch_size, seq_length]")
    print("  ↓")
    print("Embedding Layer: [batch_size, seq_length, 768]")
    print("  ↓")
    print("RoBERTa Encoder (12 layers): [batch_size, seq_length, 768]")
    print("  ├─ Each layer: MultiHeadAttention + FeedForward")
    print("  └─ Output: [batch_size, seq_length, 768]")
    print("  ↓")
    print("⭐ LM Head: [batch_size, seq_length, vocab_size]")
    print("  ├─ Dense: [batch_size, seq_length, 768]")
    print("  ├─ LayerNorm: [batch_size, seq_length, 768]")
    print("  └─ Linear (Decoder): [batch_size, seq_length, 50265] (vocab_size)")
    print("\n" + "=" * 100 + "\n")


def save_architecture_to_file(model: Module, output_path: str = "model_architecture.txt"):
    """Save architecture to a text file."""
    with open(output_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("UNIXCODER MODEL ARCHITECTURE\n")
        f.write("=" * 100 + "\n\n")

        # Model info
        f.write(f"Model Type: {model.__class__.__name__}\n")
        f.write(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
        f.write(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n\n")

        # Architecture
        architecture = get_model_architecture(model)
        for line in architecture:
            f.write(line + "\n")

        # Layer summary
        f.write("\n" + "=" * 100 + "\n")
        f.write("LAYER TYPE SUMMARY\n")
        f.write("=" * 100 + "\n\n")

        layer_types = {}
        for module in model.modules():
            module_type = module.__class__.__name__
            if module_type not in ['RobertaForMaskedLM', 'RobertaModel']:
                layer_types[module_type] = layer_types.get(module_type, 0) + 1

        for layer_type in sorted(layer_types.keys()):
            f.write(f"{layer_type}: {layer_types[layer_type]}\n")

    print(f"✓ Architecture saved to {output_path}")


def main():
    print("Loading UniXcoder base-nine...")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base-nine")
    model = RobertaForMaskedLM.from_pretrained("microsoft/unixcoder-base-nine")
    print("✓ Model loaded successfully!\n")

    # Display full architecture
    print_architecture(model)

    # Analyze lm_head
    analyze_lm_head(model)

    # Show layer summary
    get_layer_summary(model)

    # Show tensor flow
    visualize_tensor_flow(model)

    # Save to file
    save_architecture_to_file(model)


if __name__ == "__main__":
    main()