# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import os


class Model(nn.Module):
    """
    GraphCodeBERT wrapper for code search fine-tuning.
    Uses pre-trained encoder to generate embeddings for code and docstrings.
    """

    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder

    def forward(self, code_inputs=None, attention_mask=None, nl_inputs=None):
        """
        Forward pass for code search.

        Args:
            code_inputs: Token IDs of code snippets [batch_size, seq_len]
            attention_mask: Attention mask for code [batch_size, seq_len]
            nl_inputs: Token IDs of docstrings [batch_size, seq_len]

        Returns:
            embeddings: [batch_size, hidden_size]
        """
        if code_inputs is not None:
            # Process code: use [CLS] token representation (pooled)
            outputs = self.encoder(
                input_ids=code_inputs,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            # Use [CLS] token embedding (index 0)
            embeddings = outputs[1]  # Already pooled by model
            return embeddings
        else:
            # Process docstring: use [CLS] token representation (pooled)
            outputs = self.encoder(
                input_ids=nl_inputs,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            # Use [CLS] token embedding (index 0)
            embeddings = outputs[1]  # Already pooled by model
            return embeddings


def load_model_from_checkpoint(checkpoint_path, tokenizer_path=None):
    """
    Load pre-trained GraphCodeBERT model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (directory with config.json, pytorch_model.bin, etc.)
        tokenizer_path: Optional path to tokenizer (defaults to checkpoint_path)

    Returns:
        model: Wrapped Model instance
        tokenizer: Tokenizer instance
    """
    from transformers import AutoModel, AutoTokenizer

    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

    # Load tokenizer
    tok_path = tokenizer_path or checkpoint_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    # Load base model
    base_model = AutoModel.from_pretrained(checkpoint_path)

    # Wrap in Model class
    model = Model(base_model)

    return model, tokenizer