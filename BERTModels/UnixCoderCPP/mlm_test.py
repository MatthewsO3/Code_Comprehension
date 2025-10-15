"""
Utility scripts for inference and evaluation of trained UniXcoder-style MLM model
"""

import torch
import numpy as np
from typing import List, Tuple
from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn


# ============================================================
# ✅ Model Definition (same as training)
# ============================================================
class UniXcoderMLM(nn.Module):
    """UniXcoder + Custom MLM Head"""

    def __init__(self, base_model_name, vocab_size, hidden_size, device):
        super().__init__()
        self.device = device

        # Base encoder (UniXcoder backbone)
        self.encoder = RobertaModel.from_pretrained(base_model_name)

        # Custom MLM head
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size)
        ).to(device)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        logits = self.mlm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {'loss': loss, 'logits': logits}


# ============================================================
# ✅ Predictor Class
# ============================================================
class MLMPredictor:
    """Predict masked tokens in code"""

    def __init__(self, model_dir: str, device: str = None):
        # Select best available device
        if torch.backends.mps.is_available():
            device = torch.device("mps")  # Apple Silicon GPU
        elif torch.cuda.is_available():
            device = torch.device("cuda")  # NVIDIA GPU
        else:
            device = torch.device("cpu")  # CPU fallback
        self.device = device

        # Load tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(f"{model_dir}/tokenizer")

        # Load model architecture
        base_model_name = "microsoft/unixcoder-base"
        base_model = RobertaModel.from_pretrained(base_model_name)
        hidden_size = base_model.config.hidden_size

        self.model = UniXcoderMLM(base_model_name, len(self.tokenizer), hidden_size, device)
        state_dict = torch.load(f"{model_dir}/best_model.pt", map_location=device)
        self.model.load_state_dict(state_dict)

        self.model.to(device)
        self.model.eval()

        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id

    # ------------------------------------------------------------
    # Predict masked tokens
    # ------------------------------------------------------------
    def predict_masked_tokens(self, code: str, top_k: int = 5) -> List[Tuple[str, float]]:
        inputs = self.tokenizer(code, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        mask_positions = (input_ids == self.mask_token_id).nonzero(as_tuple=True)
        if len(mask_positions[0]) == 0:
            return []

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']

        results = []
        for batch_idx, seq_idx in zip(mask_positions[0], mask_positions[1]):
            probs = torch.softmax(logits[batch_idx, seq_idx], dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, top_k)
            tokens = [self.tokenizer.decode([i.item()]).strip() for i in top_k_indices]
            results.append(list(zip(tokens, top_k_probs.tolist())))

        return results

    # ------------------------------------------------------------
    # Fill all masks with top prediction
    # ------------------------------------------------------------
    def fill_masks(self, code: str) -> str:
        inputs = self.tokenizer(code, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']

        mask_positions = (input_ids == self.mask_token_id).nonzero(as_tuple=True)
        for batch_idx, seq_idx in zip(mask_positions[0], mask_positions[1]):
            predicted_token_id = logits[batch_idx, seq_idx].argmax(dim=-1)
            input_ids[batch_idx, seq_idx] = predicted_token_id

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    # ------------------------------------------------------------
    # Compute token-level perplexity
    # ------------------------------------------------------------
    def get_token_perplexity(self, code: str) -> float:
        inputs = self.tokenizer(code, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs['loss']

        perplexity = torch.exp(loss).item()
        return perplexity


# ============================================================
# ✅ Evaluation Utility
# ============================================================
def evaluate_mlm_accuracy(model_dir: str, test_dataset, num_samples: int = 1000):
    predictor = MLMPredictor(model_dir)

    correct_predictions = 0
    total_predictions = 0
    print(f"Evaluating on {num_samples} samples...")

    for i in range(min(num_samples, len(test_dataset))):
        code = test_dataset[i]['code']
        tokens = predictor.tokenizer.tokenize(code)
        if len(tokens) < 10:
            continue

        mask_idx = np.random.randint(1, min(len(tokens) - 1, 100))
        original_token = tokens[mask_idx]
        tokens[mask_idx] = predictor.mask_token
        masked_code = predictor.tokenizer.convert_tokens_to_string(tokens)

        predictions = predictor.predict_masked_tokens(masked_code, top_k=10)

        if predictions:
            top_predictions = [token for token, _ in predictions[0]]
            if original_token in top_predictions:
                correct_predictions += 1
            total_predictions += 1

        if (i + 1) % 100 == 0:
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            print(f"Processed {i + 1} samples. Current accuracy: {accuracy:.2%}")

    final_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\nFinal MLM Accuracy (Top-10): {final_accuracy:.2%}")
    print(f"Correct: {correct_predictions}/{total_predictions}")
    return final_accuracy


# ============================================================
# ✅ Example Usage
# ============================================================
if __name__ == "__main__":
    model_path = "UnixCoderCPP/unixcoder_cpp_mlm"  # path to folder, not .pt file
    predictor = MLMPredictor(model_path)

    # Example: Evaluate a small set of code snippets
    test_dataset = [
        {"code": "int add(int a, int b) { return a + b; }"},
        {"code": "void printHello() { std::cout << \"Hello, World!\" << std::endl; }"},
        {"code": "float multiply(float x, float y) { return x * y; }"}
    ]

    top1_correct = 0
    top5_correct = 0
    total_masks = 0

    for idx, sample in enumerate(test_dataset, 1):
        code = sample['code']
        tokens = predictor.tokenizer.tokenize(code)
        if len(tokens) < 5:
            continue

        # Randomly mask one token
        mask_idx = np.random.randint(1, min(len(tokens) - 1, 100))
        expected_token = tokens[mask_idx]
        tokens[mask_idx] = predictor.mask_token
        masked_code = predictor.tokenizer.convert_tokens_to_string(tokens)

        # Predict
        predictions = predictor.predict_masked_tokens(masked_code, top_k=5)

        # Top1 & Top5 accuracy
        if predictions:
            predicted_tokens = [t for t, _ in predictions[0]]
            top1_correct += int(predicted_tokens[0] == expected_token)
            top5_correct += int(expected_token in predicted_tokens)
            total_masks += 1

        # Compute perplexity
        ppl = predictor.get_token_perplexity(code)

        # Print results
        print("\n" + "=" * 60)
        print(f"Sample {idx}")
        print("=" * 60)
        print("Original code: ", code)
        print("Masked code:   ", masked_code)
        print("Expected token:", expected_token)
        if predictions:
            print("Top-5 predictions:")
            for i, (token, prob) in enumerate(predictions[0], 1):
                print(f"{i}. '{token}' (prob: {prob:.4f})")
        else:
            print("No predictions generated.")
        print(f"Perplexity: {ppl:.2f}")

    # Print overall accuracy
    print("\n" + "=" * 60)
    print("Overall Accuracy")
    print("=" * 60)
    top1_acc = top1_correct / total_masks if total_masks > 0 else 0
    top5_acc = top5_correct / total_masks if total_masks > 0 else 0
    print(f"Top-1 Accuracy: {top1_acc:.2%}")
    print(f"Top-5 Accuracy: {top5_acc:.2%}")
    print(f"Total masks evaluated: {total_masks}")