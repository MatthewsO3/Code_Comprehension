"""
Evaluate trained GraphCodeBERT model on MLM task with C++ code snippets
"""

import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForMaskedLM
from pathlib import Path
import random
from typing import List, Tuple, Dict

# Set seed for reproducibility
random.seed(42)
torch.manual_seed(42)


class MLMEvaluator:
    def __init__(self, model_path: str, device: str = None):
        """Initialize evaluator with trained model"""
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # Load model and tokenizer
        print(f"Loading model from {model_path}...")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path,add_prefix_space=False)
        self.model = RobertaForMaskedLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id

        print("Model loaded successfully!")

    def mask_tokens(self, code: str, mask_ratio: float = 0.15) -> Tuple[str, List[int], List[str]]:
        """
        Mask tokens in code
        Returns: (masked_code, masked_positions, original_tokens)
        """
        # Tokenize
        tokens = self.tokenizer.tokenize(code,add_prefix_space=False )

        if len(tokens) == 0:
            return code, [], []

        # Select positions to mask (avoid special tokens)
        num_to_mask = max(1, int(len(tokens) * mask_ratio))
        maskable_positions = list(range(len(tokens)))

        # Randomly select positions
        mask_positions = sorted(random.sample(maskable_positions, min(num_to_mask, len(maskable_positions))))

        # Store original tokens
        original_tokens = [tokens[i] for i in mask_positions]

        # Create masked version
        masked_tokens = tokens.copy()
        for pos in mask_positions:
            masked_tokens[pos] = self.mask_token

        # Convert back to string
        masked_code = self.tokenizer.convert_tokens_to_string(masked_tokens)

        return masked_code, mask_positions, original_tokens

    def predict_masked_tokens(self, masked_code: str, mask_positions: List[int], top_k: int = 5) -> List[
        List[Tuple[str, float]]]:
        """
        Predict masked tokens
        Returns: List of [(token, probability)] for each masked position
        """
        # Tokenize with special tokens
        inputs = self.tokenizer(masked_code, return_tensors='pt', max_length=512, truncation=True,add_prefix_space=False )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get predictions for each masked position
        predictions = []

        # Find mask token positions in input_ids
        mask_token_indices = (inputs['input_ids'][0] == self.mask_token_id).nonzero(as_tuple=True)[0]

        for mask_idx in mask_token_indices:
            # Get logits for this position
            token_logits = logits[0, mask_idx]

            # Get top-k predictions
            probs = torch.softmax(token_logits, dim=0)
            top_k_probs, top_k_indices = torch.topk(probs, top_k)

            # Convert to tokens
            top_predictions = []
            for prob, idx in zip(top_k_probs, top_k_indices):
                token = self.tokenizer.decode([idx.item()]).strip()
                top_predictions.append((token, prob.item()))

            predictions.append(top_predictions)

        return predictions

    def evaluate_snippet(self, code: str, mask_ratio: float = 0.15, top_k: int = 5) -> Dict:
        """Evaluate model on a single code snippet"""
        print("\n" + "=" * 80)
        print("ORIGINAL CODE:")
        print("-" * 80)
        print(code)

        # Mask tokens
        masked_code, mask_positions, original_tokens = self.mask_tokens(code, mask_ratio)

        if len(mask_positions) == 0:
            print("No tokens to mask!")
            return None

        print("\n" + "=" * 80)
        print("MASKED CODE:")
        print("-" * 80)
        print(masked_code)

        # Predict
        predictions = self.predict_masked_tokens(masked_code, mask_positions, top_k)

        # Display results
        print("\n" + "=" * 80)
        print("PREDICTIONS:")
        print("-" * 80)

        top1_correct = 0
        top5_correct = 0
        log_probs = []

        for i, (orig_token, preds) in enumerate(zip(original_tokens, predictions)):
            print(f"\nPosition {i + 1}:")
            print(f"  Expected: '{orig_token}'")
            print(f"  Top {top_k} predictions:")

            for rank, (token, prob) in enumerate(preds, 1):
                marker = "✓" if token == orig_token else " "
                print(f"    {rank}. {marker} '{token}' (prob: {prob:.4f})")

            # Check accuracy
            top1_pred = preds[0][0]
            top5_preds = [p[0] for p in preds]

            if top1_pred == orig_token:
                top1_correct += 1
            if orig_token in top5_preds:
                top5_correct += 1

            # Get log probability of correct token
            correct_prob = next((p for t, p in preds if t == orig_token), 1e-10)
            log_probs.append(np.log(correct_prob))

        # Calculate metrics
        num_masks = len(original_tokens)
        top1_acc = top1_correct / num_masks
        top5_acc = top5_correct / num_masks
        perplexity = np.exp(-np.mean(log_probs))

        print("\n" + "=" * 80)
        print("METRICS:")
        print("-" * 80)
        print(f"  Masked tokens: {num_masks}")
        print(f"  Top-1 Accuracy: {top1_acc:.2%} ({top1_correct}/{num_masks})")
        print(f"  Top-5 Accuracy: {top5_acc:.2%} ({top5_correct}/{num_masks})")
        print(f"  Perplexity: {perplexity:.4f}")
        print("=" * 80)

        return {
            'top1_accuracy': top1_acc,
            'top5_accuracy': top5_acc,
            'perplexity': perplexity,
            'num_masked': num_masks
        }


# C++ code snippets for testing
CPP_SNIPPETS = [
    """
int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}
""",
    """
class Rectangle {
private:
    int width;
    int height;
public:
    Rectangle(int w, int h) : width(w), height(h) {}
    int area() { return width * height; }
};
""",
    """
template<typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}
""",
    """
std::vector<int> mergeArrays(const std::vector<int>& arr1, const std::vector<int>& arr2) {
    std::vector<int> result;
    result.reserve(arr1.size() + arr2.size());
    result.insert(result.end(), arr1.begin(), arr1.end());
    result.insert(result.end(), arr2.begin(), arr2.end());
    return result;
}
""",
    """
for (int i = 0; i < n; i++) {
    sum += arr[i];
}
""",
]


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate GraphCodeBERT MLM model')
    parser.add_argument('--model_path', type=str, default='./graphcodebert-cpp-mlm/best_model',
                        help='Path to trained model')
    parser.add_argument('--mask_ratio', type=float, default=0.15,
                        help='Ratio of tokens to mask')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top predictions to show')

    args = parser.parse_args()

    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train.py")
        return

    # Initialize evaluator
    evaluator = MLMEvaluator(str(model_path))

    # Evaluate on all snippets
    all_results = []

    for i, snippet in enumerate(CPP_SNIPPETS, 1):
        print(f"\n\n{'#' * 80}")
        print(f"# SNIPPET {i}/{len(CPP_SNIPPETS)}")
        print(f"{'#' * 80}")

        result = evaluator.evaluate_snippet(snippet, args.mask_ratio, args.top_k)
        if result:
            all_results.append(result)

    # Overall statistics
    if all_results:
        print("\n\n" + "#" * 80)
        print("# OVERALL STATISTICS")
        print("#" * 80)

        avg_top1 = np.mean([r['top1_accuracy'] for r in all_results])
        avg_top5 = np.mean([r['top5_accuracy'] for r in all_results])
        avg_perplexity = np.mean([r['perplexity'] for r in all_results])
        total_masked = sum([r['num_masked'] for r in all_results])

        print(f"  Evaluated {len(all_results)} snippets")
        print(f"  Total masked tokens: {total_masked}")
        print(f"  Average Top-1 Accuracy: {avg_top1:.2%}")
        print(f"  Average Top-5 Accuracy: {avg_top5:.2%}")
        print(f"  Average Perplexity: {avg_perplexity:.4f}")
        print("#" * 80)


if __name__ == "__main__":
    main()