"""
UniXcoder Dataset Preparation for C++ Code
Streams the-stack dataset and prepares data for UniXcoder training
UniXcoder doesn't use DFG - it uses simpler tokenization with attention masks
"""

import json
from pathlib import Path
from datasets import load_dataset
from transformers import RobertaTokenizer
from tqdm import tqdm

# Load UniXcoder tokenizer - use unixcoder-base-nine which is trained on C++
tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base-nine")
print("✓ UniXcoder tokenizer loaded")


def preprocess_code(code: str, idx: int) -> dict:
    """
    Preprocess C++ code for UniXcoder.
    UniXcoder is simpler than GraphCodeBERT - no DFG needed.
    """
    try:
        # UniXcoder uses standard tokenization with add_prefix_space=True
        tokens = tokenizer.tokenize(code, add_prefix_space=True)

        # Filter by token length (leave room for special tokens)
        if len(tokens) < 10 or len(tokens) > 450:
            return None

        return {
            'idx': f'cpp::{idx}',
            'code': code,
            'code_tokens': tokens,
            'docstring': '',  # Keep for compatibility
            'docstring_tokens': []
        }
    except Exception:
        return None


def should_keep_code(code: str) -> bool:
    """Filter criteria for C++ code"""
    if len(code) < 100 or len(code) > 10000:
        return False
    lines = code.count('\n')
    if lines < 3 or lines > 500:
        return False
    # Basic C++ indicators
    if 'void ' not in code and 'int ' not in code and 'class ' not in code and 'std::' not in code:
        return False
    return True


def stream_and_process_dataset(output_file: str, max_samples: int = 10000):
    """Stream dataset and save in JSONL format for UniXcoder."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading dataset in streaming mode...")
    dataset = load_dataset("codeparrot/github-code-clean", "C++-all", split="train", streaming=True)

    processed_count = 0
    with open(output_path, 'w', encoding='utf-8') as f, tqdm(total=max_samples, desc="Processing C++ files") as pbar:
        for example in dataset:
            if processed_count >= max_samples:
                break

            code = example.get('code')
            if not code or not should_keep_code(code):
                continue

            processed = preprocess_code(code, processed_count)
            if processed:
                f.write(json.dumps(processed, ensure_ascii=False) + '\n')
                processed_count += 1
                pbar.update(1)

    print(f"\n{'=' * 50}\nProcessing complete!")
    print(f"Total samples processed and saved: {processed_count}")
    print(f"Data saved to: {output_path}\n{'=' * 50}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Prepare dataset for UniXcoder')
    parser.add_argument('--output_file', type=str, default='data/unixcoder_cpp.jsonl',
                        help='Output JSONL file for processed data')
    parser.add_argument('--max_samples', type=int, default=10000,
                        help='Maximum number of samples to process')
    args = parser.parse_args()
    stream_and_process_dataset(args.output_file, args.max_samples)