import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from model import Model
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import AutoTokenizer, RobertaModel
from tqdm import tqdm


# Helper function to load config
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


# Helper class to load code snippets from the eval file
class CodeDataset(Dataset):
    def __init__(self, tokenizer, args, file_path):
        self.args = args
        self.tokenizer = tokenizer
        self.code_snippets = []
        self.docstrings = []

        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                js = json.loads(line)

                # Store the raw text
                self.code_snippets.append(js['code'])
                # Handle both possible docstring keys
                doc_key = 'positive' if 'positive' in js else 'good_docstring'
                self.docstrings.append(js[doc_key])

    def __len__(self):
        return len(self.code_snippets)

    def __getitem__(self, item):
        # Encode code
        code_tokens = self.tokenizer.tokenize(self.code_snippets[item])[:self.args.code_length - 2]
        code_tokens = [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token]
        code_ids = self.tokenizer.convert_tokens_to_ids(code_tokens)
        code_mask = [1] * len(code_ids)

        # Pad code
        padding_length = self.args.code_length - len(code_ids)
        code_ids += [self.tokenizer.pad_token_id] * padding_length
        code_mask += [0] * padding_length

        return (
            torch.tensor(code_ids),
            torch.tensor(code_mask)
        )


# Collate function for the dataloader
def collate_fn_code(batch):
    code_ids = torch.stack([x[0] for x in batch])
    code_mask = torch.stack([x[1] for x in batch])
    return (code_ids, code_mask)


# Function to encode a single text query
def encode_query(query_text, model, tokenizer, args):
    model.eval()

    # Tokenize
    nl_tokens = tokenizer.tokenize(query_text)[:args.nl_length - 2]
    nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    nl_mask = [1] * len(nl_ids)

    # Pad
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length
    nl_mask += [0] * padding_length

    # Convert to tensor
    nl_ids = torch.tensor(nl_ids).unsqueeze(0).to(args.device)  # Add batch dim
    nl_mask = torch.tensor(nl_mask).unsqueeze(0).to(args.device)  # Add batch dim

    with torch.no_grad():
        # Get embedding
        query_vec = model(nl_inputs=nl_ids, attention_mask=nl_mask)

    return query_vec.cpu().numpy()


# Function to pre-compute all code embeddings
def get_code_embeddings(model, tokenizer, args):
    model.eval()

    dataset = CodeDataset(tokenizer, args, args.eval_data_file)
    dataloader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=args.eval_batch_size,
        collate_fn=collate_fn_code,
        num_workers=0  # Safer for this script
    )

    code_vecs = []
    print("Pre-computing embeddings for all code snippets...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding Code"):
            code_ids = batch[0].to(args.device)
            code_mask = batch[1].to(args.device)

            code_vec = model(code_inputs=code_ids, attention_mask=code_mask)
            code_vecs.append(code_vec.cpu().numpy())

    code_vecs = np.concatenate(code_vecs, axis=0)

    return code_vecs, dataset.code_snippets, dataset.docstrings


def main():
    parser = argparse.ArgumentParser(description='Manual Code Search')

    # Load config for defaults
    script_dir = Path(__file__).parent.parent.absolute()
    config_path = script_dir / 'config.json'

    try:
        config = load_config(config_path)
        codesearch_config = config.get('codesearch', {})
        eval_config = codesearch_config.get('evaluation', {})
    except FileNotFoundError:
        print(f"Warning: config.json not found at {config_path}. Using hardcoded defaults.")
        codesearch_config = {}
        eval_config = {}

    # --- Set sensible defaults from config or hardcode them ---
    default_model_path = script_dir / (
                codesearch_config.get('output_dir', 'CodeSearch/graphcodebert-cpp-codesearch') + '/best_model')
    default_eval_file = script_dir / eval_config.get('eval_data_file', 'data/eval.jsonl')
    default_code_len = codesearch_config.get('code_length', 256)
    default_nl_len = codesearch_config.get('nl_length', 128)
    default_batch_size = codesearch_config.get('eval_batch_size', 32)

    parser.add_argument("--model_path", type=str, default=default_model_path,
                        help="Path to trained model checkpoint")
    parser.add_argument("--eval_data_file", type=str, default=default_eval_file,
                        help="Evaluation data file (JSONL)")
    parser.add_argument("--code_length", type=int, default=default_code_len,
                        help="Max code length")
    parser.add_argument("--nl_length", type=int, default=default_nl_len,
                        help="Max query length")
    parser.add_argument("--eval_batch_size", type=int, default=default_batch_size,
                        help="Batch size for pre-computing embeddings")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top results to show")

    args = parser.parse_args()

    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args.device = device
    print(f"Using device: {device}")

    # Load tokenizer and model
    if not os.path.exists(args.model_path):
        print(f"Error: Model path not found: {args.model_path}")
        return

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    base_model = RobertaModel.from_pretrained(args.model_path)
    model = Model(base_model)
    model.to(args.device)

    # Pre-compute all code embeddings
    if not os.path.exists(args.eval_data_file):
        print(f"Error: Eval data file not found: {args.eval_data_file}")
        return

    code_embeddings, code_snippets, docstrings = get_code_embeddings(model, tokenizer, args)

    print(f"\nModel and {len(code_snippets)} code snippets loaded. Ready for search.")
    print("=" * 60)

    # --- Interactive Search Loop ---
    while True:
        try:
            query = input("\nEnter your search query (or 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            if not query:
                continue

            # 1. Encode the query
            query_embedding = encode_query(query, model, tokenizer, args)

            # 2. Compute scores (dot product)
            # scores shape: (1, num_code_snippets)
            scores = np.dot(query_embedding, code_embeddings.T)

            # 3. Get top K results
            # We sort the scores in descending order and get the indices
            # [0] is to get the results for our single query
            top_k_indices = np.argsort(-scores[0])[:args.top_k]

            print("\n--- Top {args.top_k} Results ---")

            # 4. Print results
            for i, idx in enumerate(top_k_indices):
                print(f"\nRank {i + 1} (Score: {scores[0, idx]:.4f})")
                print("-" * 20)
                print(f"Original Docstring:\n{docstrings[idx]}\n")
                print(f"Found Code:\n{code_snippets[idx]}\n")
                print("-" * 60)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()