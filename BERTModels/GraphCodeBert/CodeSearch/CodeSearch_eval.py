# ============================================================================
# CodeSearch_eval.py - OPTIMIZED with pre-computed corpus encoding
# ============================================================================
from pathlib import Path

import torch
import json
from transformers import RobertaModel, RobertaTokenizer
from CodeSearch_train import CodeSearchCollator
import torch.nn.functional as F
from tqdm import tqdm

# --- Import and set up Tree-sitter for query processing ---
from tree_sitter import Language, Parser
import tree_sitter_cpp as tscpp

CPP_LANGUAGE = Language(tscpp.language())
ts_parser = Parser(CPP_LANGUAGE)
from CodeSearch_dataset import extract_dataflow_graph


def load_encoded_corpus(corpus_file):
    """Load pre-encoded corpus from JSONL file."""
    print(f"Loading pre-encoded corpus from {corpus_file}...")
    corpus = []
    embeddings = []

    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                corpus.append({
                    'code': sample['code'],
                    'docstring': sample['docstring'],
                    'dfg': sample.get('dfg', [])
                })
                embeddings.append(sample['embedding'])

    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    print(f"✓ Loaded {len(corpus)} pre-encoded samples")

    return corpus, embeddings_tensor


def search(query, model, tokenizer, corpus, embeddings, collator, device, top_k=5):
    """
    Searches a corpus of code for the best matches to a natural language query.

    Args:
        query: Natural language query string
        model: RoBERTa model
        tokenizer: RoBERTa tokenizer
        corpus: List of code samples
        embeddings: Pre-computed tensor of corpus embeddings
        collator: CodeSearchCollator for processing
        device: torch device
        top_k: Number of results to return
    """
    model.eval()

    # Process the query
    query_processed = collator._process_item("", [], query, collator.max_query_len, is_code=False)

    with torch.no_grad():
        query_vec = model(
            input_ids=query_processed[0].unsqueeze(0).to(device),
            attention_mask=query_processed[1].unsqueeze(0).to(device),
            position_ids=query_processed[2].unsqueeze(0).to(device)
        ).pooler_output

        # Move embeddings to device
        embeddings_device = embeddings.to(device)

        # Calculate cosine similarity against pre-encoded corpus
        similarities = F.cosine_similarity(query_vec, embeddings_device)
        top_k_scores, top_k_indices = torch.topk(similarities, k=min(top_k, len(corpus)))

    print(f"\n--- Top {top_k} Results for Query: '{query}' ---")
    for i, idx in enumerate(top_k_indices):
        print(f"\nRank {i + 1} (Score: {top_k_scores[i]:.4f}):")
        print("-" * 60)
        print(corpus[idx]['code'])
        print("-" * 60)


def search_batch(queries, model, tokenizer, corpus, embeddings, collator, device, top_k=5):
    """
    Search multiple queries at once.

    Args:
        queries: List of query strings
        model: RoBERTa model
        tokenizer: RoBERTa tokenizer
        corpus: List of code samples
        embeddings: Pre-computed tensor of corpus embeddings
        collator: CodeSearchCollator for processing
        device: torch device
        top_k: Number of results per query
    """
    model.eval()

    # Process all queries
    query_vecs = []
    for query in queries:
        query_processed = collator._process_item("", [], query, collator.max_query_len, is_code=False)
        with torch.no_grad():
            query_vec = model(
                input_ids=query_processed[0].unsqueeze(0).to(device),
                attention_mask=query_processed[1].unsqueeze(0).to(device),
                position_ids=query_processed[2].unsqueeze(0).to(device)
            ).pooler_output
        query_vecs.append(query_vec)

    query_vecs_batch = torch.cat(query_vecs, dim=0)

    with torch.no_grad():
        embeddings_device = embeddings.to(device)
        similarities = F.cosine_similarity(query_vecs_batch.unsqueeze(1), embeddings_device.unsqueeze(0), dim=2)

        for q_idx, query in enumerate(queries):
            top_k_scores, top_k_indices = torch.topk(similarities[q_idx], k=min(top_k, len(corpus)))

            print(f"\n{'=' * 60}")
            print(f"Query {q_idx + 1}: '{query}'")
            print(f"{'=' * 60}")

            for rank, idx in enumerate(top_k_indices):
                print(f"\nRank {rank + 1} (Score: {top_k_scores[rank]:.4f}):")
                print("-" * 60)
                print(corpus[idx]['code'])
                print("-" * 60)


if __name__ == '__main__':
    script_dir = Path(__file__).parent.absolute()

    # Navigate up to repo root, then to config
    config_path = script_dir.parent.parent / 'GraphCodeBert/config.json'
    config = json.load(open(config_path)).get('codesearch')

    # Use pre-encoded corpus file
    encoded_corpus_file = config.get('encoded_corpus_file', './data/encoded_corpus.jsonl')
    model_path = f"{config['output_dir']}/checkpoint_epoch_{config['epochs']}"

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaModel.from_pretrained(model_path).to(device)
    collator = CodeSearchCollator(tokenizer, config['max_code_len'], config['max_query_len'])

    # Load pre-encoded corpus (instant!)
    corpus, embeddings = load_encoded_corpus(encoded_corpus_file)

    # Example queries
    queries = [
        "Program to check if N is a Centered Cubic Number | C ++ program to check if N is a centered cubic number ; Function to check if the number N is a centered cubic number ; Iterating from 1 ; Infinite loop ; Finding ith_term ; Checking if the number N is a Centered cube number ; If ith_term > N then N is not a Centered cube number ; Incrementing i ; Driver code ; Function call",
        "Check if number is prime",
        "Sort array ascending"
    ]

    # Search single query
    print("Single Query Search:")
    search(queries[0], model, tokenizer, corpus, embeddings, collator, device, top_k=3)

    # Search batch queries
    print("\n\nBatch Query Search:")
    search_batch(queries, model, tokenizer, corpus, embeddings, collator, device, top_k=3)