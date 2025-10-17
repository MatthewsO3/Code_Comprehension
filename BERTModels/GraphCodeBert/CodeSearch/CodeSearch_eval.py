# evaluate_codesearch.py (Corrected Version)

import torch
import json
from transformers import RobertaModel, RobertaTokenizer
from CodeSearch_train import CodeSearchCollator  # Reuse the collator for processing
import torch.nn.functional as F
from tqdm import tqdm

# --- NEW: Import and set up Tree-sitter for on-the-fly DFG extraction ---
from tree_sitter import Language, Parser
import tree_sitter_cpp as tscpp

CPP_LANGUAGE = Language(tscpp.language())
ts_parser = Parser(CPP_LANGUAGE)
from CodeSearch_train import extract_dataflow_graph  # Import the DFG function


def search(query, model, tokenizer, code_corpus, collator, device, top_k=5):
    """
    Searches a corpus of code for the best matches to a natural language query.
    """
    model.eval()

    # Process the query (this part was already correct)
    query_tokens = tokenizer.tokenize(query, add_prefix_space=True)
    query_processed = collator._process_item(query_tokens, [], collator.max_query_len)

    with torch.no_grad():
        query_vec = model(
            input_ids=query_processed[0].unsqueeze(0).to(device),
            attention_mask=query_processed[1].unsqueeze(0).to(device),
            position_ids=query_processed[2].unsqueeze(0).to(device)
        ).pooler_output

        # --- MODIFIED: Process the code corpus on-the-fly ---
        print("Encoding the code corpus...")
        code_vecs = []
        for code_sample in tqdm(code_corpus, desc="Encoding Corpus"):
            # 1. Get the raw code string, not tokens
            code_string = code_sample['code']

            # 2. Tokenize the code and extract DFG, just like in training
            code_bytes = code_string.encode('utf8')
            tree = ts_parser.parse(code_bytes)
            code_tokens = tokenizer.tokenize(code_string, add_prefix_space=True)
            dfg = extract_dataflow_graph(code_bytes, tree)

            # 3. Process the item with the collator
            code_processed = collator._process_item(code_tokens, dfg, collator.max_code_len)

            code_vec = model(
                input_ids=code_processed[0].unsqueeze(0).to(device),
                attention_mask=code_processed[1].unsqueeze(0).to(device),
                position_ids=code_processed[2].unsqueeze(0).to(device)
            ).pooler_output
            code_vecs.append(code_vec)

        code_vecs_tensor = torch.cat(code_vecs, dim=0)

        # Calculate cosine similarity
        similarities = F.cosine_similarity(query_vec, code_vecs_tensor)
        top_k_scores, top_k_indices = torch.topk(similarities, k=min(top_k, len(code_corpus)))

    print(f"\n--- Top {top_k} Results for Query: '{query}' ---")
    for i, idx in enumerate(top_k_indices):
        print(f"\nRank {i + 1} (Score: {top_k_scores[i]:.4f}):")
        print("-" * 20)
        # Use the original raw code for display
        print(code_corpus[idx]['code'])


if __name__ == '__main__':
    # Using an absolute path to config.json is more robust
    config_path = '/Users/czapmate/Desktop/szakdoga/GraphCodeBert_CPP/BERTModels/GraphCodeBert/config.json'
    config = json.load(open(config_path)).get('codesearch')

    # Use the absolute path for the data file as well
    data_file_path = '/Users/czapmate/Desktop/szakdoga/GraphCodeBert_CPP/BERTModels/GraphCodeBert/data/code_docstring_dataset.jsonl'

    model_path = f"{config['output_dir']}/checkpoint_epoch_{config['epochs']}"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaModel.from_pretrained(model_path).to(device)
    collator = CodeSearchCollator(tokenizer, config['max_code_len'], config['max_query_len'])

    print(f"Loading search corpus from {data_file_path}...")
    corpus = [json.loads(line) for line in open(data_file_path, 'r', encoding='utf-8')]

    queries = [
        "Sum all elements of an array"
    ]

    for q in queries:
        search(q, model, tokenizer, corpus, collator, device)