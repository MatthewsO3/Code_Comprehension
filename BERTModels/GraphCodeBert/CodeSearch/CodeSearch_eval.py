# evaluate_codesearch.py

import torch
import json
from transformers import RobertaModel, RobertaTokenizer
from train_codesearch import CodeSearchCollator  # Reuse the collator for processing
import torch.nn.functional as F


def search(query, model, tokenizer, code_corpus, collator, device, top_k=5):
    """
    Searches a corpus of code for the best matches to a natural language query.
    """
    model.eval()

    # Process the query
    query_tokens = tokenizer.tokenize(query, add_prefix_space=True)
    query_processed = collator._process_item(query_tokens, [], collator.max_query_len)

    with torch.no_grad():
        query_vec = model(
            input_ids=query_processed[0].unsqueeze(0).to(device),
            attention_mask=query_processed[1].unsqueeze(0).to(device),
            position_ids=query_processed[2].unsqueeze(0).to(device)
        ).pooler_output

        # Process the code corpus
        code_vecs = []
        for code_sample in code_corpus:
            code_tokens = code_sample['code_tokens']
            dfg = code_sample.get('dataflow_graph', [])
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
        top_k_scores, top_k_indices = torch.topk(similarities, k=top_k)

    print(f"\n--- Top {top_k} Results for Query: '{query}' ---")
    for i, idx in enumerate(top_k_indices):
        print(f"\nRank {i + 1} (Score: {top_k_scores[i]:.4f}):")
        print("-" * 20)
        print(code_corpus[idx]['code'])


if __name__ == '__main__':
    config = json.load(open('config.json')).get('codesearch')
    model_path = f"{config['output_dir']}/checkpoint_epoch_{config['epochs']}"  # Use final model

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaModel.from_pretrained(model_path).to(device)
    collator = CodeSearchCollator(tokenizer, config['max_code_len'], config['max_query_len'])

    # Load the processed dataset to use as the search corpus
    print(f"Loading search corpus from {config['processed_data_file']}...")
    corpus = [json.loads(line) for line in open(config['processed_data_file'], 'r', encoding='utf-8')]

    # Example queries
    queries = [
        "function to calculate factorial",
        "read file contents into a string",
        "simple http server"
    ]

    for q in queries:
        search(q, model, tokenizer, corpus, collator, device)