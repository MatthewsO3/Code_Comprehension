import argparse
import logging
import os
import json
from pathlib import Path

import torch
import numpy as np
from model import Model
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import AutoTokenizer, AutoModel, RobertaModel
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CodeSearchEvalDataset(Dataset):
    """Dataset for code search evaluation."""

    def __init__(self, tokenizer, args, file_path=None):
        """
        Args:
            tokenizer: Tokenizer for encoding
            args: Arguments containing code_length, nl_length
            file_path: Path to JSONL file with code and positive (docstring)
        """
        self.args = args
        self.tokenizer = tokenizer
        self.examples = []
        self.urls = []

        with open(file_path) as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                js = json.loads(line)
                self.examples.append(js)
                # Store URL if available, otherwise use index
                self.urls.append(js.get('url', f'example_{idx}'))

        logger.info(f"Loaded {len(self.examples)} examples from {file_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        """
        Returns:
            code_ids: Tokenized code
            code_mask: Attention mask for code
            url: URL or identifier for the example
        """
        example = self.examples[item]

        # Encode code
        encoded = self.tokenizer(
            example['code'],
            max_length=self.args.code_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        code_ids = encoded['input_ids'].squeeze(0)
        code_mask = encoded['attention_mask'].squeeze(0)

        return (
            code_ids,
            code_mask,
            self.urls[item]
        )


class DocstringDataset(Dataset):
    """Dataset for docstrings during evaluation."""

    def __init__(self, tokenizer, args, file_path=None):
        """
        Args:
            tokenizer: Tokenizer for encoding
            args: Arguments containing nl_length
            file_path: Path to JSONL file with positive (docstring)
        """
        self.args = args
        self.tokenizer = tokenizer
        self.examples = []
        self.urls = []

        with open(file_path) as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                js = json.loads(line)
                self.examples.append(js)
                self.urls.append(js.get('url', f'example_{idx}'))

        logger.info(f"Loaded {len(self.examples)} docstrings from {file_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        """
        Returns:
            docstring_ids: Tokenized docstring
            docstring_mask: Attention mask for docstring
            url: URL or identifier for the example
        """
        example = self.examples[item]

        # Handle both possible docstring keys
        doc_key = 'positive' if 'positive' in example else 'good_docstring'

        # Encode docstring
        encoded = self.tokenizer(
            example[doc_key],
            max_length=self.args.nl_length,  # <--- FIX
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        doc_ids = encoded['input_ids'].squeeze(0)
        doc_mask = encoded['attention_mask'].squeeze(0)

        return (
            doc_ids,
            doc_mask,
            self.urls[item]
        )


def collate_fn_code(batch):
    """Collate function for code batch."""
    code_ids = torch.stack([x[0] for x in batch])
    code_mask = torch.stack([x[1] for x in batch])
    urls = [x[2] for x in batch]
    return (code_ids, code_mask, urls)


def collate_fn_docstring(batch):
    """Collate function for docstring batch."""
    doc_ids = torch.stack([x[0] for x in batch])
    doc_mask = torch.stack([x[1] for x in batch])
    urls = [x[2] for x in batch]
    return (doc_ids, doc_mask, urls)


def compute_metrics(scores, code_urls, doc_urls, top_k=1000):
    """
    Compute evaluation metrics: MRR, NDCG, Recall@K

    Args:
        scores: Similarity scores [num_queries, num_docs]
        code_urls: URLs of code snippets (queries)
        doc_urls: URLs of docstrings (database)
        top_k: Top-k documents to evaluate

    Returns:
        Dictionary with metrics
    """
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]

    mrr_scores = []
    recall_1 = 0
    recall_5 = 0
    recall_10 = 0

    for idx, (code_url, sort_id) in enumerate(zip(code_urls, sort_ids)):
        rank = 0
        found = False

        # Find rank of correct match
        for i, doc_idx in enumerate(sort_id[:top_k]):
            if doc_urls[doc_idx] == code_url:
                rank = i + 1
                found = True
                break

        if found:
            mrr_scores.append(1.0 / rank)
            if rank <= 1:
                recall_1 += 1
            if rank <= 5:
                recall_5 += 1
            if rank <= 10:
                recall_10 += 1
        else:
            mrr_scores.append(0)

    num_queries = len(code_urls)

    metrics = {
        'mrr': float(np.mean(mrr_scores)),
        'recall@1': float(recall_1 / num_queries),
        'recall@5': float(recall_5 / num_queries),
        'recall@10': float(recall_10 / num_queries),
    }

    return metrics


def evaluate(args, model, tokenizer):
    """Evaluate the model on code search task."""

    # Load datasets
    logger.info("Loading evaluation datasets...")
    query_dataset = CodeSearchEvalDataset(tokenizer, args, args.eval_data_file)
    code_dataset = CodeSearchEvalDataset(tokenizer, args, args.eval_data_file)  # Same as query for now
    doc_dataset = DocstringDataset(tokenizer, args, args.eval_data_file)

    num_workers = 0 if str(args.device) == "mps" else 4

    query_dataloader = DataLoader(
        query_dataset,
        sampler=SequentialSampler(query_dataset),
        batch_size=args.eval_batch_size,
        collate_fn=collate_fn_code,
        num_workers=num_workers
    )

    doc_dataloader = DataLoader(
        doc_dataset,
        sampler=SequentialSampler(doc_dataset),
        batch_size=args.eval_batch_size,
        collate_fn=collate_fn_docstring,
        num_workers=num_workers
    )

    # Multi-GPU evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info(f"  Num queries = {len(query_dataset)}")
    logger.info(f"  Num docs = {len(doc_dataset)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")

    model.eval()

    # Get code embeddings
    logger.info("\nEncoding code snippets...")
    code_vecs = []
    code_urls_list = []

    with torch.no_grad():
        for batch in tqdm(query_dataloader, desc="Code embeddings"):
            code_ids = batch[0].to(args.device)
            code_mask = batch[1].to(args.device)
            urls = batch[2]

            code_vec = model(code_inputs=code_ids, attention_mask=code_mask)
            code_vecs.append(code_vec.cpu().numpy())
            code_urls_list.extend(urls)

    code_vecs = np.concatenate(code_vecs, axis=0)

    # Get docstring embeddings
    logger.info("Encoding docstrings...")
    doc_vecs = []
    doc_urls_list = []

    with torch.no_grad():
        for batch in tqdm(doc_dataloader, desc="Docstring embeddings"):
            doc_ids = batch[0].to(args.device)
            doc_mask = batch[1].to(args.device)
            urls = batch[2]

            doc_vec = model(nl_inputs=doc_ids, attention_mask=doc_mask)
            doc_vecs.append(doc_vec.cpu().numpy())
            doc_urls_list.extend(urls)

    doc_vecs = np.concatenate(doc_vecs, axis=0)

    # Compute similarity scores
    logger.info("Computing similarity scores...")
    scores = np.matmul(code_vecs, doc_vecs.T)

    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_metrics(scores, code_urls_list, doc_urls_list, top_k=1000)

    model.train()

    return metrics


def set_seed(seed=42):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Evaluate Code Search Model')

    # Config file parameter
    script_dir = Path(__file__).parent.parent.absolute()
    config_path = script_dir / 'config.json'
    parser.add_argument("--config", default=config_path, type=str,
                        help="Path to config JSON file")

    # Evaluation parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="Evaluation data file")
    parser.add_argument("--model_path", default=None, type=str,
                        help="Path to trained model checkpoint")
    parser.add_argument("--output_file", default=None, type=str,
                        help="Output file for evaluation results")
    parser.add_argument("--eval_batch_size", default=None, type=int,
                        help="Evaluation batch size")
    parser.add_argument("--code_length", default=None, type=int,
                        help="Code length")
    parser.add_argument("--nl_length", default=None, type=int,
                        help="Docstring length")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed")

    cli_args = parser.parse_args()

    # Load config from JSON
    config = load_config(cli_args.config)
    codesearch_config = config.get('codesearch', {})
    eval_config = codesearch_config.get('evaluation', {})

    # Create args object
    class Args:
        pass

    args = Args()

    # Set paths
    script_dir = Path(__file__).parent.parent.absolute()

    model_path = script_dir / (cli_args.model_path or codesearch_config.get('output_dir', '') + '/best_model')
    eval_data_path = script_dir / (cli_args.eval_data_file or eval_config.get('eval_data_file'))
    output_file = cli_args.output_file or eval_config.get('output_file', 'eval_results.json')

    args.model_path = model_path
    args.eval_data_file = eval_data_path
    args.output_file = output_file
    args.eval_batch_size = cli_args.eval_batch_size or codesearch_config.get('eval_batch_size', 32)
    args.code_length = cli_args.code_length or codesearch_config.get('code_length', 256)
    args.nl_length = cli_args.nl_length or codesearch_config.get('nl_length', 128)
    args.seed = cli_args.seed or codesearch_config.get('seed', 42)

    # Validate paths
    if not args.eval_data_file or not os.path.exists(args.eval_data_file):
        raise ValueError(f"Evaluation data file not found: {args.eval_data_file}")

    if not args.model_path or not os.path.exists(args.model_path):
        raise ValueError(f"Model path not found: {args.model_path}")

    # Set logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )

    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logger.info(f"Device: {device}, n_gpu: {args.n_gpu}")

    # Set seed
    set_seed(args.seed)

    # Load tokenizer and model
    logger.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    logger.info(f"Loading model from {args.model_path}")
    base_model = RobertaModel.from_pretrained(args.model_path)
    model = Model(base_model)
    model.to(args.device)

    logger.info(f"Model loaded successfully. Hidden size: {model.encoder.config.hidden_size}")

    # Print evaluation config
    print("\n" + "=" * 60)
    print("Evaluation Configuration")
    print("=" * 60)
    print(f"  Model path: {args.model_path}")
    print(f"  Eval data: {args.eval_data_file}")
    print(f"  Code length: {args.code_length}")
    print(f"  NL length: {args.nl_length}")
    print(f"  Batch size: {args.eval_batch_size}")
    print("=" * 60 + "\n")

    # Run evaluation
    logger.info("***** Running evaluation *****")
    metrics = evaluate(args, model, tokenizer)

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print("=" * 60 + "\n")

    # Save results
    results_dir = os.path.dirname(args.output_file)
    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with open(args.output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()