import argparse
import logging
import os
import random
from pathlib import Path

import torch
import json
import numpy as np
from model import Model
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
from tqdm import tqdm
from torch.optim import AdamW

logger = logging.getLogger(__name__)


class Args:
    """Configuration arguments class that can be pickled for multiprocessing."""
    pass


class CodeSearchDataset(Dataset):
    """Dataset for code search with triplet loss using hard negatives."""

    def __init__(self, tokenizer, args, file_path=None):
        """
        Args:
            tokenizer: Tokenizer for encoding
            args: Arguments containing code_length, nl_length
            file_path: Path to JSONL file with code, good_docstring, bad1_docstring, bad2_docstring
        """
        self.args = args
        self.tokenizer = tokenizer
        self.examples = []

        with open(file_path) as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                js = json.loads(line)
                self.examples.append(js)

        logger.info(f"Loaded {len(self.examples)} examples from {file_path}")

        # Log sample
        if len(self.examples) > 0:
            logger.info("*** Sample Example ***")
            logger.info(f"Code: {self.examples[0]['code'][:100]}...")
            logger.info(f"Good: {self.examples[0]['good_docstring'][:100]}...")
            logger.info(f"Bad1: {self.examples[0]['bad1_docstring'][:100]}...")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        """
        Returns:
            code_ids: Tokenized code
            code_mask: Attention mask for code
            good_ids: Tokenized good docstring
            good_mask: Attention mask for good docstring
            bad1_ids: Tokenized bad docstring 1
            bad1_mask: Attention mask for bad docstring 1
            bad2_ids: Tokenized bad docstring 2
            bad2_mask: Attention mask for bad docstring 2
        """
        example = self.examples[item]

        # Helper function to encode text safely
        def encode(text, max_len):
            encoded = self.tokenizer(
                text,
                max_length=max_len,       # Truncate to max_len
                padding='max_length',     # Pad to max_len
                truncation=True,
                return_tensors='pt'       # Return PyTorch tensors
            )
            # Squeeze(0) removes the batch dimension (tokenizer adds it)
            return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)

        # Encode all four items
        code_ids, code_mask = encode(example['code'], self.args.code_length)
        good_ids, good_mask = encode(example['good_docstring'], self.args.nl_length)
        bad1_ids, bad1_mask = encode(example['bad1_docstring'], self.args.nl_length)
        bad2_ids, bad2_mask = encode(example['bad2_docstring'], self.args.nl_length)

        return (
            code_ids,
            code_mask,
            good_ids,
            good_mask,
            bad1_ids,
            bad1_mask,
            bad2_ids,
            bad2_mask,
        )

def collate_fn(batch):
    """Custom collate function for batching."""
    code_ids = torch.stack([x[0] for x in batch])
    code_mask = torch.stack([x[1] for x in batch])
    good_ids = torch.stack([x[2] for x in batch])
    good_mask = torch.stack([x[3] for x in batch])
    bad1_ids = torch.stack([x[4] for x in batch])
    bad1_mask = torch.stack([x[5] for x in batch])
    bad2_ids = torch.stack([x[6] for x in batch])
    bad2_mask = torch.stack([x[7] for x in batch])

    return (code_ids, code_mask, good_ids, good_mask, bad1_ids, bad1_mask, bad2_ids, bad2_mask)


def train_epoch(model, train_dataloader, optimizer, scheduler, device, args):
    """Train for one epoch with detailed progress reporting."""
    model.train()

    total_loss = 0.0
    total_ce_loss = 0.0
    total_neg_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(train_dataloader, desc="Training", unit="batch")

    for step, batch in enumerate(progress_bar):
        # Get inputs
        code_ids = batch[0].to(args.device)
        code_mask = batch[1].to(args.device)
        good_ids = batch[2].to(args.device)
        good_mask = batch[3].to(args.device)
        bad1_ids = batch[4].to(args.device)
        bad1_mask = batch[5].to(args.device)
        bad2_ids = batch[6].to(args.device)
        bad2_mask = batch[7].to(args.device)

        optimizer.zero_grad()

        # Get code and docstring vectors
        code_vec = model(code_inputs=code_ids, attention_mask=code_mask)
        good_vec = model(nl_inputs=good_ids, attention_mask=good_mask)
        bad1_vec = model(nl_inputs=bad1_ids, attention_mask=bad1_mask)
        bad2_vec = model(nl_inputs=bad2_ids, attention_mask=bad2_mask)

        # Calculate scores: [batch_size, batch_size]
        scores = torch.einsum("ab,cb->ac", good_vec, code_vec)

        # Loss: cross-entropy where target is identity (diagonal)
        loss_fct = CrossEntropyLoss()
        ce_loss = loss_fct(scores, torch.arange(code_ids.size(0), device=scores.device))

        # Negative scores penalty
        bad_scores_1 = torch.einsum("ab,cb->ac", bad1_vec, code_vec)
        bad_scores_2 = torch.einsum("ab,cb->ac", bad2_vec, code_vec)

        # Margin-based loss for negatives
        neg_loss = 0
        margin = args.margin
        batch_size = code_ids.size(0)

        for i in range(batch_size):
            pos_score = scores[i, i]
            for j in range(batch_size):
                neg_loss += torch.clamp(margin + bad_scores_1[i, j] - pos_score, min=0)
                neg_loss += torch.clamp(margin + bad_scores_2[i, j] - pos_score, min=0)

        neg_loss = neg_loss / (batch_size ** 2)
        total_loss_batch = ce_loss + args.neg_weight * neg_loss

        # Backward
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        # Accumulate losses
        total_loss += total_loss_batch.item()
        total_ce_loss += ce_loss.item()
        total_neg_loss += neg_loss.item()
        num_batches += 1

        # Update progress bar with metrics
        avg_loss = total_loss / num_batches
        avg_ce = total_ce_loss / num_batches
        avg_neg = total_neg_loss / num_batches

        progress_bar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'CE': f'{avg_ce:.4f}',
            'Neg': f'{avg_neg:.4f}'
        })

    return total_loss / num_batches, total_ce_loss / num_batches, total_neg_loss / num_batches


def set_seed(seed=42):
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
    parser = argparse.ArgumentParser()

    # Config file parameter
    script_dir = Path(__file__).parent.parent.absolute()
    config_path = script_dir / 'config.json'
    parser.add_argument("--config", default=config_path, type=str,
                        help="Path to config JSON file")

    # Allow overriding config values via command line
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="Override config: training data file")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="Override config: output directory")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="Override config: model path")
    parser.add_argument("--tokenizer_name", default=None, type=str,
                        help="Override config: tokenizer name")
    parser.add_argument("--config_name", default=None, type=str,
                        help="Override config: config name")
    parser.add_argument("--nl_length", default=None, type=int,
                        help="Override config: docstring length")
    parser.add_argument("--code_length", default=None, type=int,
                        help="Override config: code length")
    parser.add_argument("--train_batch_size", default=None, type=int,
                        help="Override config: training batch size")
    parser.add_argument("--learning_rate", default=None, type=float,
                        help="Override config: learning rate")
    parser.add_argument("--max_grad_norm", default=None, type=float,
                        help="Override config: max gradient norm")
    parser.add_argument("--num_train_epochs", default=None, type=int,
                        help="Override config: number of epochs")
    parser.add_argument("--margin", default=None, type=float,
                        help="Override config: margin for negative loss")
    parser.add_argument("--neg_weight", default=None, type=float,
                        help="Override config: weight for negative loss")
    parser.add_argument('--seed', type=int, default=None,
                        help="Override config: random seed")

    cli_args = parser.parse_args()

    # Load config from JSON
    config = load_config(cli_args.config)
    codesearch_config = config.get('codesearch', {})

    # Create args object from config with CLI overrides
    args = Args()

    # Set all codesearch config values
    script_dir = Path(__file__).parent.parent.absolute()
    model_path = script_dir / (cli_args.model_name_or_path or codesearch_config.get('model_name_or_path'))
    output_path = script_dir / (cli_args.output_dir or codesearch_config.get('output_dir'))
    train_data_path = script_dir / (cli_args.train_data_file or codesearch_config.get('train_data_file'))

    args.train_data_file = train_data_path
    args.output_dir = output_path
    args.model_name_or_path = model_path
    args.tokenizer_name = model_path
    args.config_name = model_path
    args.nl_length = cli_args.nl_length or codesearch_config.get('nl_length', 128)
    args.code_length = cli_args.code_length or codesearch_config.get('code_length', 256)
    args.train_batch_size = cli_args.train_batch_size or codesearch_config.get('train_batch_size', 8)
    args.learning_rate = cli_args.learning_rate or codesearch_config.get('learning_rate', 5e-5)
    args.max_grad_norm = cli_args.max_grad_norm or codesearch_config.get('max_grad_norm', 1.0)
    args.num_train_epochs = cli_args.num_train_epochs or codesearch_config.get('num_train_epochs', 3)
    args.margin = cli_args.margin or codesearch_config.get('margin', 0.5)
    args.neg_weight = cli_args.neg_weight or codesearch_config.get('neg_weight', 0.5)
    args.seed = cli_args.seed or codesearch_config.get('seed', 42)

    # Validate required parameters
    if not args.train_data_file:
        raise ValueError("train_data_file must be specified in config or via CLI")
    if not args.output_dir:
        raise ValueError("output_dir must be specified in config or via CLI")
    if not args.model_name_or_path:
        raise ValueError("model_name_or_path must be specified in config or via CLI")

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

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load tokenizer and model
    logger.info(f"Loading tokenizer from {args.tokenizer_name or args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name or args.model_name_or_path)

    logger.info(f"Loading model from {args.model_name_or_path}")
    base_model = AutoModel.from_pretrained(args.model_name_or_path)
    model = Model(base_model)

    logger.info(f"Model loaded successfully. Hidden size: {model.encoder.config.hidden_size}")

    # Print training config
    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 60 + "\n")

    model.to(args.device)

    # Get training dataset
    train_dataset = CodeSearchDataset(tokenizer, args, args.train_data_file)
    train_sampler = RandomSampler(train_dataset)

    # Use num_workers=0 on macOS with MPS, otherwise use 4
    num_workers = 0 if str(args.device) == "mps" else 4

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    # Get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * args.num_train_epochs
    )

    # Multi-GPU training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total train batch size = {args.train_batch_size}")
    logger.info(f"  Total optimization steps = {len(train_dataloader) * args.num_train_epochs}")

    best_loss = float('inf')

    for epoch in range(args.num_train_epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{args.num_train_epochs}")
        print(f"{'=' * 60}")

        train_loss, train_ce, train_neg = train_epoch(model, train_dataloader, optimizer, scheduler, device, args)

        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Total Loss: {train_loss:.4f}")
        print(f"  CE Loss:    {train_ce:.4f}")
        print(f"  Neg Loss:   {train_neg:.4f}")

        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint_path = Path(args.output_dir) / "best_model"
            print(f"\nNew best model! Saving to {checkpoint_path}")

            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)

            model_to_save = model.module if hasattr(model, 'module') else model

            # Save model state dict
            model_to_save.encoder.save_pretrained(checkpoint_path)

            # Save the tokenizer
            tokenizer.save_pretrained(checkpoint_path)

            logger.info(f"Saved best model with loss: {best_loss:.4f}")


    print(f"\n{'=' * 60}")
    print(f"Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to: {checkpoint_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()