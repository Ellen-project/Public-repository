from __future__ import annotations

import argparse
from typing import Dict, Iterator, Optional

import torch
from tqdm import tqdm


def build_tokenizer(tokenizer_name: str = "gpt2"):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_c4_stream(
    subset: str = "en",
    split: str = "train",
    streaming: bool = True,
    shuffle_buffer: int = 10000,
    seed: int = 1,
):
    from datasets import load_dataset

    dataset = load_dataset("allenai/c4", subset, split=split, streaming=streaming)
    if streaming and split == "train" and shuffle_buffer and shuffle_buffer > 0:
        dataset = dataset.shuffle(buffer_size=int(shuffle_buffer), seed=int(seed))
    return dataset


def iter_c4_token_blocks(
    tokenizer,
    subset: str = "en",
    split: str = "train",
    block_size: int = 128,
    streaming: bool = True,
    shuffle_buffer: int = 10000,
    max_samples: Optional[int] = None,
    max_tokens: Optional[int] = None,
    seed: int = 1,
    add_eos: bool = True,
) -> Iterator[Dict[str, torch.Tensor]]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    dataset = load_c4_stream(
        subset=subset,
        split=split,
        streaming=streaming,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )

    buffer: list[int] = []
    raw_seen = 0
    emitted_tokens = 0
    eos_id = tokenizer.eos_token_id

    for sample in dataset:
        raw_seen += 1
        if max_samples is not None and raw_seen > int(max_samples):
            break

        text = sample.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue

        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        if add_eos and eos_id is not None:
            token_ids = list(token_ids) + [int(eos_id)]
        if not token_ids:
            continue
        buffer.extend(int(tok) for tok in token_ids)

        while len(buffer) >= block_size + 1:
            if max_tokens is not None and emitted_tokens + block_size > int(max_tokens):
                return
            block = buffer[: block_size + 1]
            del buffer[:block_size]
            emitted_tokens += block_size
            yield {
                "input_ids": torch.tensor(block[:-1], dtype=torch.long),
                "labels": torch.tensor(block[1:], dtype=torch.long),
            }


def make_c4_batch_iterator(
    tokenizer_name: str = "gpt2",
    subset: str = "en",
    split: str = "train",
    block_size: int = 128,
    batch_size: int = 8,
    streaming: bool = True,
    shuffle_buffer: int = 10000,
    max_samples: Optional[int] = None,
    max_tokens: Optional[int] = None,
    seed: int = 1,
) -> Iterator[Dict[str, torch.Tensor]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    tokenizer = build_tokenizer(tokenizer_name)
    blocks = iter_c4_token_blocks(
        tokenizer=tokenizer,
        subset=subset,
        split=split,
        block_size=block_size,
        streaming=streaming,
        shuffle_buffer=shuffle_buffer,
        max_samples=max_samples,
        max_tokens=max_tokens,
        seed=seed,
    )
    batch: list[Dict[str, torch.Tensor]] = []
    for block in blocks:
        batch.append(block)
        if len(batch) == batch_size:
            yield {
                "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
                "labels": torch.stack([item["labels"] for item in batch], dim=0),
            }
            batch.clear()


def parse_args():
    parser = argparse.ArgumentParser(description="C4 streaming token block utilities.")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--subset", type=str, default="en")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.smoke_test:
        raise SystemExit("Pass --smoke-test to run the C4 data smoke test.")

    iterator = make_c4_batch_iterator(
        tokenizer_name=args.tokenizer_name,
        subset=args.subset,
        split=args.split,
        block_size=args.block_size,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    for batch in tqdm(iterator, total=1, desc="c4_data_smoke"):
        shape = list(batch["input_ids"].shape)
        print(f"C4_DATA_SMOKE_PASS input_ids_shape={shape}")
        return
    raise RuntimeError("C4 data smoke test produced no batch")


if __name__ == "__main__":
    main()
