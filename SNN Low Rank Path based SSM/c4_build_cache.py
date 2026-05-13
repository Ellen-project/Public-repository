from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from Low_Rank_Path_SSM import save_gate_cache
from c4_data import build_tokenizer, iter_c4_token_blocks
from c4_lrp_lm import (
    create_gate_feature_encoder,
    gate_statistics,
    load_full_snn_router_for_dimensions,
    save_gate_feature_encoder,
)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def partial_path(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj.with_name(f"{path_obj.stem}.partial{path_obj.suffix}")


def parse_optional_int(value: Optional[str]) -> Optional[int]:
    if value is None or str(value).lower() == "none":
        return None
    return int(value)


def parse_args():
    parser = argparse.ArgumentParser(description="Build C4 token and SNN gate caches.")
    parser.add_argument("--subset", type=str, default="en")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--token-cache", type=str, default=None)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--max-raw-samples", type=parse_optional_int, default=None)
    parser.add_argument("--max-tokens", type=parse_optional_int, default=None)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-paths", type=int, default=8)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--router-preset", type=str, default="best_router_preset.npz")
    parser.add_argument("--output-token-cache", type=str, default="c4_token_cache.pt")
    parser.add_argument("--output-gate-cache", type=str, default="c4_gate_cache.pt")
    parser.add_argument("--output-gate-feature-encoder", type=str, default="c4_gate_feature_encoder.pt")
    parser.add_argument("--streaming", dest="streaming", action="store_true", default=True)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--use-gpu-router", action="store_true")
    parser.add_argument("--save-every", type=int, default=128)
    parser.add_argument("--strict-router-input-dim", dest="strict_router_input_dim", action="store_true", default=True)
    parser.add_argument("--no-strict-router-input-dim", dest="strict_router_input_dim", action="store_false")
    parser.add_argument("--allow-adapt-preset", action="store_true")
    parser.add_argument("--target-min-active", type=float, default=1.0)
    parser.add_argument("--target-max-active", type=float, default=2.0)
    parser.add_argument("--max-zero-gate-ratio", type=float, default=0.35)
    parser.add_argument("--fail-if-bad-gate-stats", action="store_true")
    parser.add_argument("--auto-calibrate-router", action="store_true")
    parser.add_argument("--calibrated-router-output", type=str, default="best_router_preset_auto.npz")
    return parser.parse_args()


def save_token_cache(path: str | Path, input_ids: torch.Tensor, labels: torch.Tensor, metadata: dict) -> Path:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"input_ids": input_ids.cpu(), "labels": labels.cpu(), "metadata": metadata}, path_obj)
    return path_obj


def load_token_cache(path: str | Path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    input_ids = payload["input_ids"].long()
    labels = payload["labels"].long()
    metadata = dict(payload.get("metadata", {}))
    if input_ids.shape != labels.shape:
        raise ValueError("token cache input_ids and labels shape mismatch")
    return input_ids, labels, metadata


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    if args.token_cache:
        if args.output_token_cache == "c4_token_cache.pt":
            args.output_token_cache = args.token_cache
        input_ids, labels, token_metadata = load_token_cache(args.token_cache)
        if args.num_samples and input_ids.shape[0] > args.num_samples:
            input_ids = input_ids[: args.num_samples].contiguous()
            labels = labels[: args.num_samples].contiguous()
        args.block_size = int(input_ids.shape[1])
        args.tokenizer_name = token_metadata.get("tokenizer_name", args.tokenizer_name)
        vocab_size = int(token_metadata.get("vocab_size", int(input_ids.max().item()) + 1))
        token_metadata = dict(token_metadata)
        token_metadata["source_token_cache"] = args.token_cache
        token_metadata["num_samples"] = int(input_ids.shape[0])
        save_token_cache(args.output_token_cache, input_ids, labels, token_metadata)
    else:
        tokenizer = build_tokenizer(args.tokenizer_name)
        vocab_size = len(tokenizer)
        block_iter = iter_c4_token_blocks(
            tokenizer=tokenizer,
            subset=args.subset,
            split=args.train_split,
            block_size=args.block_size,
            streaming=args.streaming,
            shuffle_buffer=args.shuffle_buffer,
            max_samples=args.max_raw_samples,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )

        input_blocks = []
        label_blocks = []
        for block in tqdm(block_iter, total=args.num_samples, desc="c4_token_blocks"):
            input_blocks.append(block["input_ids"])
            label_blocks.append(block["labels"])
            if len(input_blocks) >= args.num_samples:
                break
        if len(input_blocks) < args.num_samples:
            raise RuntimeError(f"Only produced {len(input_blocks)} C4 token blocks; requested {args.num_samples}")

        input_ids = torch.stack(input_blocks, dim=0).long()
        labels = torch.stack(label_blocks, dim=0).long()
        token_metadata = {
            "dataset_name": "allenai/c4",
            "subset": args.subset,
            "split": args.train_split,
            "tokenizer_name": args.tokenizer_name,
            "vocab_size": int(vocab_size),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "block_size": int(args.block_size),
            "num_samples": int(input_ids.shape[0]),
            "max_raw_samples": args.max_raw_samples,
            "max_tokens": args.max_tokens,
            "streaming": bool(args.streaming),
            "shuffle_buffer": int(args.shuffle_buffer),
            "seed": int(args.seed),
        }
        save_token_cache(args.output_token_cache, input_ids, labels, token_metadata)

    gate_feature_emb = create_gate_feature_encoder(vocab_size, args.model_dim, seed=args.seed, device=device)
    encoder_metadata = {
        "dataset_name": "allenai/c4",
        "subset": args.subset,
        "split": args.train_split,
        "block_size": int(args.block_size),
        "num_samples": int(input_ids.shape[0]),
    }
    save_gate_feature_encoder(
        gate_feature_emb,
        args.output_gate_feature_encoder,
        tokenizer_name=args.tokenizer_name,
        seed=args.seed,
        metadata=encoder_metadata,
    )
    if args.auto_calibrate_router:
        from argparse import Namespace
        from fix_lrp_router_calibration import calibrate_router_from_token_cache

        calib_args = Namespace(
            token_cache=args.output_token_cache,
            gate_feature_encoder=args.output_gate_feature_encoder,
            model_dim=args.model_dim,
            num_paths=args.num_paths,
            target_min_active=args.target_min_active,
            target_max_active=args.target_max_active,
            window_ms=1.0,
            dt=0.025,
            current_clip=1e-4,
            scale_min=1e-4,
            scale_max=1e-1,
            scale_steps=9,
            n_basal=1,
            n_apical=1,
            n_tuft=1,
            seed=args.seed,
            output_preset=args.calibrated_router_output,
            output_report=str(Path(args.calibrated_router_output).with_suffix(".report.json")),
            device=str(device),
        )
        calibrate_router_from_token_cache(calib_args)
        args.router_preset = args.calibrated_router_output

    router, router_metadata = load_full_snn_router_for_dimensions(
        args.router_preset,
        input_dim=args.model_dim,
        num_paths=args.num_paths,
        seed=args.seed,
        strict_input_dim=args.strict_router_input_dim,
        allow_adapt_preset=args.allow_adapt_preset,
    )

    gates = torch.zeros(input_ids.shape[0], input_ids.shape[1], args.num_paths, dtype=torch.float32)
    start = time.perf_counter()
    for sample_idx in tqdm(range(input_ids.shape[0]), desc="c4_snn_gates"):
        router.reset_state()
        with torch.no_grad():
            features = gate_feature_emb(input_ids[sample_idx].to(device)).detach().cpu()
        for t in range(input_ids.shape[1]):
            gates[sample_idx, t] = torch.from_numpy(
                router.step(features[t].numpy(), use_ema=args.use_ema, use_gpu=args.use_gpu_router)
            )
        if args.save_every > 0 and (sample_idx + 1) % args.save_every == 0:
            partial_metadata = dict(token_metadata)
            partial_metadata.update(
                {
                    "partial_num_samples": int(sample_idx + 1),
                    "gate_feature_encoder_path": args.output_gate_feature_encoder,
                    **router_metadata,
                }
            )
            save_token_cache(partial_path(args.output_token_cache), input_ids[: sample_idx + 1], labels[: sample_idx + 1], partial_metadata)
            torch.save(
                {
                    "gates": gates[: sample_idx + 1].clone(),
                    "metadata": partial_metadata,
                },
                partial_path(args.output_gate_cache),
            )

    elapsed = max(time.perf_counter() - start, 1e-9)
    gate_stats = gate_statistics(gates)
    generated_samples = int(input_ids.shape[0])
    generated_tokens = int(input_ids.numel())
    metadata = {
        "dataset_name": "allenai/c4",
        "subset": args.subset,
        "split": args.train_split,
        "tokenizer_name": args.tokenizer_name,
        "block_size": int(args.block_size),
        "num_samples": generated_samples,
        "model_dim": int(args.model_dim),
        "num_paths": int(args.num_paths),
        "rank": int(args.rank),
        "state_dim": int(args.state_dim),
        "router_preset": args.router_preset,
        "router_preset_path": args.router_preset,
        "gate_feature_encoder_path": args.output_gate_feature_encoder,
        "use_ema": bool(args.use_ema),
        "use_gpu_router": bool(args.use_gpu_router),
        "seed": int(args.seed),
        "current_scale": router.current_scale,
        "current_clip": router.current_clip,
        "window_ms": router.window_ms,
        "dt": router.dt,
        "target_min_active": float(args.target_min_active),
        "target_max_active": float(args.target_max_active),
        "max_zero_gate_ratio": float(args.max_zero_gate_ratio),
        "router_input_dim": int(router.input_dim),
        "strict_input_dim": bool(args.strict_router_input_dim),
        "allow_adapt_preset": bool(args.allow_adapt_preset),
        **router_metadata,
        "generated_samples": generated_samples,
        "generated_tokens": generated_tokens,
        "elapsed_sec": elapsed,
        "token_blocks_per_sec": generated_samples / elapsed,
        "tokens_per_sec": generated_tokens / elapsed,
        "snn_steps_per_sec": generated_tokens / elapsed,
        **gate_stats,
    }
    bad_gate_stats = (
        metadata["mean_active_paths"] < args.target_min_active
        or metadata["mean_active_paths"] > args.target_max_active
        or metadata["zero_gate_ratio"] > args.max_zero_gate_ratio
    )
    save_gate_cache(gates.numpy(), args.output_gate_cache, metadata)
    if args.fail_if_bad_gate_stats and bad_gate_stats:
        raise RuntimeError(
            "Bad gate stats: "
            f"mean_active_paths={metadata['mean_active_paths']:.4g}, "
            f"zero_gate_ratio={metadata['zero_gate_ratio']:.4g}; "
            f"target=[{args.target_min_active}, {args.target_max_active}], "
            f"max_zero_gate_ratio={args.max_zero_gate_ratio}"
        )

    print(json.dumps({k: metadata[k] for k in (
        "generated_samples",
        "generated_tokens",
        "elapsed_sec",
        "token_blocks_per_sec",
        "tokens_per_sec",
        "snn_steps_per_sec",
        "mean_active_paths",
        "zero_gate_ratio",
        "all_on_gate_ratio",
    )}, indent=2))
    print("C4_CACHE_BUILD_PASS")


if __name__ == "__main__":
    main()
