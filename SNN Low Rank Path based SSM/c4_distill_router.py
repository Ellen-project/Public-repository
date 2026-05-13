from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from Low_Rank_Path_SSM import LightweightRouter, load_gate_cache
from c4_lrp_lm import load_gate_feature_encoder, torch_load


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Distill C4 FullSNNPathRouter gates into a lightweight router.")
    parser.add_argument("--token-cache", type=str, default="c4_token_cache.pt")
    parser.add_argument("--gate-cache", type=str, default="c4_gate_cache.pt")
    parser.add_argument("--gate-feature-encoder", type=str, default="c4_gate_feature_encoder.pt")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-router", type=str, default="c4_lightweight_router.pt")
    parser.add_argument("--output-report", type=str, default="c4_distill_report.json")
    return parser.parse_args()


def load_token_cache(path: str) -> Tuple[torch.Tensor, dict]:
    payload = torch_load(path, map_location="cpu")
    return payload["input_ids"].long(), dict(payload.get("metadata", {}))


def load_gate_tensor(path: str) -> Tuple[torch.Tensor, dict]:
    gates_np, metadata = load_gate_cache(path)
    return torch.from_numpy(gates_np).float(), metadata


@torch.no_grad()
def evaluate_router(router, gate_feature_encoder, loader, device, threshold: float) -> dict:
    router.eval()
    total_loss = 0.0
    total_count = 0
    correct = 0
    total_bits = 0
    teacher_active = []
    student_active = []
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    for input_ids, teacher_gates in loader:
        input_ids = input_ids.to(device)
        teacher_gates = teacher_gates.to(device)
        features = gate_feature_encoder(input_ids)
        logits = router(features)
        loss = criterion(logits, teacher_gates)
        probs = torch.sigmoid(logits)
        pred = (probs > threshold).float()
        total_loss += float(loss.detach().cpu())
        total_count += int(teacher_gates.numel())
        correct += int((pred == teacher_gates).sum().item())
        total_bits += int(teacher_gates.numel())
        teacher_active.append(teacher_gates.sum(dim=-1).detach().cpu())
        student_active.append(pred.sum(dim=-1).detach().cpu())
    if not teacher_active:
        return {}
    teacher_active_t = torch.cat([item.reshape(-1) for item in teacher_active])
    student_active_t = torch.cat([item.reshape(-1) for item in student_active])
    return {
        "bce_loss": total_loss / max(1, total_count),
        "binary_accuracy": correct / max(1, total_bits),
        "active_path_mae": float((teacher_active_t - student_active_t).abs().float().mean().item()),
        "teacher_mean_active_paths": float(teacher_active_t.float().mean().item()),
        "student_mean_active_paths": float(student_active_t.float().mean().item()),
        "zero_gate_ratio_teacher": float((teacher_active_t <= 1e-6).float().mean().item()),
        "zero_gate_ratio_student": float((student_active_t <= 1e-6).float().mean().item()),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    input_ids, token_metadata = load_token_cache(args.token_cache)
    teacher_gates, gate_metadata = load_gate_tensor(args.gate_cache)
    if teacher_gates.shape[:2] != input_ids.shape:
        raise ValueError("gate cache shape does not match token cache")

    gate_feature_encoder, encoder_payload = load_gate_feature_encoder(args.gate_feature_encoder, map_location=device)
    gate_feature_encoder.eval()
    model_dim = int(encoder_payload["model_dim"])
    num_paths = int(teacher_gates.shape[-1])
    router = LightweightRouter(model_dim, num_paths, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(router.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(input_ids, teacher_gates), batch_size=args.batch_size, shuffle=True, drop_last=False)

    start = time.perf_counter()
    last_loss = None
    for _ in range(args.epochs):
        router.train()
        for batch_input_ids, batch_teacher in loader:
            batch_input_ids = batch_input_ids.to(device)
            batch_teacher = batch_teacher.to(device)
            with torch.no_grad():
                features = gate_feature_encoder(batch_input_ids)
            logits = router(features)
            loss = criterion(logits, batch_teacher)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())

    metrics = evaluate_router(router, gate_feature_encoder, loader, device, args.threshold)
    metrics["last_train_loss"] = last_loss
    metrics["elapsed_sec"] = max(time.perf_counter() - start, 1e-9)
    payload = {
        "model_dim": model_dim,
        "input_dim": model_dim,
        "num_paths": num_paths,
        "hidden_dim": int(args.hidden_dim),
        "threshold": float(args.threshold),
        "state_dict": router.state_dict(),
        "metrics": metrics,
        "token_metadata": token_metadata,
        "gate_metadata": gate_metadata,
    }
    Path(args.output_router).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_router)
    Path(args.output_report).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("C4_DISTILL_PASS")


if __name__ == "__main__":
    main()
