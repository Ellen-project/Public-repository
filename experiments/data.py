from __future__ import annotations

import _bootstrap  # noqa: F401

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


LEVEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LEVEL_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_project_path(path: str | Path) -> Path:
    """Resolve user paths without allowing access outside the project root.

    Relative paths are interpreted from experiments, so ../c4_token_cache.pt maps
    to the project root regardless of the caller's working directory.
    """
    path_obj = Path(path)
    if path_obj.is_absolute():
        resolved = path_obj.resolve()
    else:
        resolved = (LEVEL_DIR / path_obj).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"Refusing to access outside project root: {path}") from exc
    return resolved


def torch_load(path: str | Path, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_token_cache(path: str | Path) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    path_obj = resolve_project_path(path)
    if not path_obj.exists():
        raise FileNotFoundError(
            f"{path_obj} not found. 먼저 c4_build_cache.py를 실행해서 C4 token cache를 생성하라."
        )
    payload = torch_load(path_obj, map_location="cpu")
    input_ids = payload["input_ids"].long()
    labels = payload["labels"].long()
    metadata = dict(payload.get("metadata", {}))
    if input_ids.shape != labels.shape:
        raise ValueError(f"input_ids shape {tuple(input_ids.shape)} != labels shape {tuple(labels.shape)}")
    if input_ids.ndim != 2:
        raise ValueError("input_ids and labels must have shape [N, T]")
    return input_ids, labels, metadata


def load_gate_cache(path: str | Path) -> Tuple[torch.Tensor, dict]:
    path_obj = resolve_project_path(path)
    if not path_obj.exists():
        raise FileNotFoundError(
            f"{path_obj} not found. 먼저 c4_build_cache.py를 실행해서 C4 gate cache를 생성하라."
        )
    from Low_Rank_Path_SSM import load_gate_cache as root_load_gate_cache

    gates_np, metadata = root_load_gate_cache(str(path_obj))
    gates = torch.from_numpy(gates_np).float()
    if gates.ndim != 3:
        raise ValueError("gates must have shape [N, T, P]")
    return gates, dict(metadata)


def split_cache(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    gates: Optional[torch.Tensor] = None,
    val_ratio: float = 0.1,
    seed: int = 1,
):
    if input_ids.shape != labels.shape:
        raise ValueError("input_ids and labels shape mismatch")
    if gates is not None and gates.shape[:2] != input_ids.shape:
        raise ValueError("gates.shape[:2] must match input_ids shape")
    num_samples = int(input_ids.shape[0])
    if num_samples <= 0:
        raise ValueError("cache is empty")
    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(num_samples, generator=generator)
    val_count = max(1, int(round(num_samples * float(val_ratio)))) if num_samples > 1 else 1
    val_count = min(val_count, max(1, num_samples - 1)) if num_samples > 1 else 1
    val_idx = perm[:val_count]
    train_idx = perm[val_count:] if val_count < num_samples else perm

    def pack(indices):
        if gates is None:
            return input_ids[indices].long(), labels[indices].long(), None
        return input_ids[indices].long(), labels[indices].long(), gates[indices].float()

    return pack(train_idx), pack(val_idx)


class CacheDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor, labels: torch.Tensor, gates: Optional[torch.Tensor] = None):
        if input_ids.shape != labels.shape:
            raise ValueError("input_ids and labels shape mismatch")
        if gates is not None and gates.shape[:2] != input_ids.shape:
            raise ValueError("gates.shape[:2] must match input_ids shape")
        self.input_ids = input_ids.long()
        self.labels = labels.long()
        self.gates = None if gates is None else gates.float()

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int):
        item = {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}
        if self.gates is not None:
            item["gates"] = self.gates[idx]
        return item


def make_dataloader(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    gates: Optional[torch.Tensor] = None,
    batch_size: int = 8,
    shuffle: bool = True,
) -> DataLoader:
    dataset = CacheDataset(input_ids, labels, gates)
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle), drop_last=False)
