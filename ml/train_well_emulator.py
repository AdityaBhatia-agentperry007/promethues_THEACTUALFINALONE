from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from backend.surrogate.well_emulator import ResidualFrameEmulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frame-by-frame emulator on The Well data.")
    parser.add_argument("--dataset", default=os.getenv("WELL_DATASET", "MHD_64"))
    parser.add_argument("--base-path", default=os.getenv("WELL_BASE_PATH", "hf://datasets/polymathic-ai/"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="valid")
    parser.add_argument("--frame-size", type=int, default=128)
    parser.add_argument("--max-train-batches", type=int, default=600)
    parser.add_argument("--max-val-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--model-width", type=int, default=64)
    parser.add_argument("--seed-bank-size", type=int, default=24)
    parser.add_argument("--output", default=os.getenv("WELL_EMULATOR_OUTPUT", "backend/models/well_mhd64_emulator.pt"))
    return parser.parse_args()


def load_well_dataset(base_path: str, dataset: str, split: str):
    try:
        from the_well.data import WellDataset
    except Exception as exc:
        raise SystemExit("Install The Well first: pip install the_well huggingface_hub") from exc
    return WellDataset(
        well_base_path=base_path,
        well_dataset_name=dataset,
        well_split_name=split,
        n_steps_input=1,
        n_steps_output=1,
        use_normalization=False,
    )


def iter_tensors(obj: Any):
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from iter_tensors(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from iter_tensors(value)


def choose_field_tensor(batch: Any) -> torch.Tensor:
    candidates = [tensor for tensor in iter_tensors(batch) if tensor.is_floating_point() and tensor.ndim >= 4]
    if not candidates:
        raise RuntimeError(f"Could not find a floating field tensor in batch keys={list(batch.keys()) if isinstance(batch, dict) else type(batch)}")
    return max(candidates, key=lambda tensor: tensor.numel())


def to_scalar_frames_from_tensor(tensor: torch.Tensor, frame_size: int) -> torch.Tensor:
    """Convert a Well field tensor to `(B, T, H, W)` scalar 2D frames."""
    x = tensor.float()
    while x.ndim > 6:
        x = x[0]
    if x.ndim >= 5:
        x = select_2d_slice_after_batch_time(x)
    elif x.ndim == 4:
        # Either (T,H,W,C), (B,T,H,W), or (T,C,H,W). Normalize to B,T,H,W.
        if x.shape[-1] <= 8:
            x = x[..., 0].unsqueeze(0)
        elif x.shape[1] <= 8:
            x = x[:, 0].unsqueeze(0)
        else:
            x = x.unsqueeze(0)
    else:
        raise RuntimeError(f"Unsupported tensor shape: {tuple(x.shape)}")

    if x.ndim != 4:
        raise RuntimeError(f"Expected B,T,H,W after conversion, got {tuple(x.shape)}")
    b, t, h, w = x.shape
    x = x.reshape(b * t, 1, h, w)
    x = F.interpolate(x, size=(frame_size, frame_size), mode="bilinear", align_corners=False)
    x = x.reshape(b, t, frame_size, frame_size)
    return x


def select_2d_slice_after_batch_time(x: torch.Tensor) -> torch.Tensor:
    """Keep B,T and reduce arbitrary field axes to a deterministic H,W scalar slice."""
    if x.ndim < 5:
        return x
    tail = list(x.shape[2:])
    large_axes = [axis for axis, size in enumerate(tail) if size > 8]
    if len(large_axes) >= 2:
        keep_tail_axes = set(large_axes[:2])
    else:
        keep_tail_axes = set(sorted(range(len(tail)), key=lambda axis: tail[axis], reverse=True)[:2])

    index: list[object] = [slice(None), slice(None)]
    for axis, size in enumerate(tail):
        if axis in keep_tail_axes:
            index.append(slice(None))
        elif size <= 8:
            index.append(0)
        else:
            index.append(size // 2)
    sliced = x[tuple(index)]
    if sliced.ndim != 4:
        raise RuntimeError(f"Could not reduce field tensor {tuple(x.shape)} to B,T,H,W; got {tuple(sliced.shape)}")
    return sliced


def make_pairs(batch: Any, frame_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict) and "input_fields" in batch and "output_fields" in batch:
        inputs = to_scalar_frames_from_tensor(batch["input_fields"], frame_size)
        outputs = to_scalar_frames_from_tensor(batch["output_fields"], frame_size)
        x = inputs[:, -1:].reshape(-1, 1, frame_size, frame_size)
        y = outputs[:, :1].reshape(-1, 1, frame_size, frame_size)
    else:
        frames = to_scalar_frames_from_tensor(choose_field_tensor(batch), frame_size)
        if frames.shape[1] < 2:
            raise RuntimeError(f"Need at least two timesteps, got {tuple(frames.shape)}")
        x = frames[:, :-1].reshape(-1, 1, frame_size, frame_size)
        y = frames[:, 1:].reshape(-1, 1, frame_size, frame_size)
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return (x - mean) / std, (y - mean) / std


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    frame_size: int,
    max_batches: int,
) -> float:
    model.train()
    losses: list[float] = []
    for step, batch in enumerate(loader):
        if step >= max_batches:
            break
        x, y = make_pairs(batch, frame_size)
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step % 20 == 0:
            print(f"batch={step} train_mse={losses[-1]:.6f}", flush=True)
    return float(sum(losses) / max(1, len(losses)))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    frame_size: int,
    max_batches: int,
    seed_bank_size: int,
) -> tuple[float, torch.Tensor]:
    model.eval()
    losses: list[float] = []
    seeds: list[torch.Tensor] = []
    for step, batch in enumerate(loader):
        if step >= max_batches:
            break
        x, y = make_pairs(batch, frame_size)
        if len(seeds) < seed_bank_size:
            seeds.append(x[: max(0, seed_bank_size - len(seeds))].cpu())
        pred = model(x.to(device))
        loss = F.mse_loss(pred, y.to(device))
        losses.append(float(loss.detach().cpu()))
    seed_bank = torch.cat(seeds, dim=0)[:seed_bank_size] if seeds else torch.zeros((1, 1, frame_size, frame_size))
    return float(sum(losses) / max(1, len(losses))), seed_bank


def main() -> int:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(json.dumps({"dataset": args.dataset, "base_path": args.base_path, "device": device}, indent=2))

    trainset = load_well_dataset(args.base_path, args.dataset, args.train_split)
    valset = load_well_dataset(args.base_path, args.dataset, args.val_split)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=device == "cuda")
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=device == "cuda")

    model = ResidualFrameEmulator(channels=1, width=args.model_width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_state = None
    best_seed_bank = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, args.frame_size, args.max_train_batches)
        val_loss, seed_bank = evaluate(model, val_loader, device, args.frame_size, args.max_val_batches, args.seed_bank_size)
        row = {"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_seed_bank = seed_bank.cpu()

    if best_state is None:
        raise RuntimeError("No training batches completed.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_state_dict": best_state,
        "seed_bank": best_seed_bank,
        "meta": {
            "dataset": args.dataset,
            "base_path": args.base_path,
            "field": "auto_selected_scalar_slice",
            "frame_size": args.frame_size,
            "channels": 1,
            "model_width": args.model_width,
            "train_loss": history[-1]["train_loss"],
            "val_loss": best_val,
            "history": history,
            "datasets_trained": [
                {
                    "name": args.dataset,
                    "source": args.base_path,
                    "train_split": args.train_split,
                    "validation_split": args.val_split,
                    "train_items_available": len(trainset),
                    "validation_items_available": len(valset),
                    "n_steps_input": 1,
                    "n_steps_output": 1,
                    "field_extraction": "input_fields -> output_fields, auto-selected scalar 2D slice",
                    "emulator_frame_size": args.frame_size,
                }
            ],
            "training_budget": {
                "epochs": args.epochs,
                "max_train_batches_per_epoch": args.max_train_batches,
                "max_validation_batches_per_epoch": args.max_val_batches,
                "batch_size": args.batch_size,
                "model_width": args.model_width,
                "seed_bank_size": args.seed_bank_size,
            },
            "artifact_format": "prometheus_well_frame_emulator_v1",
        },
    }
    torch.save(artifact, output)
    print(json.dumps({"saved": str(output), "best_val_loss": best_val, "history": history}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
