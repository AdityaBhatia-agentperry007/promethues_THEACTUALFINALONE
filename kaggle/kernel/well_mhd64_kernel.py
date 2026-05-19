"""Kaggle GPU script: train a PROMETHEUS Well frame emulator.

Outputs:
  /kaggle/working/well_mhd64_emulator.pt
  /kaggle/working/training_report.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Kaggle P100 images can break if pip replaces the preinstalled CUDA-compatible
# torch build. Install The Well package without forcing a torch upgrade.
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--force-reinstall",
        "--no-cache-dir",
        "torch",
        "--index-url",
        "https://download.pytorch.org/whl/cu126",
    ],
    check=False,
)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "the_well", "--no-deps"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from the_well.data import WellDataset


DATASET = "MHD_64"
BASE_PATH = "hf://datasets/polymathic-ai/"
FRAME_SIZE = 128
BATCH_SIZE = 1
EPOCHS = 6
MAX_TRAIN_BATCHES = 600
MAX_VAL_BATCHES = 100
MODEL_WIDTH = 64
SEED_BANK_SIZE = 24
OUT = Path("/kaggle/working/well_mhd64_emulator.pt")
REPORT = Path("/kaggle/working/training_report.json")
INVENTORY = Path("/kaggle/working/training_inventory.json")


def choose_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        test = nn.Conv2d(1, 1, 1).cuda()
        _ = test(torch.zeros((1, 1, 4, 4), device="cuda"))
        return "cuda"
    except Exception as exc:
        print(f"CUDA unavailable for this Kaggle image/GPU combination; falling back to CPU: {exc}", flush=True)
        return "cpu"


class ResidualFrameEmulator(nn.Module):
    def __init__(self, channels: int = 1, width: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, width, 5, padding=2),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, channels, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def iter_tensors(obj):
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from iter_tensors(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from iter_tensors(value)


def choose_field_tensor(batch) -> torch.Tensor:
    candidates = [tensor for tensor in iter_tensors(batch) if tensor.is_floating_point() and tensor.ndim >= 4]
    if not candidates:
        raise RuntimeError("Could not find a floating spatiotemporal field tensor in The Well batch.")
    return max(candidates, key=lambda tensor: tensor.numel())


def to_scalar_frames_from_tensor(tensor: torch.Tensor, frame_size: int) -> torch.Tensor:
    x = tensor.float()
    while x.ndim > 6:
        x = x[0]
    if x.ndim >= 5:
        x = select_2d_slice_after_batch_time(x)
    elif x.ndim == 4:
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
    return x.reshape(b, t, frame_size, frame_size)


def select_2d_slice_after_batch_time(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 5:
        return x
    tail = list(x.shape[2:])
    large_axes = [axis for axis, size in enumerate(tail) if size > 8]
    if len(large_axes) >= 2:
        keep_tail_axes = set(large_axes[:2])
    else:
        keep_tail_axes = set(sorted(range(len(tail)), key=lambda axis: tail[axis], reverse=True)[:2])
    index = [slice(None), slice(None)]
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


def make_pairs(batch, frame_size: int) -> tuple[torch.Tensor, torch.Tensor]:
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


def train_epoch(model, loader, optimizer, device: str) -> float:
    model.train()
    losses = []
    for step, batch in enumerate(loader):
        if step >= MAX_TRAIN_BATCHES:
            break
        x, y = make_pairs(batch, FRAME_SIZE)
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
def evaluate(model, loader, device: str) -> tuple[float, torch.Tensor]:
    model.eval()
    losses = []
    seeds = []
    for step, batch in enumerate(loader):
        if step >= MAX_VAL_BATCHES:
            break
        x, y = make_pairs(batch, FRAME_SIZE)
        if len(seeds) < SEED_BANK_SIZE:
            seeds.append(x[: max(0, SEED_BANK_SIZE - len(seeds))].cpu())
        pred = model(x.to(device))
        loss = F.mse_loss(pred, y.to(device))
        losses.append(float(loss.detach().cpu()))
    seed_bank = torch.cat(seeds, dim=0)[:SEED_BANK_SIZE] if seeds else torch.zeros((1, 1, FRAME_SIZE, FRAME_SIZE))
    return float(sum(losses) / max(1, len(losses))), seed_bank


def main() -> None:
    device = choose_device()
    print(json.dumps({"dataset": DATASET, "base_path": BASE_PATH, "device": device, "torch": torch.__version__}, indent=2))
    trainset = WellDataset(
        well_base_path=BASE_PATH,
        well_dataset_name=DATASET,
        well_split_name="train",
        n_steps_input=1,
        n_steps_output=1,
        use_normalization=False,
    )
    valset = WellDataset(
        well_base_path=BASE_PATH,
        well_dataset_name=DATASET,
        well_split_name="valid",
        n_steps_input=1,
        n_steps_output=1,
        use_normalization=False,
    )
    inventory = {
        "datasets_trained": [
            {
                "name": DATASET,
                "source": BASE_PATH,
                "train_split": "train",
                "validation_split": "valid",
                "train_items_available": len(trainset),
                "validation_items_available": len(valset),
                "n_steps_input": 1,
                "n_steps_output": 1,
                "field_extraction": "input_fields -> output_fields, auto-selected scalar 2D slice",
                "native_dataset_resolution": "MHD_64 nominally 64^2/64^3 depending field; emulator output is bilinear-resampled to FRAME_SIZE",
                "emulator_frame_size": FRAME_SIZE,
            }
        ],
        "training_budget": {
            "epochs": EPOCHS,
            "max_train_batches_per_epoch": MAX_TRAIN_BATCHES,
            "max_validation_batches_per_epoch": MAX_VAL_BATCHES,
            "batch_size": BATCH_SIZE,
            "model_width": MODEL_WIDTH,
            "seed_bank_size": SEED_BANK_SIZE,
        },
        "interpretation": {
            "what_the_model_learns": "one-step evolution of a scalar slice from The Well MHD_64 fields",
            "what_the_frames_mean": "normalized emulator frames showing relative field intensity evolving over time",
            "not_claimed": "not black-hole GR, not reactor-grade fusion, not a full multi-field MHD solver",
        },
    }
    INVENTORY.write_text(json.dumps(inventory, indent=2))
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=device == "cuda")
    val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=device == "cuda")
    model = ResidualFrameEmulator(channels=1, width=MODEL_WIDTH).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    history = []
    best_val = float("inf")
    best_state = None
    best_seed_bank = None
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, seed_bank = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_seed_bank = seed_bank.cpu()
    if best_state is None:
        raise RuntimeError("No training batches completed.")
    artifact = {
        "model_state_dict": best_state,
        "seed_bank": best_seed_bank,
        "meta": {
            "dataset": DATASET,
            "base_path": BASE_PATH,
            "field": "auto_selected_scalar_slice",
            "frame_size": FRAME_SIZE,
            "channels": 1,
            "model_width": MODEL_WIDTH,
            "train_loss": history[-1]["train_loss"],
            "val_loss": best_val,
            "history": history,
            "datasets_trained": inventory["datasets_trained"],
            "training_budget": inventory["training_budget"],
            "artifact_format": "prometheus_well_frame_emulator_v1",
        },
    }
    torch.save(artifact, OUT)
    REPORT.write_text(
        json.dumps(
            {
                "saved": str(OUT),
                "best_val_loss": best_val,
                "history": history,
                "inventory": inventory,
            },
            indent=2,
        )
    )
    print(REPORT.read_text())


if __name__ == "__main__":
    main()
