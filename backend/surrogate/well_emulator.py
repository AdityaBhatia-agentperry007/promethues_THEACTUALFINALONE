from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

from backend import config
from backend.well_catalog import catalog_lookup, route_task_to_dataset


if TORCH_AVAILABLE:
    class ResidualFrameEmulator(nn.Module):
        """Small next-frame CNN used by the Kaggle/Colab Well training script."""

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
else:
    class ResidualFrameEmulator:
        """Dummy class when PyTorch is not available."""
        def __init__(self, *args, **kwargs) -> None:
            pass



@dataclass
class SimulationResult:
    task: str
    frames: list[list[list[float]]]
    meta: dict[str, Any]
    warning: str | None = None


def classify_task(task: str) -> dict[str, str]:
    route = route_task_to_dataset(task)
    dataset_hint = str(route["recommended_dataset"])
    lower = task.lower()
    if any(token in lower for token in ("black hole", "blackhole", "event horizon", "accretion")):
        return {
            "requested_domain": "black_hole",
            "dataset_hint": dataset_hint,
            "requested_environment": route["requested_environment"],
            "route_reason": route["reason"],
            "warning": (
                "Black-hole GR is not MHD_64. Train a post_neutron_star_merger or black-hole dataset checkpoint "
                "before claiming black-hole simulation."
            ),
        }
    if any(token in lower for token in ("supernova", "blast", "explosion")):
        return {
            "requested_domain": "supernova",
            "dataset_hint": dataset_hint,
            "requested_environment": route["requested_environment"],
            "route_reason": route["reason"],
            "warning": "",
        }
    domain = "mhd_plasma"
    if dataset_hint.startswith("acoustic"):
        domain = "acoustic_wave"
    elif dataset_hint == "rayleigh_benard":
        domain = "thermal_convection"
    elif dataset_hint == "gray_scott_reaction_diffusion":
        domain = "reaction_diffusion"
    elif dataset_hint == "planetswe":
        domain = "planetary_flow"
    elif dataset_hint == "shear_flow":
        domain = "shear_flow"
    return {
        "requested_domain": domain,
        "dataset_hint": dataset_hint,
        "requested_environment": route["requested_environment"],
        "route_reason": route["reason"],
        "warning": "",
    }


class WellEmulatorRuntime:
    def __init__(self, checkpoint_path: Path = config.WELL_EMULATOR_CHECKPOINT) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        self.model: ResidualFrameEmulator | None = None
        self.meta: dict[str, Any] = {}
        self.seed_bank: Any = None
        self.loaded = False
        self.load_error = None if TORCH_AVAILABLE else "PyTorch is not installed in this environment."
        if TORCH_AVAILABLE:
            self._load()

    def _load(self) -> None:
        if not TORCH_AVAILABLE:
            self.load_error = "PyTorch is not installed in this environment."
            return
        if not self.checkpoint_path.exists():
            self.load_error = f"missing checkpoint: {self.checkpoint_path}"
            return
        try:
            payload = torch.load(self.checkpoint_path, map_location=self.device)
            self.meta = dict(payload.get("meta", {}))
            channels = int(self.meta.get("channels", 1))
            width = int(self.meta.get("model_width", 48))
            self.model = ResidualFrameEmulator(channels=channels, width=width).to(self.device)
            self.model.load_state_dict(payload["model_state_dict"])
            self.model.eval()
            seed_bank = payload.get("seed_bank")
            if seed_bank is not None:
                self.seed_bank = seed_bank.float().to(self.device)
            self.loaded = True
        except Exception as exc:  # pragma: no cover - defensive status path
            self.load_error = f"could not load checkpoint: {exc}"

    def simulate(self, task: str, steps: int = 48) -> SimulationResult:
        task_info = classify_task(task)
        route = route_task_to_dataset(task)
        if not self.loaded or self.model is None:
            return SimulationResult(
                task=task,
                frames=[],
                meta={
                    "checkpoint_loaded": False,
                    "checkpoint_path": str(self.checkpoint_path),
                    "error": self.load_error,
                    "requested_domain": task_info["requested_domain"],
                    "dataset_hint": task_info["dataset_hint"],
                    "requested_environment": task_info["requested_environment"],
                    "route_reason": task_info["route_reason"],
                    "catalog_match": route["catalog_card"],
                    "prediction_method": "unavailable until a trained checkpoint is loaded",
                },
                warning="No The Well-trained emulator checkpoint is loaded. Run the Kaggle/Colab training script first.",
            )

        steps = max(1, min(int(steps), 256))
        x = self._seed_for_task(task)
        frames: list[np.ndarray] = []
        with torch.no_grad():
            for _ in range(steps):
                y = self.model(x)
                frame = y[0, 0].detach().cpu().numpy()
                frames.append(_normalize_frame(frame))
                x = y

        warning = task_info["warning"] or None
        dataset = str(self.meta.get("dataset", "unknown"))
        trained_for_request = task_info["dataset_hint"] == dataset
        if not trained_for_request:
            mismatch = (
                f"Requested dataset route is {task_info['dataset_hint']}, but loaded checkpoint dataset is {dataset}. "
                "Frames are generated by the loaded checkpoint only."
            )
            warning = f"{warning} {mismatch}" if warning else mismatch
        frame_stats = [_frame_stats(frame, index) for index, frame in enumerate(frames)]
        catalog_card = catalog_lookup(dataset)
        route_card = route["catalog_card"]
        interpretation = _build_interpretation(
            task=task,
            loaded_dataset=dataset,
            trained_for_request=trained_for_request,
            route=route,
            catalog_card=catalog_card,
            steps=steps,
        )
        return SimulationResult(
            task=task,
            frames=[np.round(frame, 5).tolist() for frame in frames],
            meta={
                "checkpoint_loaded": True,
                "checkpoint_path": str(self.checkpoint_path),
                "dataset": dataset,
                "field": self.meta.get("field", "unknown"),
                "frame_size": int(frames[0].shape[0]),
                "steps": steps,
                "requested_domain": task_info["requested_domain"],
                "dataset_hint": task_info["dataset_hint"],
                "requested_environment": task_info["requested_environment"],
                "route_reason": task_info["route_reason"],
                "trained_for_request": trained_for_request,
                "catalog_match": route_card,
                "loaded_dataset_card": catalog_card,
                "prediction_method": "one-step residual CNN emulator rolled forward autoregressively",
                "prediction_horizon": f"{steps} generated frames from a deterministic validation seed selected by task hash",
                "frame_stats": frame_stats,
                "interpretation": interpretation,
                "train_loss": self.meta.get("train_loss"),
                "val_loss": self.meta.get("val_loss"),
            },
            warning=warning,
        )

    def _seed_for_task(self, task: str) -> torch.Tensor:
        if self.seed_bank is not None and self.seed_bank.numel() > 0:
            digest = hashlib.sha256(task.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.seed_bank.shape[0]
            seed = self.seed_bank[index : index + 1]
            if seed.ndim == 3:
                seed = seed.unsqueeze(1)
            return seed.clone().to(self.device)
        size = int(self.meta.get("frame_size", 64))
        return torch.zeros((1, 1, size, size), dtype=torch.float32, device=self.device)


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(frame, [1.0, 99.0])
    if hi <= lo:
        return np.zeros_like(frame, dtype=np.float32)
    return np.clip((frame - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _frame_stats(frame: np.ndarray, index: int) -> dict[str, float | int]:
    dy = np.diff(frame, axis=0)
    dx = np.diff(frame, axis=1)
    gradient = 0.5 * (float(np.mean(np.abs(dx))) + float(np.mean(np.abs(dy))))
    return {
        "frame": index,
        "min": round(float(np.min(frame)), 6),
        "max": round(float(np.max(frame)), 6),
        "mean": round(float(np.mean(frame)), 6),
        "std": round(float(np.std(frame)), 6),
        "gradient_energy": round(gradient, 6),
    }


def _build_interpretation(
    task: str,
    loaded_dataset: str,
    trained_for_request: bool,
    route: dict[str, Any],
    catalog_card: dict[str, str] | None,
    steps: int,
) -> list[dict[str, str]]:
    requested = str(route["recommended_dataset"])
    loaded_card = catalog_card or {}
    routed_card = route.get("catalog_card") or {}
    if trained_for_request:
        fit = "good demo fit: the selected task routes to the same dataset as the loaded checkpoint"
    else:
        fit = f"limited fit: the task routes to {requested}, but only {loaded_dataset} is loaded locally"
    return [
        {
            "field": "task",
            "value": task,
            "why_it_matters": "This text selects the deterministic seed and the intended physics domain.",
        },
        {
            "field": "requested_environment",
            "value": str(route["requested_environment"]),
            "why_it_matters": "This is the presentation environment label for the task; it does not change the trained checkpoint.",
        },
        {
            "field": "recommended_well_dataset",
            "value": requested,
            "why_it_matters": str(route["reason"]),
        },
        {
            "field": "loaded_checkpoint_dataset",
            "value": loaded_dataset,
            "why_it_matters": fit,
        },
        {
            "field": "what_you_are_seeing",
            "value": "normalized scalar field intensity rendered with a color map and gradient-enhanced wave contours",
            "why_it_matters": "The canvas visualizes model output; it is not raw telescope imagery.",
        },
        {
            "field": "prediction",
            "value": f"{steps} autoregressive next-frame predictions",
            "why_it_matters": "Each frame is produced from the previous predicted frame, so long rollouts can drift.",
        },
        {
            "field": "trained_fields",
            "value": loaded_card.get("fields", "auto-selected scalar 2D slice"),
            "why_it_matters": "This tells viewers which physical variables the loaded model is connected to.",
        },
        {
            "field": "useful_for",
            "value": loaded_card.get("useful_for", ""),
            "why_it_matters": "Use this phrasing for the honest demo claim.",
        },
        {
            "field": "not_useful_for",
            "value": routed_card.get("not_useful_for", loaded_card.get("not_useful_for", "")),
            "why_it_matters": "This prevents overclaiming beyond the data and checkpoint.",
        },
    ]


@lru_cache(maxsize=1)
def get_well_emulator() -> WellEmulatorRuntime:
    return WellEmulatorRuntime()
