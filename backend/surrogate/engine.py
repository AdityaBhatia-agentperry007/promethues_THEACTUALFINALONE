from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from backend import config
from backend.surrogate.risk import risk_from_frames


@dataclass
class Prediction:
    frames: list[list[list[float]]]
    risk: float
    meta: dict[str, Any]


class SurrogateEngine:
    """Small deterministic surrogate wrapper for the live demo.

    The build spec expects a real MHD surrogate when the data/checkpoint is
    available. This implementation keeps the same interface and produces
    smooth plasma-shaped fields without claiming scientific novelty.
    """

    def __init__(self, field_size: int = config.DEFAULT_FIELD_SIZE) -> None:
        self.field_size = field_size
        self.loaded = True
        self.source_detail = "deterministic synthetic fallback; replace with trained MHD_64 checkpoint when available"

    def predict(self, mach_sonic: float, mach_alfvenic: float, steps: int = config.DEFAULT_STEPS) -> Prediction:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if mach_sonic <= 0 or mach_alfvenic <= 0:
            raise ValueError("Mach numbers must be positive")

        size = self.field_size
        x = np.linspace(0.0, 2.0 * np.pi, size, endpoint=False, dtype=np.float32)
        y = np.linspace(0.0, 2.0 * np.pi, size, endpoint=False, dtype=np.float32)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        kx = 1.0 + min(4.0, mach_sonic)
        ky = 1.0 + (2.5 if mach_alfvenic < 1.0 else 1.2)
        frames: list[list[list[float]]] = []

        for t in range(steps):
            phase = 0.28 * t * (0.7 + mach_sonic / 3.0)
            shear = 0.12 * t * (1.3 if mach_alfvenic < 1.0 else 0.6)
            base = (
                np.sin(kx * xx + phase)
                + 0.62 * np.cos(ky * yy - phase * 0.8)
                + 0.30 * np.sin(xx + yy + shear)
            )
            envelope = np.exp(-0.015 * t) * (1.0 + 0.04 * mach_sonic)
            shock = 0.16 * np.sin((xx - yy) * (0.5 + mach_sonic / 7.0) + phase)
            field = envelope * (base + shock)
            field = field / max(float(np.max(np.abs(field))), 1e-6)
            frames.append(np.round(field.astype(float), 4).tolist())

        risk = risk_from_frames(frames, mach_sonic, mach_alfvenic)
        meta = {
            "resolution": [size, size],
            "steps": steps,
            "source": "surrogate",
            "source_detail": self.source_detail,
            "risk_label": "derived illustrative proxy",
        }
        return Prediction(frames=frames, risk=risk, meta=meta)


@lru_cache(maxsize=1)
def get_engine() -> SurrogateEngine:
    return SurrogateEngine()

