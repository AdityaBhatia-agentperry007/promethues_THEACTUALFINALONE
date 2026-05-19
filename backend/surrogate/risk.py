from __future__ import annotations

import math

import numpy as np


def risk_from_frames(frames: list[list[list[float]]], mach_sonic: float, mach_alfvenic: float) -> float:
    """Derived instability proxy for the demo.

    This is not a trained disruption predictor. It combines field variance,
    spatial gradient energy, and the Mach-number regime into a bounded scalar.
    """
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("frames must have shape (steps, height, width)")
    grad_x = np.diff(arr, axis=1)
    grad_y = np.diff(arr, axis=2)
    gradient_energy = float(np.mean(np.abs(grad_x)) + np.mean(np.abs(grad_y)))
    variance = float(np.var(arr))
    regime = 0.18 * mach_sonic + (0.22 if mach_alfvenic < 1.0 else 0.06)
    raw = 1.9 * variance + 1.6 * gradient_energy + regime - 0.55
    risk = 1.0 / (1.0 + math.exp(-raw))
    return float(np.clip(risk, 0.0, 1.0))

