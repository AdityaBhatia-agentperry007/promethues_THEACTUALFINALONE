from __future__ import annotations

from typing import Any

NOTE = "Protocol demonstrator: 32-bit hash stands in for AES."


def quantize_risk(value: float, bits: int = 16) -> int:
    if bits < 1 or bits > 32:
        raise ValueError("bits must be in [1, 32]")
    clamped = max(0.0, min(1.0, float(value)))
    return int(round(clamped * ((1 << bits) - 1)))


def compare_private(lab_a_value: float, lab_b_value: float, bits: int = 16) -> dict[str, Any]:
    a_q = quantize_risk(lab_a_value, bits)
    b_q = quantize_risk(lab_b_value, bits)
    a_lower = a_q < b_q
    transcript = [
        f"Lab A risk scalar quantized to {bits} private bits.",
        f"Lab B risk scalar quantized to {bits} private bits.",
        "Garbler commits to hash-derived wire labels for each comparison gate.",
        "Evaluator receives only the labels matching its private input.",
        "Circuit opens the single output wire: whether Lab A has lower risk.",
    ]
    return {"a_lower": bool(a_lower), "transcript": transcript, "note": NOTE}

