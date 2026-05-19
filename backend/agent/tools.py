from __future__ import annotations

import re
from typing import Any

from backend import config
from backend.crypto.gc_bridge import compare_private as compare_private_bridge
from backend.crypto.pir_service import pir_fetch
from backend.surrogate import get_engine

PRIVATE_WORDS = ("confidential", "proprietary", "private", "privacy", "pir", "operator never", "secret")


def _nearest_scenario(mach_sonic: float, mach_alfvenic: float) -> dict[str, Any]:
    return min(
        config.SCENARIOS,
        key=lambda scenario: (scenario["mach_sonic"] - mach_sonic) ** 2
        + (scenario["mach_alfvenic"] - mach_alfvenic) ** 2,
    )


def _extract_named_float(text: str, names: tuple[str, ...]) -> float | None:
    for name in names:
        pattern = rf"{name}\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def parse_scenario(request_text: str) -> dict[str, Any]:
    lower = request_text.lower()
    mach_sonic = _extract_named_float(lower, ("mach_sonic", "sonic", "ms"))
    mach_alfvenic = _extract_named_float(lower, ("mach_alfvenic", "alfvenic", "ma"))

    if mach_sonic is None:
        if "supersonic" in lower:
            mach_sonic = 1.5
        elif "subsonic" in lower:
            mach_sonic = 0.7
        elif "extreme" in lower:
            mach_sonic = 7.0
        else:
            mach_sonic = 1.5

    if mach_alfvenic is None:
        if "sub-alf" in lower or "sub alf" in lower or "subalf" in lower:
            mach_alfvenic = 0.7
        elif "super-alf" in lower or "super alf" in lower or "superalf" in lower:
            mach_alfvenic = 2.0
        else:
            mach_alfvenic = 0.7 if "confidential" in lower else 2.0

    if any(word in lower for word in ("compare", "lower-risk", "lower risk", "two labs", "lab a")):
        intent = "compare"
    elif any(word in lower for word in PRIVATE_WORDS):
        intent = "predict_private"
    else:
        intent = "predict"

    scenario = _nearest_scenario(mach_sonic, mach_alfvenic)
    return {
        "mach_sonic": float(scenario["mach_sonic"]),
        "mach_alfvenic": float(scenario["mach_alfvenic"]),
        "mode": scenario["mode"],
        "intent": intent,
        "scenario_index": int(scenario["index"]),
        "label": scenario["label"],
    }


def run_surrogate(mach_sonic: float, mach_alfvenic: float, steps: int = config.DEFAULT_STEPS) -> dict[str, Any]:
    prediction = get_engine().predict(mach_sonic, mach_alfvenic, steps)
    return {"risk": prediction.risk, "frames_ref": "inline", "frames": prediction.frames, "meta": prediction.meta}


def private_fetch(scenario_index: int, method: str = "dpf") -> dict[str, Any]:
    return pir_fetch(scenario_index, method)  # type: ignore[arg-type]


def compare_private(lab_a_value: float, lab_b_value: float) -> dict[str, Any]:
    return compare_private_bridge(lab_a_value, lab_b_value)


def summarize(context: str) -> str:
    return context


def extract_compare_values(request_text: str) -> tuple[float, float]:
    lower = request_text.lower()
    a = _extract_named_float(lower, ("lab_a_value", "lab a", "a"))
    b = _extract_named_float(lower, ("lab_b_value", "lab b", "b"))
    numbers = [float(value) for value in re.findall(r"(?<![a-z])0?\.\d+|(?<![a-z])1\.0", lower)]
    if a is None and numbers:
        a = numbers[0]
    if b is None and len(numbers) > 1:
        b = numbers[1]
    return float(0.42 if a is None else a), float(0.58 if b is None else b)

