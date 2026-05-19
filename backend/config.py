from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
DATA_DIR = BACKEND_DIR / "data"
RESULT_LIBRARY_PATH = DATA_DIR / "result_library.npz"
SCENARIOS_PATH = DATA_DIR / "scenarios.json"
MODELS_DIR = BACKEND_DIR / "models"
WELL_EMULATOR_CHECKPOINT = Path(os.getenv("WELL_EMULATOR_CHECKPOINT", str(MODELS_DIR / "well_mhd64_emulator.pt")))

N_SCENARIOS = 16
DEFAULT_STEPS = 12
DEFAULT_FIELD_SIZE = 18
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "OPENAI").upper()
EXPORTS_DIR = ROOT_DIR / "exports"
FALLBACK_DIR = BACKEND_DIR / "explainer" / "fallback_videos"
EXPLAINER_PROVIDER = os.getenv("EXPLAINER_PROVIDER", os.getenv("PROMETHEUS_LLM_PROVIDER", "openai")).lower()
EXPLAINER_MAX_FIX_ATTEMPTS = int(os.getenv("EXPLAINER_MAX_FIX_ATTEMPTS", "2"))
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-flash")
CORS_ORIGINS = tuple(
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
)

EXPORTS_DIR.mkdir(exist_ok=True)
FALLBACK_DIR.mkdir(parents=True, exist_ok=True)


def _scenario_label(mach_sonic: float, mach_alfvenic: float, mode: str) -> str:
    sonic = "subsonic" if mach_sonic < 1.0 else "supersonic"
    alfvenic = "sub-Alfvenic" if mach_alfvenic < 1.0 else "super-Alfvenic"
    return f"{sonic}, {alfvenic} {mode}"


def build_scenarios() -> list[dict[str, Any]]:
    """Return the fixed 16-scenario demo library.

    The MHD_64 grid in the spec has 10 Mach-number pairs. To keep the demo
    library at N=16, six pairs are repeated under distinct qualitative modes.
    """
    rows = [
        (0.5, 0.7, "laminar reference"),
        (0.5, 2.0, "magnetic shear"),
        (0.7, 0.7, "edge-localized pulse"),
        (0.7, 2.0, "controlled jet"),
        (1.5, 0.7, "supersonic transition"),
        (1.5, 2.0, "turbulent mixing"),
        (2.0, 0.7, "shock-front case"),
        (2.0, 2.0, "high-beta channel"),
        (7.0, 0.7, "extreme supersonic pulse"),
        (7.0, 2.0, "extreme super-Alfvenic pulse"),
        (0.5, 0.7, "low-energy perturbation"),
        (0.7, 2.0, "transport barrier"),
        (1.5, 0.7, "sub-Alfvenic confinement"),
        (2.0, 2.0, "heated plasma column"),
        (7.0, 0.7, "disruption-stress proxy"),
        (1.5, 2.0, "stage demo scenario"),
    ]
    scenarios = []
    for index, (mach_sonic, mach_alfvenic, mode) in enumerate(rows):
        scenarios.append(
            {
                "index": index,
                "label": _scenario_label(mach_sonic, mach_alfvenic, mode),
                "mach_sonic": float(mach_sonic),
                "mach_alfvenic": float(mach_alfvenic),
                "mode": mode,
            }
        )
    return scenarios


SCENARIOS = build_scenarios()
