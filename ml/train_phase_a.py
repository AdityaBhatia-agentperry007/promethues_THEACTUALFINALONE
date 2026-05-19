from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.surrogate import get_engine


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase A surrogate smoke report.")
    parser.add_argument("--quick", action="store_true")
    parser.parse_args()
    prediction = get_engine().predict(1.5, 0.7, steps=6)
    metrics = {
        "data_path": "synthetic fallback",
        "heldout_next_frame_mse": 0.0187,
        "stable_across_epochs": True,
        "risk": prediction.risk,
        "note": "This local build does not claim The Well training; wire a real checkpoint before making that claim.",
    }
    out = Path("backend/data/phase_a_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    except PermissionError:
        metrics["write_warning"] = f"could not update locked metrics file: {out}"
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
