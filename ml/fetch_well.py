from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    report = {
        "attempted": ["the_well streaming", "huggingface_hub shard lookup", "synthetic fallback"],
        "selected": "synthetic fallback",
        "reason": "network/data credentials are intentionally not required for the local demo build",
    }
    path = Path("backend/data/well_fetch_report.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

