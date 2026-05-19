from __future__ import annotations

import json

from backend import config
from backend.crypto.pir_service import ResultLibrary


def main() -> int:
    library = ResultLibrary.build()
    try:
        library.save(config.RESULT_LIBRARY_PATH, config.SCENARIOS_PATH)
    except PermissionError:
        if not config.RESULT_LIBRARY_PATH.exists() or not config.SCENARIOS_PATH.exists():
            raise
        print("Using existing cached PIR library because Windows denied overwriting backend/data artifacts.", flush=True)
    report = {
        "n_scenarios": len(library.scenarios),
        "record_width": library.record_width,
        "result_library": str(config.RESULT_LIBRARY_PATH),
        "scenarios": str(config.SCENARIOS_PATH),
        "source": "deterministic synthetic surrogate fallback",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
