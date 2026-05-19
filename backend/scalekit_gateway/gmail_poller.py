from __future__ import annotations

import argparse
import os

REQUIRED = ("SCALEKIT_ENV_URL", "SCALEKIT_CLIENT_ID", "SCALEKIT_CLIENT_SECRET")


def main() -> int:
    parser = argparse.ArgumentParser(description="Optional Scalekit Gmail poller for PROMETHEUS.")
    parser.add_argument("--once", action="store_true", help="Run one status check and exit.")
    parser.parse_args()

    missing = [name for name in REQUIRED if not os.getenv(name)]
    if missing:
        print(
            "Scalekit Gmail gateway not configured. "
            f"Missing: {', '.join(missing)}. Dashboard requests still use the same /agent path."
        )
        return 0

    try:
        import scalekit  # type: ignore  # noqa: F401
    except Exception:
        print("Scalekit credentials are present, but the Scalekit SDK is not installed in this environment.")
        return 0

    print("Scalekit SDK detected. Wire getOrCreateConnectedAccount and Gmail proxy calls here for live email.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

