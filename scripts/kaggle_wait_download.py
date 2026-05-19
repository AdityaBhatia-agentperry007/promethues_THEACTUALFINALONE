from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_ID = "agentperry007/prometheus-well-mhd64-emulator"
OUT_DIR = ROOT / "kaggle" / "outputs"
MODEL_DST = ROOT / "backend" / "models" / "well_mhd64_emulator.pt"


def run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(cmd), flush=True)
    env = os.environ.copy()
    token_path = Path.home() / ".kaggle" / "access_token"
    if token_path.exists() and "KAGGLE_API_TOKEN" not in env:
        env["KAGGLE_API_TOKEN"] = token_path.read_text(encoding="utf-8").strip()
    return subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, encoding="utf-8", errors="replace", check=check)


def show(proc: subprocess.CompletedProcess[str]) -> str:
    text = (proc.stdout or "") + (proc.stderr or "")
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    return text


def wait() -> None:
    transient_failures = 0
    while True:
        proc = run(["kaggle", "kernels", "status", KERNEL_ID])
        text = show(proc).lower()
        if proc.returncode != 0:
            transient_failures += 1
            if transient_failures > 10:
                raise SystemExit("Too many transient Kaggle status failures.")
            time.sleep(60)
            continue
        transient_failures = 0
        if "complete" in text:
            return
        if "error" in text or "cancelled" in text:
            raise SystemExit(f"Kaggle kernel failed:\n{text}")
        time.sleep(60)


def main() -> int:
    wait()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    proc = run(["kaggle", "kernels", "output", KERNEL_ID, "-p", str(OUT_DIR), "-o"])
    text = show(proc)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    model_src = OUT_DIR / "well_mhd64_emulator.pt"
    report_src = OUT_DIR / "training_report.json"
    inventory_src = OUT_DIR / "training_inventory.json"
    if not model_src.exists():
        raise SystemExit(f"Expected model missing after output download: {model_src}\n{text}")
    MODEL_DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_src, MODEL_DST)
    report = json.loads(report_src.read_text(encoding="utf-8")) if report_src.exists() else {}
    inventory = json.loads(inventory_src.read_text(encoding="utf-8")) if inventory_src.exists() else None
    print(
        json.dumps(
            {
                "installed": str(MODEL_DST),
                "best_val_loss": report.get("best_val_loss"),
                "history_len": len(report.get("history", [])),
                "has_inventory": inventory is not None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

