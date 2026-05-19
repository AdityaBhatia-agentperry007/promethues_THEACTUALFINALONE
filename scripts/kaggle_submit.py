from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kaggle" / "kernel"
KERNEL_SCRIPT = KERNEL_DIR / "well_mhd64_kernel.py"
KERNEL_META = KERNEL_DIR / "kernel-metadata.json"
KERNEL_SLUG = "prometheus-well-mhd64-emulator"
OUT_DIR = ROOT / "kaggle" / "outputs"
MODEL_DST = ROOT / "backend" / "models" / "well_mhd64_emulator.pt"


def run(cmd: list[str], *, env: dict[str, str], check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, encoding="utf-8", errors="replace")
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc


def kaggle_env() -> dict[str, str]:
    env = os.environ.copy()
    token = env.get("KAGGLE_API_TOKEN")
    token_path = Path.home() / ".kaggle" / "access_token"
    if not token and token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit("Missing Kaggle token. Save it to ~/.kaggle/access_token or set KAGGLE_API_TOKEN.")
    env["KAGGLE_API_TOKEN"] = token
    return env


def get_username(env: dict[str, str]) -> str:
    proc = run(["kaggle", "config", "view"], env=env)
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("- username:"):
            return stripped.split(":", 1)[1].strip()
    raise SystemExit("Could not read Kaggle username from `kaggle config view`.")


def write_metadata(username: str) -> str:
    kernel_id = f"{username}/{KERNEL_SLUG}"
    metadata = {
        "id": kernel_id,
        "title": "Prometheus Well MHD64 Emulator",
        "code_file": KERNEL_SCRIPT.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": "false",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": "",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    KERNEL_META.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return kernel_id


def wait_for_kernel(kernel_id: str, env: dict[str, str]) -> None:
    terminal = ("complete", "error", "cancelled")
    transient_failures = 0
    while True:
        proc = run(["kaggle", "kernels", "status", kernel_id], env=env, check=False)
        text = (proc.stdout + "\n" + proc.stderr).lower()
        if proc.returncode != 0:
            transient_failures += 1
            if transient_failures > 10:
                raise SystemExit(f"Too many transient Kaggle status failures:\n{text}")
            time.sleep(60)
            continue
        transient_failures = 0
        if any(state in text for state in terminal):
            if "complete" in text:
                return
            raise SystemExit(f"Kaggle kernel did not complete successfully:\n{text}")
        time.sleep(60)


def main() -> int:
    if not KERNEL_SCRIPT.exists():
        raise SystemExit(f"Missing kernel script: {KERNEL_SCRIPT}")
    env = kaggle_env()
    username = get_username(env)
    kernel_id = write_metadata(username)
    run(["kaggle", "kernels", "push", "-p", str(KERNEL_DIR), "--accelerator", "gpu", "--timeout", "43200"], env=env)
    wait_for_kernel(kernel_id, env)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "kernels", "output", kernel_id, "-p", str(OUT_DIR), "-o"], env=env)
    model_src = OUT_DIR / "well_mhd64_emulator.pt"
    if not model_src.exists():
        raise SystemExit(f"Expected Kaggle artifact missing: {model_src}")
    MODEL_DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_src, MODEL_DST)
    print(json.dumps({"copied": str(MODEL_DST), "kernel": kernel_id}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
