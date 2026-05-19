from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_KERNEL = ROOT / "kaggle" / "kernel" / "well_mhd64_kernel.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit a generated The Well emulator trainer to Kaggle.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--frame-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-batches", type=int, default=180)
    parser.add_argument("--max-val-batches", type=int, default=40)
    parser.add_argument("--model-width", type=int, default=64)
    parser.add_argument("--seed-bank-size", type=int, default=24)
    parser.add_argument("--wait", action="store_true")
    return parser.parse_args()


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


def safe_name(dataset: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", dataset).strip("_")


def generate_kernel(args: argparse.Namespace, kernel_dir: Path) -> Path:
    text = BASE_KERNEL.read_text(encoding="utf-8")
    safe = safe_name(args.dataset)
    output_name = f"well_{safe}_emulator.pt"
    text = text.replace('DATASET = "MHD_64"', f'DATASET = "{args.dataset}"')
    text = text.replace("FRAME_SIZE = 128", f"FRAME_SIZE = {args.frame_size}")
    text = text.replace("EPOCHS = 6", f"EPOCHS = {args.epochs}")
    text = text.replace("MAX_TRAIN_BATCHES = 600", f"MAX_TRAIN_BATCHES = {args.max_train_batches}")
    text = text.replace("MAX_VAL_BATCHES = 100", f"MAX_VAL_BATCHES = {args.max_val_batches}")
    text = text.replace("MODEL_WIDTH = 64", f"MODEL_WIDTH = {args.model_width}")
    text = text.replace("SEED_BANK_SIZE = 24", f"SEED_BANK_SIZE = {args.seed_bank_size}")
    text = text.replace("well_mhd64_emulator.pt", output_name)
    text = text.replace(
        "MHD_64 nominally 64^2/64^3 depending field; emulator output is bilinear-resampled to FRAME_SIZE",
        f"{args.dataset} native grid; emulator output is bilinear-resampled to FRAME_SIZE",
    )
    text = text.replace(
        "one-step evolution of a scalar slice from The Well MHD_64 fields",
        f"one-step evolution of a scalar slice from The Well {args.dataset} fields",
    )
    text = text.replace(
        "not black-hole GR, not reactor-grade fusion, not a full multi-field MHD solver",
        "not a full production solver; demo checkpoint uses one auto-selected scalar slice",
    )
    script = kernel_dir / f"well_{safe}_kernel.py"
    script.write_text(text, encoding="utf-8")
    return script


def write_metadata(args: argparse.Namespace, username: str, kernel_dir: Path, script: Path) -> str:
    kernel_id = f"{username}/{args.slug}"
    metadata = {
        "id": kernel_id,
        "title": args.title,
        "code_file": script.name,
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
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return kernel_id


def wait_for_kernel(kernel_id: str, env: dict[str, str]) -> None:
    transient = 0
    while True:
        proc = run(["kaggle", "kernels", "status", kernel_id], env=env, check=False)
        text = (proc.stdout + "\n" + proc.stderr).lower()
        if proc.returncode != 0:
            transient += 1
            if transient > 10:
                raise SystemExit(f"Too many transient status failures:\n{text}")
            time.sleep(60)
            continue
        transient = 0
        if "complete" in text:
            return
        if "error" in text or "cancelled" in text:
            raise SystemExit(f"Kaggle kernel did not complete successfully:\n{text}")
        time.sleep(60)


def main() -> int:
    args = parse_args()
    env = kaggle_env()
    username = get_username(env)
    kernel_dir = ROOT / "kaggle" / "generated" / args.slug
    kernel_dir.mkdir(parents=True, exist_ok=True)
    script = generate_kernel(args, kernel_dir)
    kernel_id = write_metadata(args, username, kernel_dir, script)
    run(["kaggle", "kernels", "push", "-p", str(kernel_dir), "--accelerator", "gpu", "--timeout", "43200"], env=env)
    if not args.wait:
        print(json.dumps({"submitted": kernel_id, "wait": False, "kernel_dir": str(kernel_dir)}, indent=2))
        return 0
    wait_for_kernel(kernel_id, env)
    out_dir = ROOT / "kaggle" / f"outputs_{args.slug}"
    out_dir.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "kernels", "output", kernel_id, "-p", str(out_dir), "-o"], env=env)
    safe = safe_name(args.dataset)
    model_src = out_dir / f"well_{safe}_emulator.pt"
    model_dst = ROOT / "backend" / "models" / f"well_{safe}_emulator.pt"
    if not model_src.exists():
        raise SystemExit(f"Expected model missing: {model_src}")
    model_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_src, model_dst)
    print(json.dumps({"installed": str(model_dst), "kernel": kernel_id}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
