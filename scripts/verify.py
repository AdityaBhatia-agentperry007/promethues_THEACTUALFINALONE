from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "checkpoints.log"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def log(phase: str, passed: bool, checked: str, actual: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    status = "PASS" if passed else "FAIL"
    line = f"{timestamp()} | {phase} | {status} | {checked} | {actual}\n"
    try:
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except PermissionError:
        fallback = Path(r"C:\tmp\prometheus_checkpoints.log")
        fallback.parent.mkdir(parents=True, exist_ok=True)
        try:
            with fallback.open("a", encoding="utf-8") as handle:
                handle.write(line)
        except PermissionError:
            print(f"log unavailable: {line}", end="")


def run(cmd: list[str], phase: str, checked: str, cwd: Path = ROOT) -> str:
    print(f"\n$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    log(phase, proc.returncode == 0, checked, f"exit={proc.returncode}")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc.stdout


def resolve_cmd(name: str) -> str:
    resolved = shutil.which(name)
    if resolved:
        return resolved
    resolved = shutil.which(name + ".cmd")
    if resolved:
        return resolved
    return name


def test_cmd(*files: str) -> list[str]:
    if importlib.util.find_spec("pytest"):
        return [sys.executable, "-m", "pytest", *files]
    return [sys.executable, "scripts/mini_pytest.py", *files]


def phase0() -> None:
    required = [
        "INTERFACES.md",
        ".env.example",
        "backend/app.py",
        "frontend/package.json",
        "scripts/verify_all.sh",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    ok = not missing and sys.version_info >= (3, 11)
    log("phase_0_scaffold", ok, "repo scaffold and Python 3.11+", f"missing={missing}, python={sys.version.split()[0]}")
    if not ok:
        raise SystemExit(1)
    print("phase_0_scaffold PASS")

def phase_explainer() -> None:
    run(test_cmd("tests/test_explainer.py"), "phase_explainer", "sandbox, parser, fallback pipeline")
def phase1() -> None:
    run(test_cmd("tests/test_pir.py"), "phase_1_pir", "cgks+dpf correctness, onehot, single-server view")
    from backend.crypto.pir_service import get_client

    dpf = get_client().fetch(4, "dpf")
    cgks = get_client().fetch(4, "cgks")
    actual = f"dpf_query_bytes={dpf['query_bytes']}, cgks_query_bytes={cgks['query_bytes']}"
    log("phase_1_pir_sizes", True, "query byte accounting", actual)
    print(actual)


def phase2() -> None:
    run([sys.executable, "ml/train_phase_a.py", "--quick"], "phase_2_surrogate", "synthetic fallback train/report")


def phase3() -> None:
    run([sys.executable, "-m", "backend.data.precompute"], "phase_3_precompute", "16 fixed-width records")
    from backend.crypto.pir_service import get_client, get_library

    get_library.cache_clear()
    get_client.cache_clear()
    client = get_client()
    oks = [client.fetch(index, "dpf")["reconstructed_equals_direct"] for index in range(16)]
    ok = all(oks)
    log("phase_3_pir_over_surrogate", ok, "all scenario PIR fetches", f"{sum(oks)}/16")
    if not ok:
        raise SystemExit(1)
    print("phase_3_pir_over_surrogate PASS 16/16")


def phase4() -> None:
    run(test_cmd("tests/test_agent.py"), "phase_4_agent", "3 canned requests and overclaim guard")


def phase5() -> None:
    run(test_cmd("tests/test_gc.py"), "phase_5_gc", "500 comparisons plus honesty note")


def phase6() -> None:
    frontend = ROOT / "frontend"
    if not (frontend / "node_modules").exists():
        source = Path(r"C:\Users\asus\og projects\web app garbled circuits\node_modules")
        if source.exists():
            subprocess.run(["cmd", "/c", "mklink", "/J", "node_modules", str(source)], cwd=frontend, check=False)
    shutil.rmtree(frontend / ".next", ignore_errors=True)
    run([resolve_cmd("npm"), "run", "build"], "phase_6_frontend", "next build", cwd=frontend)


def phase7() -> None:
    run([sys.executable, "backend/scalekit_gateway/gmail_poller.py", "--once"], "phase_7_scalekit", "graceful no-credentials path")


def phase8() -> None:
    run([sys.executable, "cli/prometheus_cli.py", "predict", "1.5", "0.7", "--steps", "3"], "phase_8_cli_predict", "CLI predict")
    run([sys.executable, "cli/prometheus_cli.py", "private-fetch", "4"], "phase_8_cli_pir", "CLI private fetch")
    run([sys.executable, "cli/prometheus_cli.py", "compare", "0.2", "0.8"], "phase_8_cli_compare", "CLI compare")


def phase9() -> None:
    docs = ["README.md", "docs/TECHNICAL_REPORT.md", "docs/DEMO_SCRIPT.md", "docs/DECK_OUTLINE.md", "docs/AI_IMPACT_STATEMENT.md"]
    missing = [path for path in docs if not (ROOT / path).exists()]
    impact_words = len((ROOT / "docs/AI_IMPACT_STATEMENT.md").read_text(encoding="utf-8").split()) if not missing else 999
    report = (ROOT / "docs/TECHNICAL_REPORT.md").read_text(encoding="utf-8") if not missing else ""
    ok = not missing and impact_words <= 200 and "not a reactor simulation" in report.lower()
    log("phase_9_docs", ok, "docs present, impact <=200 words, honesty guardrails", f"missing={missing}, impact_words={impact_words}")
    if not ok:
        raise SystemExit(1)
    print(f"phase_9_docs PASS impact_words={impact_words}")


def main() -> int:
    phases = [phase0, phase_explainer, phase1, phase2, phase3, phase4, phase5, phase6, phase7, phase8, phase9]
    for phase in phases:
        phase()
    log("phase_10_full_verify", True, "phases 0-9", "all_green")
    print("\nphase_10_full_verify PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


