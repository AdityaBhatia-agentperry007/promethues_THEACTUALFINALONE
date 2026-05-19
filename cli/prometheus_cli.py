from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import typer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import config
from backend.agent.orchestrator import run_agent
from backend.crypto.gc_bridge import compare_private
from backend.crypto.pir_service import pir_fetch
from backend.surrogate import get_engine

app = typer.Typer(help="PROMETHEUS demo CLI.")


def _print(payload: dict[str, Any]) -> None:
    typer.echo(json.dumps(payload, indent=2))


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        config.BACKEND_URL + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError):
        if path == "/predict":
            prediction = get_engine().predict(payload["mach_sonic"], payload["mach_alfvenic"], payload.get("steps", 12))
            return {"frames": prediction.frames, "risk": prediction.risk, "meta": prediction.meta, "offline_fallback": True}
        if path == "/pir/fetch":
            result = pir_fetch(payload["scenario_index"], payload.get("method", "dpf"))
            result["offline_fallback"] = True
            return result
        if path == "/mpc/compare":
            result = compare_private(payload["lab_a_value"], payload["lab_b_value"])
            result["offline_fallback"] = True
            return result
        if path == "/agent":
            result = run_agent(payload["request_text"], payload.get("channel", "dashboard"))
            result["offline_fallback"] = True
            return result
        raise


@app.command()
def predict(mach_sonic: float, mach_alfvenic: float, steps: int = 12) -> None:
    payload = _post("/predict", {"mach_sonic": mach_sonic, "mach_alfvenic": mach_alfvenic, "steps": steps})
    payload["frames"] = f"{len(payload['frames'])} frames"
    _print(payload)


@app.command("private-fetch")
def private_fetch(scenario_index: int, method: str = "dpf") -> None:
    _print(_post("/pir/fetch", {"scenario_index": scenario_index, "method": method}))


@app.command()
def compare(lab_a_value: float, lab_b_value: float) -> None:
    _print(_post("/mpc/compare", {"lab_a_value": lab_a_value, "lab_b_value": lab_b_value}))


@app.command()
def agent(request_text: str, channel: str = "dashboard") -> None:
    _print(_post("/agent", {"request_text": request_text, "channel": channel}))


if __name__ == "__main__":
    app()
