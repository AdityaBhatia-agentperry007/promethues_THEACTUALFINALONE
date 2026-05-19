from __future__ import annotations

from backend.agent.orchestrator import run_agent
from backend.crypto.pir_service import get_client


def main() -> int:
    request = "Confidential proprietary request: Predict MHD evolution for a supersonic sub-Alfvenic plasma case and flag instability risk."
    response = run_agent(request, "dashboard")
    fetch = get_client().fetch(response["parsed"]["scenario_index"], "dpf")
    print("canned_request=", request)
    print("scenario_index=", response["parsed"]["scenario_index"])
    print("risk=", round(response["result"]["risk"], 4))
    print("pir_reconstructed_equals_direct=", fetch["reconstructed_equals_direct"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

