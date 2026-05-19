from __future__ import annotations

from typing import Any

from backend.agent import prompts
from backend.agent import tools


def _summarize_output(tool_name: str, output: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "run_surrogate":
        return {
            "risk": round(float(output["risk"]), 4),
            "resolution": output["meta"]["resolution"],
            "steps": output["meta"]["steps"],
            "source": output["meta"]["source"],
        }
    if tool_name == "private_fetch":
        return {
            "risk": round(float(output["record_summary"]["risk"]), 4),
            "reconstructed_equals_direct": output["reconstructed_equals_direct"],
            "method": output["method"],
            "query_bytes": output["query_bytes"],
            "index_bits_leaked_to_any_single_server": output["index_bits_leaked_to_any_single_server"],
        }
    if tool_name == "compare_private":
        return {"a_lower": output["a_lower"], "note": output["note"]}
    return output


def _trace(tool_name: str, input_payload: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    return {"tool": tool_name, "input": input_payload, "output_summary": _summarize_output(tool_name, output)}


def _guard_answer(text: str) -> str:
    lowered = text.lower()
    for phrase in prompts.FORBIDDEN_CLAIMS:
        if phrase in lowered:
            raise ValueError(f"forbidden overclaim in answer: {phrase}")
    return text


def run_agent(request_text: str, channel: str = "dashboard") -> dict[str, Any]:
    parsed = tools.parse_scenario(request_text)
    plan = ["parse_scenario", "run_surrogate"]
    trace: list[dict[str, Any]] = []

    trace.append(_trace("parse_scenario", {"request_text": request_text}, parsed))

    if parsed["intent"] == "compare":
        lab_a, lab_b = tools.extract_compare_values(request_text)
        comparison = tools.compare_private(lab_a, lab_b)
        trace.append(_trace("compare_private", {"lab_a_value": lab_a, "lab_b_value": lab_b}, comparison))
        answer = (
            f"Private comparison complete: Lab A lower risk = {comparison['a_lower']}. "
            "Only the boolean answer is revealed. Protocol demonstrator: 32-bit hash stands in for AES."
        )
        return {
            "parsed": parsed,
            "plan": ["parse_scenario", "compare_private", "summarize"],
            "tool_trace": trace,
            "result": {"risk": lab_a, "used_private_fetch": False, "comparison": comparison["a_lower"]},
            "answer_text": _guard_answer(answer),
        }

    prediction = tools.run_surrogate(parsed["mach_sonic"], parsed["mach_alfvenic"], steps=12)
    trace.append(
        _trace(
            "run_surrogate",
            {"mach_sonic": parsed["mach_sonic"], "mach_alfvenic": parsed["mach_alfvenic"], "steps": 12},
            prediction,
        )
    )
    risk = float(prediction["risk"])
    used_private_fetch = False

    if parsed["intent"] == "predict_private":
        plan.append("private_fetch")
        fetched = tools.private_fetch(parsed["scenario_index"], "dpf")
        trace.append(_trace("private_fetch", {"scenario_index": parsed["scenario_index"], "method": "dpf"}, fetched))
        risk = float(fetched["record_summary"]["risk"])
        used_private_fetch = True

    plan.append("summarize")
    privacy_sentence = (
        " The scenario was fetched through two-server PIR, so no single server receives the index."
        if used_private_fetch
        else ""
    )
    answer = (
        f"Selected scenario {parsed['scenario_index']} ({parsed['label']}). "
        f"Derived instability proxy risk is {risk:.3f}.{privacy_sentence} "
        "This is a demo surrogate result, not a reactor simulation or production-security claim."
    )
    return {
        "parsed": parsed,
        "plan": plan,
        "tool_trace": trace,
        "result": {"risk": risk, "used_private_fetch": used_private_fetch, "scenario_index": parsed["scenario_index"]},
        "answer_text": _guard_answer(answer),
    }

