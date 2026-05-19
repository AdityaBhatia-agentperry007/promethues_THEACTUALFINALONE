from backend.agent.orchestrator import run_agent


def test_plain_predict_request():
    response = run_agent("Predict MHD evolution for a supersonic plasma case", "dashboard")
    assert response["parsed"]["intent"] == "predict"
    assert response["tool_trace"]
    assert response["result"]["used_private_fetch"] is False
    assert "reactor simulation" in response["answer_text"]


def test_confidential_predict_uses_private_fetch():
    response = run_agent(
        "Confidential proprietary request: predict a supersonic sub-Alfvenic plasma case",
        "dashboard",
    )
    assert response["parsed"]["intent"] == "predict_private"
    assert response["result"]["used_private_fetch"] is True
    assert any(step["tool"] == "private_fetch" for step in response["tool_trace"])


def test_compare_request():
    response = run_agent("Compare Lab A 0.31 and Lab B 0.58 for lower risk", "dashboard")
    assert response["parsed"]["intent"] == "compare"
    assert any(step["tool"] == "compare_private" for step in response["tool_trace"])
    assert "32-bit hash" in response["answer_text"]


def test_answer_has_no_forbidden_overclaim():
    forbidden = ("military-grade", "unbreakable", "fully homomorphic")
    response = run_agent("private PIR prediction", "dashboard")
    answer = response["answer_text"].lower()
    assert not any(phrase in answer for phrase in forbidden)

