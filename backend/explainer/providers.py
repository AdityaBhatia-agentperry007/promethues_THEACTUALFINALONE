from __future__ import annotations

import os

from backend import config
from backend.llm_explain import _request_json


def dispatch_text(prompt: str) -> str:
    errors: list[str] = []
    for provider in _provider_order(config.EXPLAINER_PROVIDER):
        try:
            if provider == "vertex":
                text = _vertex(prompt)
            elif provider == "gemini":
                text = _gemini_api(prompt)
            elif provider == "openai":
                text = _openai(prompt)
            else:
                errors.append(f"{provider}: unknown provider")
                continue
            if text.strip():
                return text.strip()
            errors.append(f"{provider}: empty response")
        except Exception as exc:  # noqa: BLE001 - provider failover is intentional for demos.
            errors.append(f"{provider}: {_safe_provider_error(exc)}")
    raise RuntimeError("; ".join(errors) or "no explainer provider available")


def _provider_order(selected: str) -> list[str]:
    selected = (selected or "openai").lower()
    order: list[str] = []
    if selected in {"auto", "openai", "gemini", "vertex"} and selected != "auto":
        order.append(selected)
    if os.getenv("OPENAI_API_KEY") and "openai" not in order:
        order.append("openai")
    if os.getenv("GEMINI_API_KEY") and "gemini" not in order:
        order.append("gemini")
    if config.GOOGLE_CLOUD_PROJECT and "vertex" not in order:
        order.append("vertex")
    return order or ["openai", "gemini"]


def _safe_provider_error(exc: Exception) -> str:
    text = str(exc).replace(os.getenv("OPENAI_API_KEY", "__none__"), "[redacted]")
    text = text.replace(os.getenv("GEMINI_API_KEY", "__none__"), "[redacted]")
    return text[:240]


def _vertex(messages_text: str) -> str:
    import vertexai
    from vertexai.generative_models import GenerativeModel

    vertexai.init(project=config.GOOGLE_CLOUD_PROJECT, location=config.GOOGLE_CLOUD_LOCATION)
    model = GenerativeModel(config.VERTEX_MODEL)
    response = model.generate_content(
        messages_text,
        generation_config={"temperature": 0.3, "top_p": 0.95, "max_output_tokens": 4096},
    )
    return response.text or ""


def _gemini_api(messages_text: str) -> str:
    key = os.environ["GEMINI_API_KEY"]
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    payload = _request_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
        {"Content-Type": "application/json"},
        {
            "contents": [{"parts": [{"text": messages_text}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 4096},
        },
    )
    candidates = payload.get("candidates", [])
    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
    return "\n".join(str(part.get("text", "")) for part in parts).strip()


def _openai(messages_text: str) -> str:
    key = os.environ["OPENAI_API_KEY"]
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    payload = _request_json(
        "https://api.openai.com/v1/responses",
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": messages_text}],
                }
            ],
            "max_output_tokens": 4096,
        },
    )
    if payload.get("output_text"):
        return str(payload["output_text"]).strip()
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                return str(content.get("text", "")).strip()
    return ""
