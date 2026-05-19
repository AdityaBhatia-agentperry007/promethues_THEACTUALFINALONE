from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any


def llm_status() -> dict[str, str]:
    provider = os.getenv("PROMETHEUS_LLM_PROVIDER", "").strip().lower()
    enabled = os.getenv("PROMETHEUS_LLM_EXPLAIN", "0") == "1"
    if not enabled:
        return {
            "status": "disabled",
            "provider": provider or "none",
            "reason": "PROMETHEUS_LLM_EXPLAIN is not set to 1",
        }
    missing = _missing_key(provider)
    if missing:
        return {"status": "missing_key", "provider": provider or "none", "reason": missing}
    return {"status": "ready", "provider": provider, "reason": "runtime env vars are present"}


def maybe_explain(task: str, meta: dict[str, Any], warning: str | None) -> dict[str, Any]:
    status = llm_status()
    if status["status"] != "ready":
        return status
    provider = status["provider"]
    payload = _prompt_payload(task, meta, warning)
    try:
        if provider == "openai":
            text = _call_openai(payload)
        elif provider == "anthropic":
            text = _call_anthropic(payload)
        elif provider == "gemini":
            text = _call_gemini(payload)
        else:
            return {"status": "unsupported_provider", "provider": provider, "reason": "use openai, anthropic, or gemini"}
    except (OSError, urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
        return {"status": "error", "provider": provider, "reason": _safe_error(exc)}
    return {"status": "ok", "provider": provider, "text": text}


def _missing_key(provider: str) -> str:
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY is not set"
    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY is not set"
    if provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        return "GEMINI_API_KEY is not set"
    if provider not in {"openai", "anthropic", "gemini"}:
        return "PROMETHEUS_LLM_PROVIDER must be openai, anthropic, or gemini"
    return ""


def _prompt_payload(task: str, meta: dict[str, Any], warning: str | None) -> str:
    small_meta = {
        "task": task,
        "dataset": meta.get("dataset"),
        "dataset_hint": meta.get("dataset_hint"),
        "trained_for_request": meta.get("trained_for_request"),
        "frame_size": meta.get("frame_size"),
        "steps": meta.get("steps"),
        "prediction_method": meta.get("prediction_method"),
        "prediction_horizon": meta.get("prediction_horizon"),
        "train_loss": meta.get("train_loss"),
        "val_loss": meta.get("val_loss"),
        "catalog_match": meta.get("catalog_match"),
        "loaded_dataset_card": meta.get("loaded_dataset_card"),
        "warning": warning,
    }
    return (
        "Explain this physics emulator output for a demo audience in 5 compact bullet lines. "
        "Do not claim the model is NASA imagery, a full solver, or trained on a dataset that is not loaded. "
        f"Metadata JSON: {json.dumps(small_meta, ensure_ascii=True)}"
    )


def _request_json(url: str, headers: dict[str, str], data: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _safe_error(exc: BaseException) -> str:
    text = str(exc)
    text = re.sub(r"key=[^&\s]+", "key=<redacted>", text)
    text = re.sub(r"Bearer\s+[A-Za-z0-9._-]+", "Bearer <redacted>", text)
    return text[:500]


def _call_openai(prompt: str) -> str:
    key = os.environ["OPENAI_API_KEY"]
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    data = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        "max_output_tokens": 260,
    }
    payload = _request_json(
        "https://api.openai.com/v1/responses",
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        data,
    )
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                return str(content.get("text", "")).strip()
    return str(payload.get("output_text", "")).strip()


def _call_anthropic(prompt: str) -> str:
    key = os.environ["ANTHROPIC_API_KEY"]
    model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
    payload = _request_json(
        "https://api.anthropic.com/v1/messages",
        {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        {
            "model": model,
            "max_tokens": 260,
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    parts = payload.get("content", [])
    if parts:
        return str(parts[0].get("text", "")).strip()
    return ""


def _call_gemini(prompt: str) -> str:
    key = os.environ["GEMINI_API_KEY"]
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    payload = _request_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
        {"Content-Type": "application/json"},
        {"contents": [{"parts": [{"text": prompt}]}]},
    )
    candidates = payload.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "\n".join(str(part.get("text", "")).strip() for part in parts if part.get("text")).strip()
