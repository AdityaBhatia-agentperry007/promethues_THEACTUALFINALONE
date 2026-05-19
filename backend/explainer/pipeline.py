from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from backend import config
from backend.explainer import llm_codegen
from backend.explainer import render as render_module
from backend.explainer.fallback import nearest_fallback
from backend.explainer.sandbox import SandboxError, assert_safe
from backend.llm_explain import _safe_error
from backend.well_catalog import route_task_to_dataset

LABEL = "CONCEPT ANIMATION | generated illustration | anchored to The Well"


def run_pipeline(concept: str) -> dict[str, Any]:
    route = route_task_to_dataset(concept)
    attempts = 0
    last_error = ""
    brief: dict[str, Any] | None = None
    try:
        brief = llm_codegen.author_brief(concept, route)
        code, scene_name = llm_codegen.generate_code_from_brief(brief)
        for attempt in range(1, config.EXPLAINER_MAX_FIX_ATTEMPTS + 2):
            attempts = attempt
            try:
                assert_safe(code)
            except SandboxError as exc:
                last_error = _safe_error(exc)
                if attempt > config.EXPLAINER_MAX_FIX_ATTEMPTS:
                    break
                code, scene_name = llm_codegen.fix_code(concept, code, last_error, brief)
                continue

            try:
                mp4 = render_module.render(code, scene_name)
                destination = config.EXPORTS_DIR / f"{uuid.uuid4().hex}.mp4"
                shutil.move(mp4, destination)
                _cleanup_render_tmp(Path(mp4))
                return {
                    "status": "done",
                    "url": _publish(destination),
                    "route": route,
                    "label": LABEL,
                    "attempts": attempts,
                    "provider": config.EXPLAINER_PROVIDER,
                    "explanation": _explanation(concept, route, brief),
                    "brief": brief,
                    "manim_prompt": brief.get("manim_prompt") if brief else None,
                    "authoring": brief,
                }
            except render_module.RenderError as exc:
                last_error = _safe_error(exc)
                if attempt > config.EXPLAINER_MAX_FIX_ATTEMPTS:
                    break
                code, scene_name = llm_codegen.fix_code(concept, code, last_error, brief)
    except Exception as exc:  # noqa: BLE001 - a failed live render must become a fallback result.
        last_error = _safe_error(exc)

    return {
        "status": "done_fallback",
        "url": nearest_fallback(route),
        "route": route,
        "label": LABEL,
        "attempts": attempts,
        "provider": config.EXPLAINER_PROVIDER,
        "reason": last_error,
        "explanation": _explanation(concept, route, brief),
        "brief": brief or {},
        "manim_prompt": brief.get("manim_prompt") if brief else None,
        "authoring": brief or {},
    }


def _publish(path: Path) -> str:
    if config.GCS_BUCKET:
        from google.cloud import storage

        client = storage.Client()
        blob = client.bucket(config.GCS_BUCKET).blob(f"explainer/{path.name}")
        blob.upload_from_filename(str(path), content_type="video/mp4")
        return blob.public_url
    return f"/exports/{path.name}"


def _cleanup_render_tmp(mp4_path: Path) -> None:
    try:
        shutil.rmtree(mp4_path.parents[3], ignore_errors=True)
    except IndexError:
        return


def _explanation(concept: str, route: dict[str, Any], brief: dict[str, Any] | None) -> str:
    dataset = route.get("recommended_dataset", "MHD_64")
    reason = route.get("reason", "The prompt was mapped to the closest The Well physics domain.")
    visual_goal = brief.get("visual_goal") if isinstance(brief, dict) else None
    focus = f" Visual goal: {visual_goal}." if visual_goal else ""
    return (
        f"Animated Explainer for: {concept.strip()}. "
        f"Anchor dataset: {dataset}. {reason}.{focus} "
        "This is a generated concept illustration, not predicted simulation field data."
    )
