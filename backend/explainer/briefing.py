from __future__ import annotations

import json
import re
from typing import Any

from backend.explainer.providers import dispatch_text

AUTHORING_SYSTEM_PROMPT = """You are a motion-design director and physics pedagogy writer.
Turn a natural-language physics concept into a concise brief for a single Manim scene.

Return exactly one JSON object and nothing else. Do not wrap it in markdown.

Required keys:
- title
- anchor_dataset
- route_reason
- visual_goal
- scene_beats
- labels
- palette
- camera
- narration
- safety_notes
- manim_prompt

Rules:
- manim_prompt must be a detailed, precise prompt for a Manim author.
- The prompt must describe a single self-contained scene, smooth transitions, clean labels,
  no external assets, and a 3Blue1Brown-like explanatory cadence.
- Mention the anchor dataset and the fact that the animation is an illustration, not raw simulation output.
- Keep the prompt compact enough for one scene under about 40 seconds.
"""

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_LEADING_VERB = re.compile(r"^(?:explain|show|create|generate|illustrate|visualize|demonstrate|render|build)\s+", re.IGNORECASE)


def author_brief(concept: str, route: dict[str, Any]) -> dict[str, Any]:
    payload = _compose_user_payload(concept, route)
    try:
        raw = dispatch_text(f"{AUTHORING_SYSTEM_PROMPT}\n\nCONCEPT:\n{concept}\n\nROUTE:\n{json.dumps(payload, ensure_ascii=True)}")
    except Exception:
        return _fallback_brief(concept, route)
    parsed = _parse_json(raw)
    if parsed:
        return _normalize_brief(parsed, concept, route)
    return _fallback_brief(concept, route)


def _compose_user_payload(concept: str, route: dict[str, Any]) -> dict[str, Any]:
    card = route.get("catalog_card") if isinstance(route, dict) else None
    return {
        "concept": concept,
        "requested_environment": route.get("requested_environment") if isinstance(route, dict) else None,
        "recommended_dataset": route.get("recommended_dataset") if isinstance(route, dict) else None,
        "reason": route.get("reason") if isinstance(route, dict) else None,
        "catalog_card": card,
    }


def _parse_json(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    candidates = [raw]
    match = _JSON_BLOCK.search(raw)
    if match:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _normalize_brief(data: dict[str, Any], concept: str, route: dict[str, Any]) -> dict[str, Any]:
    subject = _concept_phrase(concept)
    dataset = str(data.get("anchor_dataset") or route.get("recommended_dataset") or "MHD_64")
    route_reason = str(data.get("route_reason") or route.get("reason") or "Mapped to the nearest The Well domain.")
    visual_goal = str(data.get("visual_goal") or f"Explain {subject} with a single scene.")
    scene_beats = data.get("scene_beats") if isinstance(data.get("scene_beats"), list) else []
    labels = data.get("labels") if isinstance(data.get("labels"), list) else []
    palette = data.get("palette") if isinstance(data.get("palette"), list) else []
    camera = data.get("camera") if isinstance(data.get("camera"), list) else []
    narration = data.get("narration") if isinstance(data.get("narration"), list) else []
    safety_notes = data.get("safety_notes") if isinstance(data.get("safety_notes"), list) else []
    manim_prompt = str(data.get("manim_prompt") or _fallback_manim_prompt(subject, dataset, route_reason, visual_goal))
    title = str(data.get("title") or _default_title(subject))
    return {
        "title": title,
        "anchor_dataset": dataset,
        "route_reason": route_reason,
        "visual_goal": visual_goal,
        "scene_beats": [str(item) for item in scene_beats] or _default_beats(subject, dataset),
        "labels": [str(item) for item in labels] or _default_labels(subject),
        "palette": [str(item) for item in palette] or _default_palette(),
        "camera": [str(item) for item in camera] or ["single centered camera", "gentle dolly-ins only if needed"],
        "narration": [str(item) for item in narration] or _default_narration(subject),
        "safety_notes": [str(item) for item in safety_notes] or _default_safety_notes(),
        "manim_prompt": manim_prompt,
    }


def _fallback_brief(concept: str, route: dict[str, Any]) -> dict[str, Any]:
    subject = _concept_phrase(concept)
    dataset = str(route.get("recommended_dataset") or "MHD_64")
    route_reason = str(route.get("reason") or "Mapped to the nearest The Well domain.")
    visual_goal = f"Explain {subject} as a single clear scene."
    return {
        "title": _default_title(subject),
        "anchor_dataset": dataset,
        "route_reason": route_reason,
        "visual_goal": visual_goal,
        "scene_beats": _default_beats(subject, dataset),
        "labels": _default_labels(subject),
        "palette": _default_palette(),
        "camera": ["single centered camera", "no busy cuts"],
        "narration": _default_narration(subject),
        "safety_notes": _default_safety_notes(),
        "manim_prompt": _fallback_manim_prompt(subject, dataset, route_reason, visual_goal),
    }


def _concept_phrase(concept: str) -> str:
    cleaned = _LEADING_VERB.sub("", concept.strip())
    cleaned = re.sub(r"^the concept of\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;-")
    return cleaned or concept.strip()


def _fallback_manim_prompt(concept: str, dataset: str, route_reason: str, visual_goal: str) -> str:
    return (
        f"Create one self-contained Manim Scene that explains the concept: {concept}. "
        f"Anchor the explanation to The Well dataset {dataset}. {route_reason} "
        f"Visual goal: {visual_goal}. Use a 3Blue1Brown-like style with clean geometry, "
        "smooth transitions, clear labels, and no external assets. Keep every object near the frame center, "
        "avoid overlaps, and keep the sequence readable in under about 40 seconds. "
        "Prefer simple shapes, arrows, labels, and motion that communicates the underlying physics clearly."
    )


def _default_title(concept: str) -> str:
    return f"Animated explanation of {concept}".strip()


def _default_beats(concept: str, dataset: str) -> list[str]:
    return [
        f"Introduce the concept {concept} in one clean sentence.",
        f"Map the idea to the {dataset} physical domain.",
        "Show the core mechanism with a simple visual metaphor.",
        "Resolve with a final labeled takeaway and a quiet pause.",
    ]


def _default_labels(concept: str) -> list[str]:
    return [
        concept,
        "cause",
        "effect",
        "key takeaway",
    ]


def _default_palette() -> list[str]:
    return ["deep blue", "cyan", "amber", "white on black", "subtle red accent"]


def _default_narration(concept: str) -> list[str]:
    return [
        f"We are explaining {concept}.",
        "Keep the motion centered and easy to read.",
        "Use labels that connect the picture to the idea.",
    ]


def _default_safety_notes() -> list[str]:
    return [
        "Single scene only.",
        "No external media or file access.",
        "Keep all geometry inside the frame.",
    ]
