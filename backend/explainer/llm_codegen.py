from __future__ import annotations

import json
import re
from typing import Any

from backend.explainer.briefing import author_brief
from backend.explainer.providers import dispatch_text
from backend.explainer.system_prompts import FIX_ERROR_PROMPT, MANIM_SYSTEM_PROMPT

CODE_BLOCK = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
SCENE_NAME = re.compile(r"class\s+(\w+)\s*\(\s*Scene\s*\)")


class CodegenError(RuntimeError):
    pass


def generate_code(concept: str, route: dict[str, Any] | None = None) -> tuple[str, str, dict[str, Any]]:
    brief = author_brief(concept, route or {})
    code, scene_name = generate_code_from_brief(brief)
    return code, scene_name, brief


def generate_code_from_brief(brief: dict[str, Any]) -> tuple[str, str]:
    try:
        raw = dispatch_text(_compose(MANIM_SYSTEM_PROMPT, str(brief.get("manim_prompt", ""))))
        code, scene_name = _parse(raw)
        from backend.explainer.sandbox import assert_safe

        assert_safe(code)
        return code, scene_name
    except Exception:
        return _template_code(brief), "AutoExplainer"


def fix_code(concept: str, code: str, error: str, brief: dict[str, Any] | None = None) -> tuple[str, str]:
    fallback_brief = brief or {"title": concept, "visual_goal": concept}
    if error.lower().startswith(("syntax error", "banned", "import not allowed", "import-from not allowed", "dunder")):
        return _template_code(fallback_brief), "AutoExplainer"
    prompt = brief.get("manim_prompt", concept) if isinstance(brief, dict) else concept
    try:
        raw = dispatch_text(FIX_ERROR_PROMPT.format(concept=prompt, code=code, error=error[:2000]))
        fixed_code, scene_name = _parse(raw)
        from backend.explainer.sandbox import assert_safe

        assert_safe(fixed_code)
        return fixed_code, scene_name
    except Exception:
        return _template_code(fallback_brief), "AutoExplainer"


def _compose(system: str, user: str) -> str:
    return f"{system}\n\nMANIM BRIEF:\n{user}\n\nReturn one ```python block only."


def _parse(raw: str) -> tuple[str, str]:
    match = CODE_BLOCK.search(raw)
    code = (match.group(1) if match else raw).strip()
    if "from manim import" not in code and "import manim" not in code:
        code = "from manim import *\n\n" + code
    scene_names = SCENE_NAME.findall(code)
    if len(scene_names) != 1:
        raise CodegenError(f"expected exactly one `class X(Scene)`, found {len(scene_names)}")
    return code, scene_names[0]


def _template_code(brief: dict[str, Any]) -> str:
    title = _clip(str(brief.get("title") or "Animated physics explanation"), 58)
    goal = _clip(str(brief.get("visual_goal") or brief.get("manim_prompt") or title), 92)
    dataset = _clip(str(brief.get("anchor_dataset") or "The Well"), 42)
    labels = brief.get("labels") if isinstance(brief.get("labels"), list) else []
    beats = brief.get("scene_beats") if isinstance(brief.get("scene_beats"), list) else []
    label_0 = _clip(str(labels[0] if len(labels) > 0 else title), 34)
    label_1 = _clip(str(labels[1] if len(labels) > 1 else "cause"), 22)
    label_2 = _clip(str(labels[2] if len(labels) > 2 else "effect"), 22)
    beat_0 = _clip(str(beats[0] if len(beats) > 0 else goal), 72)
    beat_1 = _clip(str(beats[1] if len(beats) > 1 else f"Anchored to {dataset}"), 72)
    prompt_blob = " ".join(str(brief.get(key, "")) for key in ("title", "visual_goal", "manim_prompt", "anchor_dataset")).lower()
    scene_kind = "black_hole" if any(token in prompt_blob for token in ("black hole", "accretion", "photon ring", "doppler", "event horizon")) else "generic"
    return _BLACK_HOLE_TEMPLATE.format(
        title=json.dumps(title),
        goal=json.dumps(goal),
        dataset=json.dumps(dataset),
        label_0=json.dumps(label_0),
        label_1=json.dumps(label_1),
        label_2=json.dumps(label_2),
        beat_0=json.dumps(beat_0),
        beat_1=json.dumps(beat_1),
    ) if scene_kind == "black_hole" else _GENERIC_TEMPLATE.format(
        title=json.dumps(title),
        goal=json.dumps(goal),
        dataset=json.dumps(dataset),
        label_0=json.dumps(label_0),
        label_1=json.dumps(label_1),
        label_2=json.dumps(label_2),
        beat_0=json.dumps(beat_0),
        beat_1=json.dumps(beat_1),
    )


def _clip(text: str, limit: int) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    return clean if len(clean) <= limit else clean[: max(0, limit - 1)].rstrip() + "..."


_BLACK_HOLE_TEMPLATE = '''from manim import *
import numpy as np

class AutoExplainer(Scene):
    def construct(self):
        self.camera.background_color = "#08090d"
        title = Text({title}, color=WHITE).scale(0.42).to_edge(UP)
        goal = Text({goal}, color=GRAY_B).scale(0.22).next_to(title, DOWN, buff=0.12)
        self.play(Write(title), FadeIn(goal, shift=DOWN * 0.08), run_time=1.2)

        disk_back = ParametricFunction(lambda t: np.array([3.35*np.cos(t), 0.72*np.sin(t), 0]), t_range=[0, TAU], color="#8b3d21").set_stroke(width=14, opacity=0.45)
        disk_hot = ParametricFunction(lambda t: np.array([2.9*np.cos(t), 0.48*np.sin(t), 0]), t_range=[-0.2, 3.45], color="#ffb54a").set_stroke(width=15, opacity=0.95)
        disk_cool = ParametricFunction(lambda t: np.array([3.15*np.cos(t), 0.58*np.sin(t), 0]), t_range=[3.0, TAU + 0.1], color="#97421f").set_stroke(width=12, opacity=0.65)
        shadow = Circle(radius=0.72, color=BLACK, fill_opacity=1).set_stroke("#151515", width=2)
        photon = Circle(radius=0.95, color="#ffd76d").set_stroke(width=5, opacity=0.95)
        crescent = Arc(radius=1.13, start_angle=-1.15, angle=2.35, color="#fff0a8").set_stroke(width=9, opacity=1.0)
        group = VGroup(disk_back, disk_cool, disk_hot, photon, shadow, crescent).move_to(ORIGIN + DOWN * 0.15)
        self.play(Create(disk_back), Create(disk_cool), Create(disk_hot), run_time=1.8)
        self.play(FadeIn(shadow), Create(photon), Create(crescent), run_time=1.4)

        horizon_label = Text("event horizon", color=GRAY_A).scale(0.24).next_to(shadow, DOWN, buff=0.18)
        ring_label = Text("photon ring", color="#ffd76d").scale(0.25).move_to(RIGHT * 2.6 + UP * 1.18)
        doppler_label = Text("Doppler-bright crescent", color="#fff0a8").scale(0.25).move_to(LEFT * 2.75 + DOWN * 1.45)
        arrow1 = Arrow(ring_label.get_left(), photon.point_at_angle(0.72), buff=0.08, color="#ffd76d", stroke_width=3)
        arrow2 = Arrow(doppler_label.get_right(), crescent.point_from_proportion(0.35), buff=0.08, color="#fff0a8", stroke_width=3)
        self.play(FadeIn(horizon_label), Write(ring_label), GrowArrow(arrow1), Write(doppler_label), GrowArrow(arrow2), run_time=1.8)

        beat = Text({beat_0}, color=WHITE).scale(0.27).to_edge(DOWN)
        self.play(Write(beat), run_time=0.9)
        self.wait(0.4)
        self.play(Rotate(group, angle=0.65, about_point=ORIGIN, rate_func=smooth), run_time=2.0)
        self.play(Transform(beat, Text({beat_1}, color=WHITE).scale(0.27).to_edge(DOWN)), run_time=0.8)
        self.wait(0.6)

        badge = VGroup(
            Text("anchor", color=GRAY_B).scale(0.18),
            Text({dataset}, color="#7bdff2").scale(0.24),
            Text("illustration, not raw simulation output", color=GRAY_C).scale(0.18),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06).to_corner(DR)
        self.play(FadeIn(badge, shift=UP * 0.12), run_time=0.8)
        self.wait(1.0)
'''

_GENERIC_TEMPLATE = '''from manim import *
import numpy as np

class AutoExplainer(Scene):
    def construct(self):
        self.camera.background_color = "#0a0d12"
        title = Text({title}, color=WHITE).scale(0.42).to_edge(UP)
        goal = Text({goal}, color=GRAY_B).scale(0.22).next_to(title, DOWN, buff=0.12)
        self.play(Write(title), FadeIn(goal, shift=DOWN * 0.08), run_time=1.1)

        left = Circle(radius=0.82, color="#70d6ff", fill_opacity=0.12).move_to(LEFT * 3.0)
        center = Circle(radius=0.92, color="#ffd166", fill_opacity=0.12).move_to(ORIGIN)
        right = Circle(radius=0.82, color="#ef476f", fill_opacity=0.12).move_to(RIGHT * 3.0)
        left_t = Text({label_1}, color="#70d6ff").scale(0.28).move_to(left)
        center_t = Text({label_0}, color=WHITE).scale(0.25).move_to(center)
        right_t = Text({label_2}, color="#ef476f").scale(0.28).move_to(right)
        a1 = Arrow(left.get_right(), center.get_left(), buff=0.14, color=GRAY_A)
        a2 = Arrow(center.get_right(), right.get_left(), buff=0.14, color=GRAY_A)
        self.play(Create(left), Create(center), Create(right), Write(left_t), Write(center_t), Write(right_t), run_time=1.6)
        self.play(GrowArrow(a1), GrowArrow(a2), run_time=0.9)

        wave = ParametricFunction(lambda t: np.array([t, 0.38*np.sin(2.7*t), 0]), t_range=[-5.2, 5.2], color="#06d6a0").shift(DOWN * 1.55)
        wave_label = Text("motion reveals the mechanism", color="#06d6a0").scale(0.24).next_to(wave, DOWN, buff=0.18)
        self.play(Create(wave), FadeIn(wave_label), run_time=1.3)
        self.play(wave.animate.shift(UP * 0.2), center.animate.set_fill("#ffd166", opacity=0.28), run_time=1.1)

        beat = Text({beat_0}, color=WHITE).scale(0.27).to_edge(DOWN)
        self.play(Write(beat), run_time=0.9)
        self.wait(0.4)
        self.play(Transform(beat, Text({beat_1}, color=WHITE).scale(0.27).to_edge(DOWN)), run_time=0.8)
        badge = VGroup(
            Text("anchor", color=GRAY_B).scale(0.18),
            Text({dataset}, color="#70d6ff").scale(0.24),
            Text("generated illustration", color=GRAY_C).scale(0.18),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06).to_corner(DR)
        self.play(FadeIn(badge, shift=UP * 0.12), run_time=0.8)
        self.wait(1.0)
'''

