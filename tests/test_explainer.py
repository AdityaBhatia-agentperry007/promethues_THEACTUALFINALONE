from backend.explainer.briefing import author_brief
from backend.explainer import llm_codegen
from backend.explainer.fallback import nearest_fallback
from backend.explainer.pipeline import run_pipeline
from backend.explainer.sandbox import SandboxError, assert_safe
from backend.well_catalog import route_task_to_dataset

GOOD = "from manim import *\nclass Demo(Scene):\n    def construct(self):\n        self.play(Write(Text('hi')))\n"


def test_sandbox_allows_clean_scene():
    assert_safe(GOOD)


def test_sandbox_blocks_os():
    try:
        assert_safe("import os\nclass D(Scene):\n    def construct(self):\n        os.system('ls')\n")
        assert False, "should have raised"
    except SandboxError:
        pass


def test_sandbox_blocks_open():
    try:
        assert_safe("class D(Scene):\n    def construct(self):\n        open('x')\n")
        assert False, "should have raised"
    except SandboxError:
        pass


def test_sandbox_blocks_dunder_escape():
    try:
        assert_safe("class D(Scene):\n    def construct(self):\n        ().__class__.__bases__\n")
        assert False, "should have raised"
    except SandboxError:
        pass


def test_author_brief_anchors_dataset_and_prompt():
    route = route_task_to_dataset("show a supernova shock front forming")
    brief = author_brief("show a supernova shock front forming", route)
    assert brief["anchor_dataset"] == "supernova_explosion_64"
    assert "manim_prompt" in brief and brief["manim_prompt"]
    assert "3Blue1Brown" not in brief["title"] or isinstance(brief["title"], str)


def test_parse_finds_scene_and_adds_import():
    code, name = llm_codegen._parse("```python\nclass Foo(Scene):\n    def construct(self):\n        pass\n```")
    assert name == "Foo"
    assert "from manim import" in code


def test_parse_rejects_no_scene():
    try:
        llm_codegen._parse("```python\nprint('no scene')\n```")
        assert False, "should have raised"
    except llm_codegen.CodegenError:
        pass


def test_nearest_fallback_routes_dataset():
    route = route_task_to_dataset("show a supernova shock front forming")
    assert nearest_fallback(route).endswith("/supernova_intro.mp4")


def test_pipeline_falls_back_when_codegen_fails():
    original = llm_codegen.generate_code_from_brief

    def fail_codegen(_brief: dict):
        raise RuntimeError("missing test key")

    llm_codegen.generate_code_from_brief = fail_codegen
    try:
        result = run_pipeline("explain magnetic reconnection")
    finally:
        llm_codegen.generate_code_from_brief = original
    assert result["status"] == "done_fallback"
    assert result["url"].startswith("/fallback/") or result["url"].startswith("https://storage.googleapis.com/")
    assert "label" in result
    assert "brief" in result