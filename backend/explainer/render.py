from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


class RenderError(RuntimeError):
    pass


def render(code: str, scene_name: str, timeout: int = 120) -> str:
    tmp = Path(tempfile.mkdtemp(prefix="explainer_"))
    scene_file = tmp / "scene.py"
    scene_file.write_text(code, encoding="utf-8")
    command = [sys.executable, "-m", "manim", "-ql", "--media_dir", str(tmp), str(scene_file), scene_name]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.CalledProcessError as exc:
        raise RenderError(exc.stderr or exc.stdout or "manim failed") from exc
    except subprocess.TimeoutExpired as exc:
        raise RenderError(f"render timed out after {timeout}s") from exc
    except OSError as exc:
        raise RenderError(f"manim executable unavailable: {exc}") from exc

    output = tmp / "videos" / "scene" / "480p15" / f"{scene_name}.mp4"
    if not output.exists():
        raise RenderError("render produced no mp4")
    return str(output)
