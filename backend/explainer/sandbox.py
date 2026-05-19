from __future__ import annotations

import ast

ALLOWED_IMPORTS = {"manim", "numpy", "np", "math", "random"}
BANNED_CALLS = {
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "input",
    "exit",
    "quit",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
}
BANNED_ATTR_ROOTS = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "socket",
    "requests",
    "urllib",
    "pathlib",
    "builtins",
    "importlib",
    "ctypes",
}


class SandboxError(ValueError):
    pass


def assert_safe(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise SandboxError(f"syntax error: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                root = name.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    raise SandboxError(f"import not allowed: {name.name}")

        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                raise SandboxError(f"import-from not allowed: {node.module}")

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BANNED_CALLS:
                raise SandboxError(f"banned call: {node.func.id}")

        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                raise SandboxError(f"dunder access not allowed: {node.attr}")
            base = node
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id in BANNED_ATTR_ROOTS:
                raise SandboxError(f"banned attribute root: {base.id}")

        if isinstance(node, ast.Name) and node.id.startswith("__") and node.id.endswith("__"):
            raise SandboxError(f"dunder name not allowed: {node.id}")