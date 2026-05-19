from __future__ import annotations

import importlib.util
import inspect
import sys
import time
import traceback
from pathlib import Path


def _load_module(path: Path):
    name = "mini_pytest_" + path.stem + "_" + str(abs(hash(path)))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str]) -> int:
    root = Path.cwd()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    files = [Path(arg) for arg in argv[1:]]
    if not files:
        files = sorted(Path("tests").glob("test_*.py"))
    start = time.time()
    total = 0
    failed = 0
    print("============================= test session starts =============================")
    for path in files:
        module = _load_module(path)
        tests = [
            (name, fn)
            for name, fn in vars(module).items()
            if name.startswith("test_") and callable(fn) and len(inspect.signature(fn).parameters) == 0
        ]
        marks = []
        for name, fn in tests:
            total += 1
            try:
                fn()
                marks.append(".")
            except Exception:
                failed += 1
                marks.append("F")
                print(f"\nFAILED {path}::{name}")
                traceback.print_exc()
        print(f"{path} {''.join(marks)}")
    elapsed = time.time() - start
    if failed:
        print(f"=========================== {failed} failed, {total - failed} passed in {elapsed:.2f}s ===========================")
        return 1
    print(f"=========================== {total} passed in {elapsed:.2f}s ===========================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

