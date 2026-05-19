from __future__ import annotations

import threading
import time
import uuid
from typing import Any

_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}


def create() -> str:
    job_id = uuid.uuid4().hex
    with _LOCK:
        _JOBS[job_id] = {"status": "queued", "created_at": time.time()}
    return job_id


def set(job_id: str, state: dict[str, Any]) -> None:
    with _LOCK:
        current = _JOBS.get(job_id, {})
        current.update(state)
        current["updated_at"] = time.time()
        _JOBS[job_id] = current


def get(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        state = _JOBS.get(job_id)
        return dict(state) if state is not None else None