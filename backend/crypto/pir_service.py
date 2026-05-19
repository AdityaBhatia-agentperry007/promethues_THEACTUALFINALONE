from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import numpy as np

from backend import config
from backend.crypto import pir
from backend.surrogate import get_engine

PIRMethod = Literal["dpf", "cgks"]


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _unpad(record: bytes) -> bytes:
    return record.rstrip(b"\0")


def _pad_records(records: list[bytes]) -> tuple[list[bytes], list[int], int]:
    if not records:
        raise ValueError("result library cannot be empty")
    widths = [len(record) for record in records]
    width = max(widths)
    padded = [record + b"\0" * (width - len(record)) for record in records]
    assert all(len(record) == width for record in padded)
    return padded, widths, width


def serialize_prediction_record(scenario: dict[str, Any], prediction: Any) -> bytes:
    payload = {
        "scenario": scenario,
        "risk": prediction.risk,
        "frames": prediction.frames,
        "meta": prediction.meta,
    }
    return _json_bytes(payload)


@dataclass
class ResultLibrary:
    records: list[bytes]
    raw_lengths: list[int]
    record_width: int
    scenarios: list[dict[str, Any]]

    @classmethod
    def build(cls, steps: int = config.DEFAULT_STEPS) -> "ResultLibrary":
        engine = get_engine()
        raw_records = [
            serialize_prediction_record(scenario, engine.predict(scenario["mach_sonic"], scenario["mach_alfvenic"], steps))
            for scenario in config.SCENARIOS
        ]
        records, raw_lengths, width = _pad_records(raw_records)
        return cls(records=records, raw_lengths=raw_lengths, record_width=width, scenarios=config.SCENARIOS)

    @classmethod
    def load_or_build(cls) -> "ResultLibrary":
        if not config.RESULT_LIBRARY_PATH.exists():
            library = cls.build()
            library.save(config.RESULT_LIBRARY_PATH, config.SCENARIOS_PATH)
            return library
        with np.load(config.RESULT_LIBRARY_PATH, allow_pickle=False) as data:
            matrix = data["records"].astype(np.uint8)
            raw_lengths = data["raw_lengths"].astype(int).tolist()
            scenarios = json.loads(str(data["scenarios_json"]))
        records = [bytes(row.tolist()) for row in matrix]
        width = int(matrix.shape[1])
        return cls(records=records, raw_lengths=raw_lengths, record_width=width, scenarios=scenarios)

    def save(self, library_path: Any, scenarios_path: Any) -> None:
        library_path = config.Path(library_path) if isinstance(library_path, str) else library_path
        scenarios_path = config.Path(scenarios_path) if isinstance(scenarios_path, str) else scenarios_path
        library_path.parent.mkdir(parents=True, exist_ok=True)
        matrix = np.asarray([list(record) for record in self.records], dtype=np.uint8)
        np.savez_compressed(
            library_path,
            records=matrix,
            raw_lengths=np.asarray(self.raw_lengths, dtype=np.int32),
            scenarios_json=json.dumps(self.scenarios, sort_keys=True),
            record_width=np.asarray([self.record_width], dtype=np.int32),
        )
        scenarios_path.write_text(json.dumps({"scenarios": self.scenarios}, indent=2), encoding="utf-8")

    def direct_record(self, index: int) -> bytes:
        self._check_index(index)
        return self.records[index]

    def parse_record(self, record: bytes) -> dict[str, Any]:
        return json.loads(_unpad(record).decode("utf-8"))

    def summary_for(self, record: bytes) -> dict[str, Any]:
        payload = self.parse_record(record)
        return {
            "risk": float(payload["risk"]),
            "resolution": payload["meta"]["resolution"],
            "steps": int(payload["meta"]["steps"]),
        }

    def _check_index(self, index: int) -> None:
        if index < 0 or index >= len(self.records):
            raise IndexError(f"scenario_index must be in [0, {len(self.records) - 1}]")


class PIRServer:
    def __init__(self, library: ResultLibrary, name: str) -> None:
        self.library = library
        self.name = name

    def answer_cgks(self, query: bytes) -> bytes:
        return pir.cgks_answer(self.library.records, query)

    def answer_dpf(self, key: Any) -> bytes:
        h = max(1, (len(self.library.records) - 1).bit_length())
        flags = pir.dpf_evalfull(key, len(self.library.records), h)
        acc = bytearray(self.library.record_width)
        for flag, record in zip(flags, self.library.records):
            if flag:
                acc = bytearray(pir._xor(bytes(acc), record))
        return bytes(acc)


def _encode_dpf_key(key: Any) -> bytes:
    root, party, correction_words = key
    payload = {
        "root_seed": root.hex(),
        "party": party,
        "correction_words": [
            {"seed_cw": seed_cw.hex(), "t_cw_left": int(t_left), "t_cw_right": int(t_right)}
            for seed_cw, t_left, t_right in correction_words
        ],
    }
    return _json_bytes(payload)


def _dpf_key_logical_size(key: Any) -> int:
    root, _party, correction_words = key
    return len(root) + 1 + sum(len(seed_cw) + 1 + 1 for seed_cw, _t_left, _t_right in correction_words)


class PIRClient:
    def __init__(self, library: ResultLibrary, server0: PIRServer, server1: PIRServer) -> None:
        self.library = library
        self.server0 = server0
        self.server1 = server1

    def fetch(self, scenario_index: int, method: PIRMethod = "dpf") -> dict[str, Any]:
        self.library._check_index(scenario_index)
        if method not in {"dpf", "cgks"}:
            raise ValueError("method must be 'dpf' or 'cgks'")

        if method == "cgks":
            q0, q1 = pir.cgks_query(len(self.library.records), scenario_index)
            a0, a1 = self.server0.answer_cgks(q0), self.server1.answer_cgks(q1)
            reconstructed = pir._xor(a0, a1)
            view0, view1 = q0, q1
            server0_view = view0.hex()
            server1_view = view1.hex()
            query_bytes = len(view0) + len(view1)
        else:
            h = max(1, (len(self.library.records) - 1).bit_length())
            k0, k1 = pir.dpf_gen(scenario_index, h)
            a0, a1 = self.server0.answer_dpf(k0), self.server1.answer_dpf(k1)
            reconstructed = pir._xor(a0, a1)
            view0, view1 = _encode_dpf_key(k0), _encode_dpf_key(k1)
            server0_view = view0.hex()
            server1_view = view1.hex()
            query_bytes = _dpf_key_logical_size(k0) + _dpf_key_logical_size(k1)

        direct = self.library.direct_record(scenario_index)
        return {
            "record_summary": self.library.summary_for(reconstructed),
            "reconstructed_equals_direct": reconstructed == direct,
            "server0_view": server0_view,
            "server1_view": server1_view,
            "index_bits_leaked_to_any_single_server": 0,
            "method": method,
            "query_bytes": query_bytes,
        }


@lru_cache(maxsize=1)
def get_library() -> ResultLibrary:
    return ResultLibrary.load_or_build()


@lru_cache(maxsize=1)
def get_client() -> PIRClient:
    library = get_library()
    return PIRClient(library, PIRServer(library, "server0"), PIRServer(library, "server1"))


def pir_fetch(scenario_index: int, method: PIRMethod = "dpf") -> dict[str, Any]:
    return get_client().fetch(scenario_index, method)
