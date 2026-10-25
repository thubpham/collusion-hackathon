from __future__ import annotations

import json
import threading
from pathlib import Path

KEYS = [
    "call_id", "game_id", "round", "player", "purpose", "provider", "model", "temperature",
    "max_tokens", "messages", "raw_response", "parsed", "parse_ok", "retry_index", "error",
    "tokens_in", "tokens_out", "latency_s", "timestamp",
]


class CallLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._n = 0
        if self.path.exists():
            with self.path.open() as f:
                self._n = sum(1 for _ in f)

    def append(self, **fields) -> str:
        with self._lock:
            self._n += 1
            call_id = f"c_{self._n:06d}"
            record = {"call_id": call_id, **fields}
            missing = [k for k in KEYS if k not in record]
            if missing:
                raise ValueError(f"call-log record missing keys: {missing}")
            record = {k: record[k] for k in KEYS}
            with self.path.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return call_id
