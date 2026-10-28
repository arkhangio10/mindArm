"""Append-only, thread-safe JSONL session records."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    """Return an ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


class SessionLog:
    """Flush every record; include a stable session ID and UTC timestamp."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:12]
        self.path = directory / f"{self.session_id}.jsonl"
        self._lock = Lock()
        self.write({"type": "session_start", "schema_version": 1})

    def write(self, record: dict[str, Any]) -> None:
        """Append exactly one JSON object per line without buffering across calls."""
        payload = {**record, "timestamp": utc_now(), "session_id": self.session_id}
        line = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
