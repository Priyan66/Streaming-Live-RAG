"""Structured JSON event log (spec §14, gate G6).

Every event carries: event type, session id, turn, stream timestamp (seconds
since the turn began, same axis as the transcript chunk timestamps) and a
wall-clock offset in ms. Events are kept in memory and optionally appended to a
JSONL file. The schema is documented in docs/telemetry_schema.md.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np


def _clean(v):
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating,)):
        return round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, float):
        return round(v, 4)
    return v


class TraceLogger:
    def __init__(self, path: str | Path | None = None, echo: bool = False, echo_filter: set[str] | None = None):
        self.events: list[dict] = []
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")
        else:
            self._fh = None
        self.echo = echo
        self.echo_filter = echo_filter
        self._t0 = time.perf_counter()

    def emit(self, session, event: str, **fields) -> dict:
        rec = {
            "event": event,
            "session_id": getattr(session, "id", None),
            "turn": getattr(session, "turn", None),
            "t_stream": round(session.clock.now(), 4) if session is not None else None,
            "t_wall_ms": round((time.perf_counter() - self._t0) * 1000, 2),
        }
        rec.update(_clean(fields))
        self.events.append(rec)
        if self._fh:
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()
        if self.echo and (self.echo_filter is None or event in self.echo_filter):
            print(_pretty(rec), file=sys.stdout, flush=True)
        return rec

    def for_session(self, session_id: str) -> list[dict]:
        return [e for e in self.events if e["session_id"] == session_id]

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


def _pretty(rec: dict) -> str:
    t = rec.get("t_stream")
    head = f"  [{t:6.3f}s] {rec['event']:<22}" if t is not None else f"  {rec['event']:<22}"
    skip = {"event", "session_id", "turn", "t_stream", "t_wall_ms"}
    body = ", ".join(f"{k}={v}" for k, v in rec.items() if k not in skip and v not in (None, [], {}))
    return head + (body[:220] + "…" if len(body) > 220 else body)
