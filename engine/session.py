"""Ephemeral, session-bound state (hard constraint: no cross-session memory).

A ``Session`` lives only in process memory for one conversation and is dropped
when the conversation ends. Nothing is persisted except the telemetry trace.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field

from engine.claims import AnswerGraph, Branch


class StreamClock:
    """Stream-time clock (seconds since the current turn started).

    ``realtime``: wall clock — chunks are delivered by sleeping until their timestamp.
    virtual: time jumps to each chunk's timestamp, then advances with measured
    wall-clock processing, so latency numbers stay honest without sleeping.
    """

    def __init__(self, realtime: bool = False):
        self.realtime = realtime
        self.reset()

    def reset(self):
        self._base_stream = 0.0
        self._base_wall = time.perf_counter()

    def now(self) -> float:
        return self._base_stream + (time.perf_counter() - self._base_wall)

    def advance_to(self, t: float):
        if self.realtime:
            return
        self._base_stream = max(t, self.now())
        self._base_wall = time.perf_counter()


@dataclass
class Session:
    clock: StreamClock
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    turn: int = 0
    buffer: str = ""
    n_chunks: int = 0
    prev_top_cluster: int | None = None
    branches: dict[int, Branch] = field(default_factory=dict)     # alive branches this turn, by cluster
    all_branches: list[Branch] = field(default_factory=list)
    graph: AnswerGraph | None = None
    answer_version: int = 0
    versions: list[dict] = field(default_factory=list)            # answer-version lineage
    last_answer: dict | None = None
    tasks: set = field(default_factory=set)
    turn_events: list[dict] = field(default_factory=list)          # retrieval events of this turn
    turn_cost: dict = field(default_factory=dict)
    suppressed: str | None = None
    first_retrieval_t: float | None = None
    cite_cache: dict = field(default_factory=dict)                 # (query, scope) -> future, this turn only
    early_patched: set = field(default_factory=set)

    def start_turn(self):
        self.turn += 1
        self.buffer = ""
        self.n_chunks = 0
        self.prev_top_cluster = None
        self.branches = {}
        self.turn_events = []
        self.turn_cost = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "est_usd": 0.0,
                          "would_be_prompt_tokens": 0, "retrievals": 0}
        self.suppressed = None
        self.first_retrieval_t = None
        self.cite_cache = {}
        self.early_patched = set()
        self.clock.reset()

    def spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    async def quiesce(self):
        while self.tasks:
            await asyncio.gather(*list(self.tasks), return_exceptions=True)
