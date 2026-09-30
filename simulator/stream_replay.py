"""Stream replay harness: feeds timestamped transcript chunks to an engine.

Scenario format (JSON)::

    {"id": "example1",
     "turns": [
        {"chunks": [[0.0, "I need to plan a customer workshop in"],
                    [0.8, "Pune for 30 people, and I need"],
                    [1.6, "the cancellation policy and the catering options."]],
         "end": 2.1}
     ]}

Timestamps are seconds from the start of each turn (the guide's convention).
``realtime=True`` sleeps until each chunk is due and lets branch work overlap
with the incoming stream; the default virtual clock jumps to each chunk time,
drains the work that chunk triggered, and charges measured processing time to
stream time — equivalent to a single consumer and deterministic.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from engine.session import StreamClock


def load_scenario(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


async def replay(engine, scenario: dict, realtime: bool = False, on_turn=None) -> list[dict]:
    clock = StreamClock(realtime=realtime)
    session = engine.new_session(clock)
    outputs = []
    for turn in scenario["turns"]:
        engine.begin_turn(session)
        for t, text in turn["chunks"]:
            if realtime:
                await asyncio.sleep(max(0.0, t - clock.now()))
            else:
                clock.advance_to(t)
            await engine.on_chunk(session, text)
            if not realtime:
                await session.quiesce()
        end = turn.get("end", turn["chunks"][-1][0] + 0.5)
        if realtime:
            await asyncio.sleep(max(0.0, end - clock.now()))
        else:
            clock.advance_to(end)
        out = await engine.on_utterance_end(session)
        outputs.append(out)
        if on_turn:
            on_turn(out)
    engine.logger.emit(session, "session_end", turns=len(outputs), answer_versions=session.versions)
    return outputs


def chunk_text(text: str, start: float = 0.0, words_per_chunk: int = 5, gap: float = 0.8,
               end_pause: float = 0.5) -> dict:
    """Split plain text into a streamed turn (used by the benchmark generator)."""
    words = text.split()
    chunks, t = [], start
    for i in range(0, len(words), words_per_chunk):
        chunks.append([round(t, 2), " ".join(words[i:i + words_per_chunk])])
        t += gap
    return {"chunks": chunks, "end": round(chunks[-1][0] + end_pause, 2)}
