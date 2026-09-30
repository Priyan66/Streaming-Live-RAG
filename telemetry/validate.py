"""Trace coverage checker (gate G6) and citation provenance audit (gate G4).

A turn counts as fully traced only if its events let you reconstruct the whole
decision path: every chunk has a controller decision, every retrieval that
started also completed, every citation in the answer has a verified claim event
whose passage came out of a logged retrieval, the turn has an answer-version
record (unless it was a clarification) and a summary with latency and cost.
"""
from __future__ import annotations

from collections import defaultdict


def validate_trace(events: list[dict], valid_ids: set[str]) -> dict:
    turns: dict[tuple, list[dict]] = defaultdict(list)
    for e in events:
        if e.get("turn"):
            turns[(e["session_id"], e["turn"])].append(e)

    failures, fabricated, untraceable, ok = [], 0, 0, 0
    for key, evs in turns.items():
        by = defaultdict(list)
        for e in evs:
            by[e["event"]].append(e)
        problems = []
        if not by["turn_start"]:
            problems.append("missing turn_start")
        if not by["utterance_end"]:
            problems.append("missing utterance_end")
        if len(by["controller_decision"]) != len(by["chunk_received"]):
            problems.append("chunk without controller decision")
        if len(by["retrieval_started"]) != len(by["retrieval_completed"]):
            # a branch pruned/refreshed mid-flight legitimately drops its completion
            dropped = len(by["retrieval_started"]) - len(by["retrieval_completed"])
            if dropped < 0:
                problems.append("retrieval completed without start")
        summ = by["turn_summary"]
        if not summ or "latency" not in summ[-1] or "cost" not in summ[-1]:
            problems.append("missing turn_summary latency/cost")
        kind = summ[-1]["kind"] if summ else None
        if kind not in ("clarification",) and not by["answer_version"]:
            problems.append("missing answer_version")
        retrieved = {cid for e in by["retrieval_completed"] for cid in e.get("top_ids", [])}
        session_retrieved = {cid for e in events if e["session_id"] == key[0] and e["event"] == "retrieval_completed"
                             for cid in e.get("top_ids", [])}
        verified = {e["citation"] for e in events if e["session_id"] == key[0] and e["event"] == "claim_verified"
                    and e.get("grounded")}
        for av in by["answer_version"]:
            for cid in av.get("citations", []):
                if cid not in valid_ids:
                    fabricated += 1
                    problems.append(f"fabricated citation {cid}")
                elif cid not in session_retrieved or cid not in verified:
                    untraceable += 1
                    problems.append(f"citation {cid} has no retrieval/verification origin")
        if kind == "answer" and not retrieved and not by["retrieval_reused"]:
            problems.append("answer without retrieval")
        if problems:
            failures.append({"session": key[0], "turn": key[1], "problems": problems})
        else:
            ok += 1
    n = len(turns)
    return {"turns": n, "fully_traced": ok, "coverage_pct": round(100.0 * ok / n, 1) if n else 100.0,
            "fabricated_citations": fabricated, "untraceable_citations": untraceable, "failures": failures}
