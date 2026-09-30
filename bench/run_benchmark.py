"""Benchmark: streaming engine vs. batch baseline, plus three ablations.

    python run_demo.py --bench

Writes docs/benchmark_results.json (all numbers + per-turn records) and
docs/benchmark_results.md (tables). The narrative report is docs/benchmark_report.md.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from bench.scenarios import scenarios
from simulator.stream_replay import replay
from telemetry.logger import TraceLogger
from telemetry.validate import validate_trace

RETRIEVAL_KINDS = {"single", "compound", "refinement", "followup", "topic_shift"}
NO_RETRIEVAL_KINDS = {"presentation", "noise"}


def _doc(cid: str) -> str:
    return cid.split(" ")[0]


async def run_config(rt, name: str, kind: str = "streaming", split: str = "test", **overrides) -> dict:
    logger = TraceLogger()
    eng = rt.engine(logger, kind=kind, **overrides)
    rt.index.embedder.clear_cache()           # no config benefits from another config's query embeddings
    await replay(eng, {"id": "warmup", "turns": scenarios("dev")[0]["turns"]})
    logger.events.clear()
    rt.index.embedder.clear_cache()
    records = []
    t0 = time.perf_counter()
    for sc in scenarios(split):
        outs = await replay(eng, sc)
        prev = None
        for turn, out in zip(sc["turns"], outs):
            records.append(score_turn(sc["id"], turn["label"], out, prev, logger, eng.index))
            prev = out
    trace = validate_trace(logger.events, eng.index.valid_ids)
    return {"name": name, "records": records, "trace": trace, "seconds": round(time.perf_counter() - t0, 1),
            "metrics": aggregate(records, trace)}


def score_turn(sid, label, out, prev, logger, index) -> dict:
    kind = label["kind"]
    gold = label.get("gold", [])
    cited_docs = [_doc(c) for c in out["citations"]]
    turn_events = [e for e in logger.events if e["session_id"] == out["session_id"] and e["turn"] == out["turn"]]
    retr = [e for e in turn_events if e["event"] == "retrieval_completed"]
    covered = sum(any(d in g for d in cited_docs) for g in gold)
    gold_docs = {d for g in gold for d in g}
    rec = {
        "scenario": sid, "turn": out["turn"], "kind": kind, "system_kind": out["kind"],
        "sub_queries": len(out["sub_queries"]), "gold_intents": len(gold),
        "retrievals": out["cost"]["retrievals"], "llm_calls": out["cost"]["llm_calls"],
        "would_be_prompt_tokens": out["cost"].get("would_be_prompt_tokens", 0),
        "early": out["latency"]["early_retrieval"], "post_ms": out["latency"]["post_utterance_ms"],
        "first_retrieval_s": out["latency"]["first_retrieval_s"], "end_s": out["latency"]["utterance_end_s"],
        "citations": out["citations"], "n_citations": len(out["citations"]),
        "cited_in_gold": sum(d in gold_docs for d in cited_docs),
        "fabricated": sum(c not in index.valid_ids for c in out["citations"]),
        "intents_covered": covered, "uncertainty": bool(out["uncertainty"]),
        "full_corpus_searches": sum(e["chunks_searched"] >= e["corpus_size"] for e in retr),
        "answer": out["answer"],
    }
    if kind == "refinement" and prev is not None:
        before = {c["id"] for c in prev["claims"]}
        after = {c["id"] for c in out["claims"]}
        rec["state_preserved"] = bool(before) and before <= after
        rec["version_advanced"] = out["answer_version"] == (prev["answer_version"] or 0) + 1
    return rec


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


def _pct(xs):
    xs = list(xs)
    return round(100.0 * sum(bool(x) for x in xs) / len(xs), 1) if xs else None


def aggregate(recs: list[dict], trace: dict) -> dict:
    need = [r for r in recs if r["kind"] in RETRIEVAL_KINDS]
    comp = [r for r in recs if r["kind"] == "compound"]
    single = [r for r in recs if r["kind"] == "single"]
    none = [r for r in recs if r["kind"] in NO_RETRIEVAL_KINDS]
    ooc = [r for r in recs if r["kind"] == "out_of_corpus"]
    refine = [r for r in recs if r["kind"] == "refinement"]
    cited = sum(r["n_citations"] for r in recs if r["kind"] in RETRIEVAL_KINDS)
    post = sorted(r["post_ms"] for r in need)
    lead = [r["end_s"] - r["first_retrieval_s"] for r in need if r["first_retrieval_s"] is not None and r["early"]]
    return {
        "G2_early_retrieval_pct": _pct(r["early"] for r in need),
        "G2_false_trigger_pct": _pct(r["retrievals"] > 0 for r in none),
        "G3_compound_split_pct": _pct(r["sub_queries"] >= 2 for r in comp),
        "G3_exact_intent_count_pct": _pct(r["sub_queries"] == r["gold_intents"] for r in comp),
        "over_fragmentation_pct": _pct(r["sub_queries"] > 1 for r in single),
        "G4_citation_gold_precision_pct": round(100.0 * sum(r["cited_in_gold"] for r in need) / cited, 1) if cited else None,
        "G4_fabricated_ids": sum(r["fabricated"] for r in recs),
        "G4_trace_untraceable_citations": trace["untraceable_citations"],
        "intent_coverage_pct": round(100.0 * sum(r["intents_covered"] for r in need) /
                                     max(1, sum(r["gold_intents"] for r in need)), 1),
        "out_of_corpus_abstain_pct": _pct(r["n_citations"] == 0 for r in ooc),
        "G5_state_preserved_pct": _pct(r.get("state_preserved") for r in refine),
        "G5_no_full_corpus_search_pct": _pct(r["full_corpus_searches"] == 0 for r in refine),
        "G5_refinement_gold_cited_pct": _pct(r["intents_covered"] > 0 for r in refine),
        "G6_trace_coverage_pct": trace["coverage_pct"],
        "post_utterance_ms_mean": _mean(post),
        "post_utterance_ms_p50": post[len(post) // 2] if post else None,
        "post_utterance_ms_p95": post[min(len(post) - 1, int(0.95 * len(post)))] if post else None,
        "retrieval_lead_time_s_mean": _mean(lead),
        "retrievals_per_turn": _mean(r["retrievals"] for r in recs),
        "generator_calls_per_turn": _mean(r["llm_calls"] for r in recs),
        "would_be_prompt_tokens_per_turn": _mean(r["would_be_prompt_tokens"] for r in recs),
        "turns": len(recs),
    }


CONFIGS = [
    # name, engine kind, overrides
    ("Streaming (full system)", "streaming", {}),
    ("Baseline: batch RAG", "baseline", {}),
    ("A1 hybrid, no cluster prior", "streaming", {"cluster_prior": False}),
    ("A1 dense-only retrieval", "streaming", {"retrieval_mode": "dense"}),
    ("A1 sparse-only retrieval", "streaming", {"retrieval_mode": "sparse"}),
    ("A2 rule-based controller", "streaming", {"controller": "rule"}),
    ("A2 end-only controller", "streaming", {"controller": "end_only"}),
    ("A3 naive comma-split", "streaming", {"decomposer": "comma"}),
    ("A3 no decomposition", "streaming", {"decomposer": "none"}),
]

COLUMNS = [
    ("G2 early %", "G2_early_retrieval_pct"), ("false trig %", "G2_false_trigger_pct"),
    ("G3 split %", "G3_compound_split_pct"), ("exact n %", "G3_exact_intent_count_pct"),
    ("over-frag %", "over_fragmentation_pct"), ("G4 gold-prec %", "G4_citation_gold_precision_pct"),
    ("fabricated", "G4_fabricated_ids"), ("intent cov %", "intent_coverage_pct"),
    ("OOC abstain %", "out_of_corpus_abstain_pct"), ("G5 state %", "G5_state_preserved_pct"),
    ("G5 scoped %", "G5_no_full_corpus_search_pct"), ("G6 trace %", "G6_trace_coverage_pct"),
    ("post-utt ms p50", "post_utterance_ms_p50"), ("p95", "post_utterance_ms_p95"),
    ("retr/turn", "retrievals_per_turn"), ("gen calls/turn", "generator_calls_per_turn"),
]


def to_markdown(results: list[dict], split: str) -> str:
    lines = [f"# Benchmark results ({split} split)\n",
             "Generated by `python run_demo.py --bench`. See docs/benchmark_report.md for analysis.\n",
             "| config | " + " | ".join(c for c, _ in COLUMNS) + " |",
             "|---|" + "---|" * len(COLUMNS)]
    for r in results:
        m = r["metrics"]
        lines.append(f"| {r['name']} | " + " | ".join("–" if m[k] is None else str(m[k]) for _, k in COLUMNS) + " |")
    return "\n".join(lines) + "\n"


async def run_benchmark(rt, out_dir: Path, split: str = "test", configs=CONFIGS) -> list[dict]:
    results = []
    for name, kind, ov in configs:
        print(f"[bench] {name} ...", flush=True)
        r = await run_config(rt, name, kind, split, **ov)
        print(f"        {r['seconds']}s  " + ", ".join(f"{k}={r['metrics'][k]}" for _, k in COLUMNS[:8]), flush=True)
        results.append(r)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"benchmark_results{'' if split == 'test' else '_' + split}.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    md = to_markdown(results, split)
    (out_dir / f"benchmark_results{'' if split == 'test' else '_' + split}.md").write_text(md, encoding="utf-8")
    print(md)
    return results
