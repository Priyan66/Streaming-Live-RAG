"""Single entrypoint (gate G1).

    python run_demo.py                      # build index, replay the guide's examples, write traces + dashboard
    python run_demo.py --realtime           # same, delivering chunks at their real timestamps
    python run_demo.py --scenario my.json   # replay any scenario file(s)
    python run_demo.py --corpus other/docs  # swap in a different corpus
    python run_demo.py --bench              # benchmark vs. baseline + ablations -> docs/benchmark_results.*
    python run_demo.py --all                # demo + benchmark

Runs unattended with no API key (extractive backend). Set LLM_BACKEND=openai plus
LLM_BASE_URL / LLM_API_KEY / LLM_MODEL to use a hosted model for cite-then-write.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import argparse
import asyncio
import json
import logging
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

from engine import DEFAULT_CORPUS, Runtime  # noqa: E402
from simulator.stream_replay import load_scenario, replay  # noqa: E402
from telemetry.dashboard import write_dashboard  # noqa: E402
from telemetry.logger import TraceLogger  # noqa: E402
from telemetry.validate import validate_trace  # noqa: E402

ECHO = {"chunk_received", "controller_decision", "branch_forked", "branch_pruned", "branch_promoted",
        "retrieval_started", "retrieval_reused", "fallback_sequential", "claim_stale", "claim_patched",
        "topic_shift", "followup_question", "refinement", "retrieval_decision", "utterance_end", "answer_version"}


def print_turn(out: dict):
    lat = out["latency"]
    print(f"\n  ── turn {out['turn']} [{out['kind']}] answer v{out['answer_version']}"
          f"{' (from v' + str(out['parent_version']) + ')' if out['parent_version'] else ''}")
    print(f"  transcript : {out['transcript']}")
    if out["sub_queries"]:
        print("  sub-queries: " + " | ".join(out["sub_queries"]))
    print("  answer     : " + out["answer"].replace("\n", "\n               "))
    print(f"  citations  : {out['citations']}")
    if out["uncertainty"]:
        print(f"  uncertainty: {out['uncertainty']}")
    first = "none" if lat["first_retrieval_s"] is None else f"{lat['first_retrieval_s']}s"
    print(f"  latency    : first retrieval at {first}, utterance end {lat['utterance_end_s']}s, "
          f"answer {lat['post_utterance_ms']} ms after end, early={lat['early_retrieval']}")
    b = out["branches"]
    print(f"  branches   : forked {b['forked']}, pruned {b['pruned']}, promoted {b['promoted']}; "
          f"retrievals {out['cost']['retrievals']}, generator calls {out['cost']['llm_calls']}")


async def run_scenarios(rt: Runtime, paths: list[Path], run_dir: Path, realtime: bool, echo: bool) -> list[dict]:
    logger = TraceLogger(run_dir / "events.jsonl", echo=echo, echo_filter=ECHO)
    engine = rt.engine(logger)
    all_out = []
    for p in paths:
        sc = load_scenario(p)
        print(f"\n=== {sc.get('id', p.stem)}: {sc.get('description', '')}")
        outs = await replay(engine, sc, realtime=realtime, on_turn=print_turn)
        all_out.append({"scenario": sc.get("id", p.stem), "turns": outs})
    logger.close()
    (run_dir / "outputs.json").write_text(json.dumps(all_out, indent=2, ensure_ascii=False), encoding="utf-8")
    report = validate_trace(logger.events, rt.index.valid_ids)
    (run_dir / "trace_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_dashboard(logger.events, run_dir / "dashboard.html")
    print(f"\nTrace validation (G6): {report['coverage_pct']}% of turns fully traced; "
          f"fabricated citation IDs: {report['fabricated_citations']}; "
          f"citations without a retrieval origin: {report['untraceable_citations']}")
    print(f"Artifacts: {run_dir / 'events.jsonl'}\n           {run_dir / 'outputs.json'}\n"
          f"           {run_dir / 'dashboard.html'}")
    return all_out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", nargs="*", type=Path, help="scenario JSON file(s); default: examples/*.json")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--realtime", action="store_true", help="deliver chunks at their real timestamps")
    ap.add_argument("--quiet", action="store_true", help="don't echo the live event stream")
    ap.add_argument("--bench", action="store_true", help="run only the benchmark")
    ap.add_argument("--all", action="store_true", help="demo + benchmark")
    ap.add_argument("--split", default="test", choices=["dev", "test"], help="benchmark split")
    ap.add_argument("--out", type=Path, default=ROOT / "runs")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    t0 = time.perf_counter()
    rt = Runtime(args.corpus)
    idx = rt.index
    print(f"Index: {len(idx.chunks)} chunks from {len({c.doc_id for c in idx.chunks})} documents, "
          f"{idx.n_clusters} topic clusters, encoder={idx.embedder.encoder.kind}, verifier={rt.verifier.kind}, "
          f"generator={rt.llm.name} ({time.perf_counter() - t0:.1f}s)")

    if not args.bench:
        paths = args.scenario or sorted((ROOT / "examples").glob("*.json"))
        run_dir = args.out / time.strftime("%Y%m%d-%H%M%S")
        asyncio.run(run_scenarios(rt, paths, run_dir, args.realtime, echo=not args.quiet))
    if args.bench or args.all:
        from bench.run_benchmark import run_benchmark

        asyncio.run(run_benchmark(rt, ROOT / "docs", split=args.split))


if __name__ == "__main__":
    main()
