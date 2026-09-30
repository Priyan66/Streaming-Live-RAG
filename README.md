# Streaming Live RAG — Theme 4, Samsung PRISM GenAI Hackathon 3rd Edition

An event-driven RAG engine for live voice and chat. It listens to timestamped transcript chunks and **starts retrieval before the user finishes speaking**. It splits compound requests into parallel sub-queries, **patches individual claims** when a late constraint arrives instead of restarting, and makes citation fabrication **structurally impossible**. Every decision is traced as structured JSON.

```
python run_demo.py
```

That command builds the index, replays the guide's three worked examples plus a topic-shift scenario and a slot-change scenario, prints each answer with its citations and latency, and writes the trace, outputs and an HTML branch-timeline dashboard to `runs/<timestamp>/`. No API key or network access is needed after the first model download.

## Results (held-out test split, 30 conversations / 39 turns)

| Gate | Target | Streaming engine | Batch baseline |
|---|---|---|---|
| G1 Reproducibility | one command, unattended | `python run_demo.py --all` · 33 tests pass | — |
| G2 Early retrieval | ≥ 80 % | **90.6 %** (0 % false triggers on no-retrieval turns) | 0 % (100 % false triggers) |
| G3 Multi-intent split | ≥ 70 % | **100 %** (87.5 % exact intent count) | 0 % |
| G4 Citation support | ≥ 85 %, 0 fabricated IDs | **95.3 %** gold-document precision, **0** fabricated, 0 untraceable | 76.5 %, 0 fabricated |
| G5 Session refinement | refine, don't restart | **75 %** state preserved · **100 %** of refinements searched a scoped subset | 0 % · 0 % |
| G6 Telemetry | 100 % trace coverage | **100 %** | 100 % |
| Answer latency after utterance end | — | **p50 29 ms**, p95 197 ms | p50 445 ms, p95 536 ms |

Full tables with all three ablations are in [docs/benchmark_results.md](docs/benchmark_results.md). The analysis, including the edge-case failures, is in [docs/benchmark_report.md](docs/benchmark_report.md).

## How it works (one paragraph)

The engine uses three shared primitives: **one encoder** (`all-MiniLM-L6-v2`), **one k-means cluster index** over corpus chunks, and **one NLI verifier** (`nli-MiniLM2-L6-H768`). All four pipeline stages reuse them instead of adding a model per stage.

- **Fork policy:** each chunk's running buffer is compared with the cluster centroids once. Low affinity means wait. Low entropy forks one retrieval branch; near-tied clusters fork up to four cheap retrieval-only branches.
- **Decomposition:** clauses are mapped to clusters, and clauses that land on distinct clusters become parallel sub-queries.
- **Cite-then-write:** the generator sees only throwaway indices `[1..k]`, so it can't emit a real Doc ID. Its claim must then pass a keyword/entity overlap check and an NLI entailment check.
- **Claim patching:** a late constraint marks claims stale by slot-diff and embedding proximity. Only those claims are re-verified, over a scoped subset of the corpus.

See [docs/architecture_brief.md](docs/architecture_brief.md).

## Commands

| | |
|---|---|
| `python run_demo.py` | replay `examples/*.json` (guide Examples 1–3 plus topic shift and slot change) |
| `python run_demo.py --realtime` | the same, delivering chunks at their real timestamps (for the demo video) |
| `python run_demo.py --scenario file.json` | replay any scenario ([format](simulator/stream_replay.py)) |
| `python run_demo.py --corpus path/to/docs` | use another corpus (`.md`/`.txt` with `## §N` sections, or `.jsonl`) |
| `python run_demo.py --bench [--split dev]` | benchmark vs. baseline plus ablations → `docs/benchmark_results.*` |
| `python run_demo.py --all` | demo and benchmark |
| `python -m pytest -q` | 33 tests: guide Examples 1–3, zero-hallucination, patching, primitives |
| `docker compose up --build` | container: tests, then demo and benchmark (models baked in, runs offline) |

**Setup:** Python 3.11+, then `pip install -r requirements.txt`. The first run downloads the two models (~420 MB) from Hugging Face; later runs load them from the local cache.

**Hosted LLM (optional):** copy `.env.example` to `.env` and set `LLM_BACKEND=openai` plus `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL`. Any OpenAI-compatible endpoint works, including Gemini's. The default `extractive` generator is deterministic, offline and zero-token: it answers the same closed-index JSON contract by selecting verbatim corpus sentences.

## Repository layout

```
run_demo.py              single entrypoint (G1)
corpus/documents/        20 synthetic policy documents, "Doc_ID §Section" structure (71 chunks)
corpus/loader.py         chunking with provenance; also accepts .txt / .jsonl corpora
corpus/indexer.py        BM25 + FAISS + k-means clusters + sentence embeddings + gazetteer (built once, cached)
engine/primitives.py     embed() / cosine / entropy — the shared encoder primitive
engine/slots.py          extract_slots() — regex + corpus gazetteer, never generative
engine/fork_policy.py    §9 wait / fork / suppress / patch controller
engine/decomposer.py     §10 clause-to-cluster multi-intent decomposition
engine/retriever.py      BM25 + FAISS + cluster prior, fused with RRF, scoped search
engine/grounding.py      §11 cite-then-write, overlap + NLI verifier (micro-batched)
engine/claims.py         §8 Claim / Branch / SubQuery, §12 claim-level patching
engine/presentation.py   presentation-only turns (pitfall 4)
engine/pipeline.py       event-driven orchestrator, branch lifecycle (§13)
engine/baseline.py       naive batch RAG used for comparison
engine/llm_client.py     async pluggable generators (extractive default, OpenAI-compatible)
engine/session.py        ephemeral session state + stream clock
telemetry/               JSONL event logger, trace validator (G6), HTML dashboard
simulator/stream_replay.py  timestamped chunk replay (virtual or real time)
bench/                   benchmark conversations (dev/test) + runner/scorer
tests/                   pytest suite
docs/                    architecture brief, benchmark report + results, telemetry schema
```

## Constraints, and how they are met

- **Corpus isolation:** the default generator can only return corpus sentences. A hosted LLM only ever sees retrieved passages, and every claim it writes must be entailed by its cited chunk or it is dropped as uncertain.
- **No hardcoding:** nothing in `engine/` references corpus content or test prompts. The cluster count, cluster labels and gazetteer are all derived from whatever corpus is loaded. Thresholds are generic similarity and probability levels, set on the dev split.
- **Session-bound state:** a `Session` lives in memory for one conversation and nothing crosses sessions. Only the telemetry trace is written to disk.
- **Parsimony:** 2 small models (~110 M parameters in total) and 0 services. At most one generator call per sub-question, and forked branches spend retrieval only.
