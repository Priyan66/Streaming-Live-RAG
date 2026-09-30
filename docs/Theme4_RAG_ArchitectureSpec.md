# Streaming Live RAG — Architecture Spec (Theme 4, Samsung PRISM GenAI Hackathon 3rd Edition)

Handoff document for implementation. This captures the full design worked out in planning, ready to build from. Read fully before writing code — the four core mechanisms deliberately share the same three primitives (embedding + cluster-centroid comparison, slot extraction, NLI/overlap verification), and that reuse is the architecture's central design principle, not an accident.

---

## 1. Problem statement (from official guide)

Standard RAG is batch-oriented: user finishes speaking, system retrieves, system answers. This is too slow and too rigid for live voice/chat:

- **High conversational latency** — waiting for a full multi-sentence utterance before retrieving.
- **Compound/multi-intent requests** — one utterance packages several distinct questions (e.g. venue capacity + cancellation policy + catering options).
- **Late-arriving constraints** — user adds/changes a detail mid-conversation ("actually the trip was international"); naive systems discard context and restart instead of refining.

### Target system must:
1. **Listen incrementally** — process timestamped transcript chunks in real time, predicting retrieval intent before the user finishes speaking.
2. **Decompose multi-intent queries** — identify and parallelize retrieval for multiple sub-questions in one utterance.
3. **Refine rather than restart** — selectively update answers/citations when late constraints arrive, preserving session state.
4. **Guarantee corpus grounding** — strict provenance/citation checks; explicit uncertainty when evidence is insufficient.

## 2. Hard constraints (non-negotiable)

- **Corpus isolation**: all factual claims must derive exclusively from the supplied corpus. No web scraping, no third-party knowledge, no unindexed parametric model memory.
- **No hardcoding / no precomputation**: benchmark replay eval is held-out and private. No embedded prompts/queries/canned responses tied to specific test content.
- **Rigorous factual grounding**: every factual assertion attributed to a verifiable chunk ID (`Doc_ID §Section`). If corpus lacks evidence, emit an explicit uncertainty indicator or request clarification — never fabricate.
- **Session-bound state**: memory is ephemeral, scoped to the active conversation only. No cross-session profiling/tracking.
- **Architectural parsimony**: every added component must justify its latency/compute cost. Multi-agent/complex orchestration is penalized if unjustified — **this is why the design below deliberately reuses 3 primitives across all 4 pipeline stages instead of adding a new model per stage.**

## 3. Evaluation gates (target thresholds)

| Gate | Criterion | Threshold |
|---|---|---|
| G1 | Reproducibility | Pass/fail — single-command launch, automated replay completes unattended |
| G2 | Early retrieval | ≥80% of eligible queries begin retrieval before transcript completion |
| G3 | Multi-intent identification | ≥70% of compound queries correctly split into ≥2 distinct sub-intents |
| G4 | Factual grounding | ≥85% citation support; **zero** fabricated/hallucinated document IDs |
| G5 | Session refinement | Late constraints narrow/update existing answers without clearing session state or re-running full-corpus search |
| G6 | Telemetry & observability | 100% trace coverage — timestamps, retrieval triggers, citations, answer-version lineage, token cost |

## 4. Known pitfalls to design around

1. **Eager/premature retrieval on noise** — triggering search on every token thrashes the system. Must wait for semantic stability.
2. **Context loss on late constraints** — discarding retrieved context on clarification, causing latency + disjointed replies.
3. **Citation hallucination** — fabricated `[Doc_999]` IDs or citing chunks that don't actually contain the stated fact.
4. **Ignoring presentation-only turns** — querying the vector DB when the user just asks to reformat/shorten/translate prior output wastes tokens and risks drift.
5. **Over-fragmenting sub-queries** — splitting a simple question into near-identical queries pollutes the reranker and exhausts token limits.

## 5. Recommended stack (no heavy infra, single-command run)

- **Language**: Python 3.11+, `asyncio` event-driven core.
- **Sparse retrieval**: `rank_bm25` (pure Python, no server).
- **Dense retrieval**: `sentence-transformers` (small model, e.g. `all-MiniLM-L6-v2`) + `faiss-cpu`.
- **Fusion**: Reciprocal Rank Fusion (RRF) combining BM25 + dense ranks.
- **NLI/entailment verifier**: small local cross-encoder NLI model (e.g. distilled MiniLM-NLI) — a few ms per pair, no external call.
- **LLM for synthesis**: pluggable async client behind an interface (Gemini/OpenAI-compatible) — must be async; a blocking call freezes the event loop (same lesson as Theme 5's kit).
- **Packaging**: plain CLI runner (`python run_demo.py`) or `docker compose up` — whichever is simpler to demo; must satisfy G1 with zero manual steps.
- **No separate vector DB service, no separate multi-intent model, no separate timing-classifier model** — deliberately, per the parsimony rule (see §6 unifying primitives).

## 6. The unifying design principle

Three cheap primitives, computed once per index/session and reused everywhere:

1. **`embed(text)`** — the same sentence-transformer encoder for corpus chunks, cluster centroids, transcript buffers, clauses, claims, and constraints.
2. **`cosine_sim(vec, cluster_centroids)`** — corpus chunks are clustered once at index time (k-means/hierarchical over chunk embeddings). This single comparison answers four different questions across four pipeline stages (see below).
3. **`extract_slots(text)`** — lightweight regex/NER slot extraction (dates, names, numbers, entities) — never generative, never invents information not actually spoken.
4. **NLI/overlap verifier** — `entails(premise_chunk, hypothesis_claim)` — used both for initial grounding and for patch re-verification.

This reuse across all four stages (fork-seeding, decomposition, patching, grounding) is the core "architectural parsimony" argument for the design brief: one encoder + one cluster index + one verifier, four jobs, not four separate models.

---

## 7. Pipeline architecture (event-driven, per official guide's 4 stages + our own mechanism)

```
Incoming Stream: [Chunk 0.0s] → [Chunk 0.8s] → [Chunk 1.6s] → [Utterance End 2.1s]
        │
        ▼
[1] Retrieval Controller (fork policy)   — decide Wait / Fork / Suppress
        │ (fork triggered)
        ▼
[2] Multi-Intent Decomposer               — clause-to-cluster mapping
        │
        ▼
[3] Corpus Retrieval & Fusion             — BM25 + FAISS + RRF, per sub-query, per branch
        │
        ▼
[4] Cite-then-Write Synthesis             — closed-set citation + independent verification
        │
        ▼
[5] Session-Aware Claim Graph             — patch in place on late constraints
        │
        ▼
Output: Streamed answer + grounded citations + uncertainty + telemetry
```

### Signature idea: Speculative Multi-Branch Retrieval with Retroactive Pruning

Instead of a single wait/retrieve decision, the controller **forks multiple cheap, disposable retrieval branches** at plausible sentence-boundaries *before* the utterance finishes — seeded not by LLM generation (too expensive per-branch) but by **corpus-cluster lookup** (near-free, corpus-grounded by construction). Retrieval for the eventual correct branch is already done — often before the user stops talking — so the synthesis step (the one expensive LLM call) fires the instant the utterance ends, not after. Wrong branches are pruned for near-zero cost (only BM25/FAISS lookups were spent, no LLM call).

**Fallback safety valve**: if no branch reaches the grounding threshold by utterance-end, fall back to the plain sequential wait→retrieve→decompose→synthesize pipeline. Worst case = exactly as good as the naive design; best case = a full utterance-length of latency saved, for free.

---

## 8. Core data structures

```python
@dataclass
class Claim:
    id: str
    text: str
    supporting_chunk_id: str | None      # e.g. "Doc_12 §2"; None if ungrounded
    source_sub_query: str
    topic_cluster: int                   # cluster id from the shared index
    slots_used: dict[str, str]
    embedding: Vector                    # cached, reused for conflict detection
    confidence: float                    # from NLI verifier
    version: int
    stale: bool = False

@dataclass
class Branch:
    id: str
    hypothesis_cluster: int              # seeded topic guess
    sub_queries: list[SubQuery]
    claims: list[Claim]
    grounding_score: float               # mean(claim.confidence for non-stale claims)
    status: Literal["forking", "alive", "promoted", "pruned", "patched"]
    answer_version: int = 1

@dataclass
class SubQuery:
    topic: str
    slots: dict[str, str]                # inherited global slots, not just clause-local
```

---

## 9. Mechanism 1 — Fork-seeding (Retrieval Controller)

**Goal**: decide when to start retrieval and what to retrieve, before the utterance ends, without an LLM call per hypothesis.

**Mechanism**:
1. On each incoming transcript chunk, embed the running buffer.
2. Compare against **precomputed cluster centroids** (k-means over corpus chunk embeddings, computed once at index time) — not individual chunks.
3. Compute entropy of the similarity distribution across clusters:
   - **Low max similarity** (below threshold) → not enough signal yet → wait, no fork.
   - **Low entropy** (one cluster dominates) → confident single direction → fork **1** branch (or **0** if it matches the currently promoted branch — this is presentation-only-turn suppression, pitfall #4, handled for free).
   - **High entropy** (2–4 clusters near-tied) → genuine ambiguity → fork **one branch per top cluster**, capped (e.g. max 4) for predictable cost.
4. Each branch's hypothesis = cluster's topic + slots extracted so far from the partial utterance (never generated text).
5. Branch immediately begins retrieval (BM25+FAISS) on `{cluster_topic} + {slots_so_far}`.

```python
async def fork_policy(partial_text: str, session: Session) -> list[Branch]:
    vec = await embed(partial_text)
    sims = cosine_sim(vec, cluster_centroids)
    entropy = softmax_entropy(sims)

    if max(sims) < LOW_CONF_THRESHOLD:
        return []                                          # not enough signal — wait

    if entropy < STABLE_THRESHOLD:
        top = [argmax(sims)]
        if matches_promoted_branch(top[0], session):
            return []                                       # suppress — nothing new (pitfall #4)
    else:
        top = top_k_clusters(sims, k=min(4, ...))            # ambiguous — hedge

    slots = extract_slots(partial_text)
    return [Branch(hypothesis_cluster=c, sub_queries=[SubQuery(topic=c, slots=slots)]) for c in top]
```

**Cost accounting**: forking is retrieval-only (single-digit ms per branch, no LLM). The one expensive call (LLM synthesis) happens exactly once, on the promoted branch — identical cost to a naive pipeline in the worst case, strictly better in the common case.

---

## 10. Mechanism 2 — Multi-Intent Decomposition

**Goal**: split a compound utterance into distinct sub-queries without over-fragmenting (pitfall #5), without an LLM call, and without losing shared context.

**Mechanism**:
1. **Syntactic segmentation** (rule/dependency-based — spaCy or conjunction/comma-boundary regex, no LLM) splits the completed utterance into clauses.
2. **Embed each clause separately**, compare to the same cluster centroids used in fork-seeding.
3. **Group clauses by cluster assignment**: clauses landing on the *same* cluster are merged into one sub-query (prevents over-fragmentation — granularity is decided by the corpus's own topic structure, not an arbitrary cap). Clauses landing on *distinct* clusters become *separate* sub-queries — this is the structural signature of genuine multi-intent.
4. **Slot inheritance**: extract slots **once**, globally, from the full utterance; attach the full global slot set to every sub-query (not just per-clause slots) — this is what preserves shared context ("Pune, 30 people" applies to the cancellation and catering sub-queries too, even though neither clause restates it).
5. **Confidence gate**: if clause-cluster assignment is low-confidence, or all clauses land on one cluster, fall back to a single holistic sub-query for the whole utterance — never force a split on unclear input.
6. **Non-destructive growth**: when a growing utterance reveals more clauses over time (per the guide's 0.8s→1.6s example), previously-created claims from earlier sub-queries are *kept*, not regenerated — new clauses only add new sub-queries/claims to the existing claim graph.

```python
async def decompose(utterance: str) -> list[SubQuery]:
    clauses = syntactic_split(utterance)
    global_slots = extract_slots(utterance)

    assignments = []
    for clause in clauses:
        vec = await embed(clause)
        sims = cosine_sim(vec, cluster_centroids)
        if max(sims) < LOW_CONF_THRESHOLD:
            continue
        assignments.append((argmax(sims), clause))

    by_cluster = group_by(assignments, key=lambda a: a[0])

    if len(by_cluster) <= 1:
        return [SubQuery(topic=utterance, slots=global_slots)]

    return [
        SubQuery(topic=merge_clause_text(clauses_in_cluster), slots=global_slots)
        for cluster_id, clauses_in_cluster in by_cluster.items()
    ]
```

**Ablation to report** (for benchmarking deliverable): cluster-based clause splitting vs. naive comma-splitting vs. LLM-prompted splitting — a clean 3-way comparison, no separate model needed to justify.

---

## 11. Mechanism 3 — Cite-then-Write Grounding

**Goal**: make citation fabrication structurally impossible (not just checked after the fact), and independently verify that a real citation actually supports the claim text.

**Step 1 — Closed-set citation.** Retrieved chunks for a sub-query are presented to the LLM with **local throwaway indices** (`[1]`, `[2]`, `[3]`), never real `Doc_ID` strings. The model can only ever emit an index; the real ID is resolved server-side after generation. This makes fabricating a nonexistent document ID structurally impossible — closes the vulnerability at the interface, not via post-hoc sanitization.

**Step 2 — Force selection before prose.** Single generation call, ordered JSON schema:
```json
{"selected_index": 1, "claim_text": "..."}
```
or, if nothing supports the sub-query:
```json
{"selected_index": null, "uncertainty_reason": "..."}
```
Because generation is autoregressive, requesting `selected_index` before `claim_text` conditions the written sentence on the already-committed evidence choice — cheap alternative to full constrained decoding.

**Step 3 — Independent deterministic verification** (never trust the model's own citation claim):
1. **Keyword/entity overlap check** — do the claim's key tokens (numbers, names, entities) actually appear in the selected chunk?
2. **NLI entailment check** — does `chunk[selected_index]` entail `claim_text`? (small local cross-encoder, a few ms/pair)

Only if **both** pass is the claim marked grounded. Otherwise, force to an explicit uncertainty state — never silently keep an unverified claim.

```python
async def cite_then_write(sub_query: str, candidates: list[Chunk]) -> ClaimResult:
    indexed = {i+1: c for i, c in enumerate(candidates)}
    prompt = build_closed_index_prompt(sub_query, indexed)
    resp = await llm_generate_json(prompt)

    if resp.selected_index is None:
        return ClaimResult(grounded=False, text=None, reason=resp.uncertainty_reason)

    chunk = indexed[resp.selected_index]
    overlap_ok = keyword_overlap(resp.claim_text, chunk.text) > OVERLAP_THRESHOLD
    entails = await nli_check(premise=chunk.text, hypothesis=resp.claim_text)

    grounded = overlap_ok and entails.label == "entailment" and entails.score > NLI_THRESHOLD
    return ClaimResult(
        grounded=grounded,
        text=resp.claim_text if grounded else UNCERTAIN_MARKER,
        chunk_id=chunk.doc_id if grounded else None,
        confidence=entails.score,
    )
```

`Branch.grounding_score = mean(claim.confidence for claim in branch.claims if not stale)` — used for branch promotion AND patch re-verification AND session-level uncertainty reporting. Same function, three call sites.

**Why this satisfies G4**: fabricated IDs are structurally impossible (Step 1); misattributed-but-real citations are caught by a verifier that doesn't share the generator's blind spots (Step 3, independent binary entailment check, not free generation).

---

## 12. Mechanism 4 — Claim-Level Patching (Session Refinement)

**Goal**: when a late constraint arrives, patch only the claims it actually invalidates — never blindly restart, never blindly append.

**Step 0 — Patch vs. new-fork gate.** Embed the constraint, compare to the *current promoted branch's* cluster centroid (not the whole corpus). If similarity is low, this isn't a refinement — it's a full intent change ("forget the flight, my TV is broken") → discard the claim graph, re-enter `fork_policy` from scratch. If similarity is reasonably high, proceed with patching below.

**Tier 1 — Slot-diff (near-free, deterministic).** Extract slots from the constraint. If any claim's `slots_used` overlaps a slot the constraint changes, mark that claim `stale`.

**Tier 2 — Embedding-proximity trigger (catches implicit dependencies).** Embed the constraint; compare against every claim's cached embedding (not the chunk's — the claim's own assertion, more precise). Claims above a conservative similarity threshold are also flagged `stale`, even with zero literal word overlap (e.g. "the trip was international" flags a reimbursement-rate claim that never mentioned "international"). Lean toward overflagging — a false positive costs one extra cheap re-retrieval; a false negative breaks G5.

**Tier 3 — Real re-verification, only for flagged claims.** Build a delta sub-query from `{claim.topic_cluster} + {updated slots}`, retrieve fresh chunks, re-run `cite_then_write`. If grounded, replace text/citation, bump `version`. If not, flip to explicit uncertainty (never delete silently — a visible retraction is a feature, not a bug, for the demo).

**Cascade check — one hop only.** After Tier 3 patches a claim, embed its *new* text and compare against remaining unflagged claims; flag second-order conflicts, patch those too. Do not recurse further — a need for a second hop signals a full intent change, not a refinement.

```python
async def apply_constraint(constraint_text: str, branch: Branch, session: Session):
    vec = await embed(constraint_text)
    if cosine_sim(vec, branch.cluster_centroid) < TOPIC_SHIFT_THRESHOLD:
        return await fork_policy(constraint_text, session)      # new intent, not a refinement

    new_slots = extract_slots(constraint_text)
    stale = set()

    for claim in branch.claims:                                  # Tier 1
        if set(new_slots) & set(claim.slots_used) and \
           any(new_slots[k] != claim.slots_used[k] for k in new_slots if k in claim.slots_used):
            stale.add(claim.id)

    for claim in branch.claims:                                  # Tier 2
        if cosine_sim(vec, claim.embedding) > CLAIM_CONFLICT_THRESHOLD:
            stale.add(claim.id)

    patched = []
    for claim in branch.claims:
        if claim.id not in stale:
            continue
        sub_query = build_delta_query(claim.topic_cluster, new_slots)
        chunks = await retrieve(sub_query)
        result = await cite_then_write(sub_query, chunks)
        claim.text = result.text if result.grounded else UNCERTAIN_MARKER
        claim.supporting_chunk_id = result.chunk_id
        claim.version += 1
        patched.append(claim)

    for other in branch.claims:                                  # one-hop cascade
        if other.id in stale:
            continue
        if any(cosine_sim(p.embedding, other.embedding) > CLAIM_CONFLICT_THRESHOLD for p in patched):
            stale.add(other.id)
            # re-run the same patch step once for `other`

    branch.answer_version += 1
    return branch
```

---

## 13. Branch lifecycle (full state machine)

```
FORK POLICY (per chunk) → suppress | patch | fork
        │ fork
        ▼
   FORKING (t0, cheap) — seed hypothesis from cluster lookup
        │ decompose sub-queries, dispatch BM25+FAISS in parallel per branch
        ▼
   ALIVE — retrieving + scoring grounding_score continuously
        │ utterance ends, or hypothesis invalidated by actual words
    ┌───┴────┐
    ▼        ▼
 PRUNED   scored against final utterance (best grounding_score + hypothesis match)
              │
    grounding_score ≥ θ ?
    ┌────┴─────┐
   yes         no (or 0 branches survived)
    │           │
    ▼           ▼
 PROMOTED    FALLBACK SEQUENTIAL (safety valve: normal wait→retrieve→decompose→synth,
 (single       never worse than the naive pipeline)
 cite-then-
 write pass,
 stream answer)
    │
    ▼
 session holds this branch's claim-graph; late constraint → PATCHED (§12), not re-forked
```

---

## 14. Telemetry & observability (G6)

Log, per request, as structured JSON (100% coverage required):
- Every fork event: timestamp, seeding cluster(s), entropy value, hypothesis.
- Every prune event: timestamp, reason (hypothesis mismatch / low grounding).
- Every promotion: timestamp, grounding_score, latency from utterance-start.
- Every claim: id, version history, supporting chunk, confidence, stale transitions.
- Every patch event: which tier flagged it (slot-diff / embedding-proximity / cascade), before/after text, before/after citation.
- Token cost per LLM call (decomposition fallback, cite-then-write, synthesis).
- Answer version lineage (v1 → v2 → ...) per session.

This is the same data needed for the "live telemetry dashboard" demo idea — render the branch timeline directly from this log (fork/prune/promote events plotted against wall-clock) for the 5-minute demo video: show 3 forks spawned mid-sentence, 2 pruned instantly, 1 promoted the moment the user stops talking, answer streaming ~100ms later because retrieval was already done.

---

## 15. Repo layout

```
theme4-streaming-rag/
├── README.md                    # setup + one-command run
├── requirements.txt
├── run_demo.py                  # single entrypoint: build index, run examples, print results
├── corpus/
│   ├── documents/                # synthetic corpus (~15-20 docs, Doc_ID §Section structure)
│   ├── loader.py                 # chunking with doc_id/section metadata
│   └── indexer.py                 # builds BM25 + FAISS indices + k-means cluster centroids (once)
├── engine/
│   ├── fork_policy.py             # §9 mechanism
│   ├── decomposer.py              # §10 mechanism
│   ├── retriever.py               # BM25 + FAISS + RRF fusion, per sub-query
│   ├── grounding.py                # §11 cite-then-write + NLI verifier
│   ├── claims.py                   # Claim/Branch dataclasses (§8) + patching logic (§12)
│   ├── session.py                  # ephemeral per-session state, claim graph storage
│   └── llm_client.py                # async, pluggable (Gemini/OpenAI-compatible), no blocking calls
├── telemetry/
│   └── logger.py                    # structured JSON event log (§14)
├── simulator/
│   └── stream_replay.py             # our own harness: replays a transcript as timestamped
│                                     # chunks (mirrors the guide's 0.0s/0.8s/1.6s/2.1s example)
├── tests/
│   ├── test_example1_multi_intent.py      # reproduces guide's Example 1
│   ├── test_example2_late_constraint.py   # reproduces guide's Example 2
│   ├── test_example3_query_suppression.py # reproduces guide's Example 3
│   └── test_grounding_zero_hallucination.py
└── docs/
    ├── architecture_brief.md        # ≤6 pages, required deliverable
    └── benchmark_report.md          # required deliverable, includes 3 ablations
```

## 16. Build order (phased, matches guide's roadmap section)

1. **Phase 1 — Foundation**: synthetic corpus, chunking, BM25+FAISS indexing, k-means clustering, structured event schemas.
2. **Phase 2 — Controller & streaming**: `simulator/stream_replay.py`, `fork_policy.py`, entropy-based wait/fork/suppress logic, latency logging.
3. **Phase 3 — Decomposition & fusion**: `decomposer.py`, parallel per-sub-query retrieval, RRF fusion/dedup.
4. **Phase 4 — Grounding & refinement**: `grounding.py` (cite-then-write + NLI verifier), `claims.py` patching logic, ephemeral session store.
5. **Phase 5 — Telemetry & packaging**: structured logging, single-command runner, benchmark report against held-out streaming test prompts, 3 required ablations (hybrid vs. dense-only retrieval; rule-based vs. model-based controller; cluster-split vs. naive-split decomposition).

## 17. Required deliverables (per official checklist)

- [ ] Reproducible repo: source, pinned dependencies, one-command run (`docker compose up` or CLI).
- [ ] System architecture brief (≤6 pages): design rationale, retrieval trigger logic, query decomposition strategy, data provenance, trade-offs, failure-mode mitigations.
- [ ] Benchmarking & evaluation report: quantitative comparison vs. baseline pipeline, ≥3 analyzed edge-case failures, ≥2 architectural ablations.
- [ ] System demonstration video (≤5 min): early retrieval triggering, multi-intent decomposition, late-detail refinement, presentation-query suppression, citation traceability, runtime telemetry.
- [ ] Telemetry & observability schema: structured logs, end-to-end latencies, retrieval trigger events, answer version updates, inference cost estimates.

Also required for the overall hackathon submission (separate from Theme 4 kit): `CollegeName_TeamName_Submission.pptx` deck (theme, gaps, architecture, demo, tech stack, impact, innovation highlights, what's next, brownie-points slide, GitHub checklist) and `LangAI3.0_AI_Disclosure.docx` (AI-tool-usage disclosure — do not reuse the "No AI tools were used" line from unrelated prior projects; disclose accurately for this build).

---

## 18. Open items for the builder to decide/confirm

- Exact synthetic corpus content (placeholder domain suggested: workshop/venue-booking + travel-reimbursement, mirroring the guide's own worked examples, to allow direct reproduction of Examples 1–3 as tests).
- Concrete threshold values (`LOW_CONF_THRESHOLD`, `STABLE_THRESHOLD`, `CLAIM_CONFLICT_THRESHOLD`, `TOPIC_SHIFT_THRESHOLD`, `OVERLAP_THRESHOLD`, `NLI_THRESHOLD`) — start with reasonable defaults, tune against the benchmarking report's ablations.
- Choice of LLM provider for synthesis (must be async-compatible, declared under `requirements`/`env` per submission rules — no secrets committed).
- Whether Docker packaging is worth the complexity vs. a clean CLI-only runner for G1.
