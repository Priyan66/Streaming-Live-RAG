# System Architecture Brief — Streaming Live RAG

*Theme 4, Samsung PRISM GenAI Hackathon 3rd Edition.*

## 1. Design goal and central idea

Batch RAG waits for the user to stop talking, then retrieves, then answers. This engine does the retrieval **while the user is still speaking**, so that when the utterance ends only one cheap grounding step remains.

It achieves this with **three shared primitives** that are computed once and reused by every stage, rather than one model per stage:

| Primitive | Implementation | Used by |
|---|---|---|
| `embed(text)` | `all-MiniLM-L6-v2` (22 M params), cached, single model thread | fork policy, decomposition, retrieval, claim patching, presentation detection |
| `cosine(vec, cluster_centroids)` | k-means over chunk embeddings; k chosen by silhouette at index time (22 clusters for 71 chunks) | fork seeding, clause→intent mapping, retrieval prior, patch scope, topic-shift gate |
| `extract_slots(text)` | regex (headcount, money, %, dates, durations) plus a gazetteer of proper nouns **derived from the corpus** | early stability signal, slot inheritance, Tier-1 patching, numeric/location consistency |
| `entails(premise, claim)` | `nli-MiniLM2-L6-H768` cross-encoder (82 M), micro-batched | initial grounding and every patch re-verification |

There is no vector-database service, no separate intent model and no timing classifier. The whole system is two small CPU models with no network services.

## 2. Pipeline

```
chunk t=0.0 ─► [1] Fork policy ── wait ──────────────────────────────────────────┐
chunk t=0.8 ─►     embed(buffer) · centroids · entropy · slots                    │
chunk t=1.6 ─►     ├─ fork: 1 branch (stable) or ≤4 (near-tied clusters)          │
                   ├─ suppress: presentation-only / nothing new vs. promoted      │
                   └─ patch: slot changed on the promoted answer → early re-verify│
                        │                                                          │
                  [2] Partial decomposition (clauses → clusters) ─► branches       │
                        │   each branch: async BM25 + FAISS + cluster prior → RRF  │
                        │   (+ speculative cite-then-write when the generator is local)
utterance end ─► [3] Presentation? ── yes ─► restructure prior claims, 0 retrieval
                     Refinement?   ── yes ─► [5] claim patching (scoped search)
                     else          ────────► final decomposition; promote branches that
                                             match a final sub-query, prune the rest;
                                             [4] cite-then-write per sub-query (parallel);
                                             ungrounded → fallback sequential (full corpus)
                                             ─► answer + citations + uncertainty + trace
```

### 2.1 Retrieval trigger logic (fork policy, `engine/fork_policy.py`)

For every chunk, the running buffer is embedded **once** and compared with every cluster centroid. The controller then applies these rules in order:

1. **Too little signal:** fewer than 3 content tokens (2 if the sentence has ended) → *wait*. This prevents firing on "I need to…".
2. **Presentation-only turn:** the session already holds an answer, the utterance adds no content slot, and it is closer to generic "restructure your last answer" prototypes than to any corpus topic → *suppress* (pitfall 4).
3. **Slot change on the promoted answer:** for example "…now 12 people" → *patch*. Tier-1 re-verification starts mid-utterance.
4. **Low corpus affinity:** max centroid similarity below 0.30 → *wait* (pitfall 1).
5. **Stability:** a fork needs a concrete slot, the same top cluster on two consecutive chunks, a finished sentence, a confident match (≥ 0.55), or continuity with the existing claim graph. On the guide's Example 1 this means the controller waits at 0.0 s ("…customer workshop in") and forks at 0.8 s once *Pune* and *30 people* arrive, matching the guide's timeline.
6. **Entropy of softmax(sims / 0.05):**
   - Below 0.55: one direction → fork one branch. If that direction is already the promoted answer and nothing new was said → *suppress*.
   - Otherwise: clusters within 0.04 of the best are near-tied → fork one *hedge* branch per cluster, capped at 4. Each hedge is seeded with the cluster's c-TF-IDF label plus the slots heard so far, never generated text.

Forks are cheap: retrieval takes 1–2 ms per branch and no LLM call is made. Branches whose cluster stops matching the growing utterance are **pruned** immediately.

### 2.2 Query decomposition strategy (`engine/decomposer.py`)

1. **Syntactic segmentation.** Split on sentence punctuation, commas and coordinators (*and, as well as, plus…*). Pure-filler fragments ("and I need") and discourse retractions ("forget the trip") are dropped. Fragments too small to be a question are merged into a neighbour.
2. **Clause → cluster mapping.** Each clause is embedded and assigned to its nearest centroid.
3. **Intent test.** A clause becomes an intent only if (a) its similarity is at least 0.22 and (b) it still says something once its slot spans are removed. So "for a trip to London" is treated as *shared context*, not as an intent.
4. **Grouping.** Clauses on the same cluster **and** with similar meaning (cosine ≥ 0.8) merge into one sub-query (pitfall 5). Clauses on distinct clusters stay separate: that difference is the structural signature of multi-intent.
5. **Confidence gate.** With fewer than two confident groups, the whole utterance becomes one holistic sub-query; a split is never forced. When no passage is close either, the system asks for clarification instead of retrieving.
6. **Slot inheritance without drag.** Slots are extracted once from the whole utterance and attached to every sub-query as *structured* context. They reach the dense query through an utterance-context blend (weight 0.3), and they reach the generator prompt and the numeric/location consistency scoring. They are **not** appended to each sub-query's lexical query: the benchmark showed that appending a named venue to every clause pulled unrelated intents to that venue's page (report §4, F2).
7. **Non-destructive growth.** Decomposition re-runs on every chunk. Existing branches are kept, refreshed only if their clause text grew, and new clauses fork new branches. On Example 1 the cancellation and catering branches fork at 1.6 s, and the venue branch forked at 0.8 s is kept.

### 2.3 Retrieval and fusion (`engine/retriever.py`)

Each sub-query produces **three rankings**:

- BM25 over light-stemmed tokens (`cancel`, `cancelled` and `cancellations` share a key)
- FAISS inner product over normalised MiniLM vectors
- a **cluster prior**: that sub-query's own topic cluster, ranked by dense similarity

The three are fused with Reciprocal Rank Fusion (k = 60), and near-duplicate chunks (cosine ≥ 0.97) are removed. The cluster prior is the decomposition's clause→cluster decision reused as retrieval evidence. It is what disambiguates "cancellation policy" (venue bookings) from hotel no-show cancellations in Example 1. `scope=` restricts the search to a chunk subset; claim patching relies on this.

### 2.4 Grounding: cite-then-write (`engine/grounding.py`)

1. **Closed-set citation.** Passages are shown to the generator as `[1]..[k]`, never as Doc IDs, and a test asserts that no prompt contains `Doc_`. The generator must return `{"selected_index", "claim_text"}` in that order, so the evidence choice is fixed before the prose is written. It can also return `{"selected_index": null, "uncertainty_reason"}`. An index outside `1..k`, or one that is not an integer, is **rejected, never repaired**. A fabricated `Doc_999` is therefore impossible by construction.
2. **Independent verification.** A claim is grounded only if both checks pass:
   - **Keyword/entity overlap:** at least 50 % of stemmed content tokens appear in the chunk, and *every* number and proper noun in the claim appears there.
   - **NLI entailment ≥ 0.5:** the premise is the chunk sentence that best covers the claim, plus the 2-sentence window around it. Small SNLI/MNLI cross-encoders label a verbatim sentence "neutral" when the premise is a long multi-sentence passage (we measured 0.02 entailment), so the premise must be a contiguous span *inside the cited chunk*.

   Failing either check turns the claim into an explicit uncertainty. Tests show both checks catching a wrong number, a contradiction and a misattributed but true fact.
3. **Generators.** The default `ExtractiveBackend` is deterministic, offline and zero-token. It honours the same JSON contract by picking verbatim sentences, scored by:
   - sub-question similarity, passage similarity, lexical and heading overlap
   - factual density
   - numeric compatibility ("groups of 25 to 150" fits 30 people; "up to 20" does not)
   - location consistency

   It returns `null` when nothing is relevant: the question's slot-free *focus* must match too, so "weather in Pune" is not answered from the Pune venue list. `OpenAICompatBackend` (httpx, async) plugs in any hosted model — OpenAI, Gemini's OpenAI endpoint, Ollama, vLLM — and faces the same verifier.
4. **Speculative synthesis.** When the generator is local, cite-then-write runs as soon as a clause is complete, so the answer is usually ready before the utterance ends. With a paid generator this is off by default: the single generator call per sub-question happens only after promotion.

### 2.5 Session refinement: claim-level patching (`engine/claims.py`)

The session keeps the promoted **claim graph**: claims with text, citation, sub-question, topic cluster, slots used, cached embedding, NLI confidence, version and history. When a new utterance arrives with a graph in place, it is handled in this order:

0. **Gate.** Presentation → restructure only. Topic shift → discard the graph and re-fork. A shift requires low relation to the graph (anchor, clusters, claims, max < 0.35) **and** a top cluster outside the graph's topic and evidence clusters, or an explicit retraction ("forget…", "never mind…"). A slot change always counts as a refinement.
1. **Tier 1 — slot diff.** Claims whose `slots_used` conflict with a new slot are marked stale. This starts mid-utterance as `early_patch`.
2. **Tier 2 — embedding proximity.** Claims whose own embedding is close to the constraint (> 0.45) are marked stale, even with no word overlap. This tier leans towards over-flagging.
3. **Tier 3 — re-verification.** Each stale claim's sub-question is rewritten with the new slot values ("30 people" → "12 people") and re-run through cite-then-write, **searching only the graph's clusters**. The outcome is *reconfirmed*, *patched* (text and citation replaced, version+1) or *retracted* (a visible uncertainty, never a silent deletion).
4. **One-hop cascade.** A patched claim's new text is compared with the unflagged claims; conflicts are re-verified once and never recursed.
5. **Delta sub-queries.** Clauses of the constraint that say something new ("the booking was made after travel") are decomposed, given the conversation's content terms as context, retrieved within the graph's neighbourhood (graph clusters plus the constraint's top-2) and added as new claims.

The answer is re-rendered as *Still applies / Updated / In addition*, with `answer_version` and parent version. On the test split **every** refinement searched a scoped subset and never the full corpus.

### 2.6 Branch lifecycle

```
forking → alive → promoted | pruned
```

- **forking → alive:** a branch retrieves speculatively as soon as it is forked.
- **alive → pruned:** the hypothesis was invalidated by later words, or there is no final sub-query on its cluster.
- **alive → promoted:** a final sub-query lands on the branch's cluster. Its retrieval (and speculative claim) is reused.
- **Fallback:** a promoted branch whose claim fails verification falls back to the plain sequential path (full-corpus retrieval with the final sub-question text). The worst case therefore matches the naive pipeline.

## 3. Data provenance

- **Corpus.** 20 synthetic policy documents for a fictional company: venue booking, cancellation, catering, travel reimbursement with an international addendum and booking exceptions, per diem, hotels, leave, IT, and meeting-room displays. They were written for this project so the guide's three worked examples can be reproduced, and they contain no third-party or real-company data.
- **Chunk IDs.** Every chunk is `Doc_ID §Section`, parsed from `## §N` headings. The loader also accepts plain `.txt` (paragraphs become §1..n) and `.jsonl`, so a held-out corpus can be dropped in unchanged.
- **Models.** Both models are public Hugging Face checkpoints, downloaded once and cached. The container bakes them in and runs with `HF_HUB_OFFLINE=1`.
- **Answer text.** With the default generator every sentence of an answer is a verbatim corpus span. With a hosted generator, every claim is independently entailment-checked against its cited chunk.
- **Session data.** Memory is per-session and in-process only. The only artefact written is the telemetry trace.

## 4. Trade-offs

| Decision | Benefit | Cost / risk |
|---|---|---|
| Cluster centroids as the single comparison target | one cheap lookup answers four questions; corpus-grounded seeds | a centroid of a mixed-topic cluster can sit far from its members, so the controller waits too long (F1). The final abstain decision therefore uses passage-level affinity |
| Retrieval-only forks, ≤ 4 branches | no LLM cost per hypothesis; 1–2 ms each | +0.3 retrievals per turn vs. waiting (1.8 vs 1.2 in ablation A2) |
| Extractive default generator | offline, zero-token, deterministic, verbatim claims | can prefer a lexically echoing sentence; no paraphrase or translation (translation requests abstain explicitly) |
| Strict dual verification | 0 fabricated IDs; misattributions caught | occasionally drops a correct paraphrase; ~70 ms per NLI pair on CPU (mitigated by 2 targeted premises and micro-batching) |
| Scoped patching | no full-corpus search on refinement (G5) | a constraint whose answer lives outside the graph's neighbourhood can be missed |
| Heuristic thresholds (dev split) | transparent, corpus-agnostic similarity levels | need re-tuning for a very different embedding model or corpus density |

## 5. Failure-mode mitigations (the guide's pitfalls)

1. **Eager retrieval on noise:** the token floor, the stability rule, affinity and entropy gates, and the decomposition confidence gate. Result: 0 % false triggers on noise and presentation turns. Naive comma-splitting reaches 40 %.
2. **Context loss on late constraints:** claim-level patching with version lineage. Prior claims are *preserved* or *reconfirmed*, never cleared.
3. **Citation hallucination:** closed-set indices plus overlap and NLI verification. 0 fabricated and 0 untraceable citations across every run, checked by `telemetry/validate.py`.
4. **Presentation-only turns:** suppressed mid-stream and answered by restructuring prior claims, with 0 retrievals, 0 generator calls and the same citations.
5. **Over-fragmented sub-queries:** same-cluster clauses with similar meaning are merged, context-only clauses are not treated as intents, and an ungated split never happens. Over-fragmentation is 11 % on single-intent test turns (see F4).

## 6. Observability

Every chunk decision, fork, prune, promotion, retrieval (query, trigger, scope, latency), generator call (tokens, latency, estimated cost), claim verification (overlap, NLI label and score), stale flag (tier), patch (before/after) and answer version (parent, cause) is emitted as one JSON line. `telemetry/validate.py` proves 100 % trace coverage and citation provenance. `telemetry/dashboard.py` renders the branch timeline used in the demo video. The schema is in [telemetry_schema.md](telemetry_schema.md).
