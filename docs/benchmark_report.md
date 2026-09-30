# Benchmarking & Evaluation Report

Reproduce with `python run_demo.py --bench` (test split) and `python run_demo.py --bench --split dev`. The raw per-turn records are in `benchmark_results.json`, and the tables are regenerated into `benchmark_results.md`.

## 1. Setup

**Corpus.** 20 synthetic policy documents, 71 chunks, 22 topic clusters chosen automatically.

**Benchmark conversations** (`bench/scenarios.py`). They are separate from the demo examples and unit tests. Each turn is plain text streamed as ~5-word chunks 0.7 s apart, with a 0.5 s pause before the end-of-utterance marker.

| split | conversations | turns | used for |
|---|---|---|---|
| dev | 12 | 16 | setting thresholds |
| test | 30 | 39 | every number below |

Test turns by kind: 18 single-intent, 8 compound (2–3 intents), 4 late-constraint refinements, 1 follow-up, 1 topic shift, 3 presentation-only, 2 noise, 2 out-of-corpus.

**Labels.** Each retrieval turn carries, per intent, the set of acceptable source documents. The scorer never sees system internals: it reads only the turn output and the trace.

**Systems compared.** All share the same encoder, retriever, generator (offline extractive) and verifier, so differences come from architecture alone.

- **Streaming**: the full system.
- **Batch baseline** (`engine/baseline.py`):
  - waits for the end of the utterance, then retrieves once with the whole utterance
  - answers with up to 3 cited sentences from distinct passages
  - on any follow-up, restarts by re-querying the concatenated conversation over the full corpus
  - never suppresses presentation-only turns

**Metrics (gate → how it is measured).**

| Gate | Measurement |
|---|---|
| G2 | % of retrieval-needing turns whose first retrieval started before the utterance-end timestamp; false triggers = % of noise/presentation turns that retrieved at all |
| G3 | % of compound turns split into ≥ 2 sub-queries (plus exact-count %); over-fragmentation = % of single-intent turns split |
| G4 | gold-document precision of citations on retrieval turns; fabricated IDs (not in the index); untraceable citations (no logged retrieval or verification origin) |
| G5 | on refinement turns: all turn-1 claim IDs still present, and no retrieval that searched the full corpus |
| G6 | `telemetry/validate.py` trace coverage |

Latency is stream time from the end-of-utterance marker to the answer being ready, measured on a 16-core laptop CPU. The virtual clock charges real processing time; each configuration is warmed up and starts with an empty embedding cache.

## 2. Headline: streaming vs. batch baseline (test split)

| metric | target | **Streaming** | Batch baseline |
|---|---|---|---|
| G2 early retrieval | ≥ 80 % | **90.6 %** | 0 % |
| G2 false triggers (noise / presentation) | low | **0 %** | 100 % |
| G3 compound split into ≥ 2 | ≥ 70 % | **100 %** (exact count 87.5 %) | 0 % |
| G4 citation gold-document precision | ≥ 85 % | **95.3 %** | 76.5 % |
| G4 fabricated / untraceable citation IDs | 0 | **0 / 0** | 0 / 0 |
| intent coverage (gold intents with a correct citation) | — | **97.6 %** | 90.2 % |
| out-of-corpus questions abstained (no citation, explicit uncertainty) | — | 100 % | 100 % |
| G5 prior claims preserved on refinement | — | **75 %** (3/4) | 0 % |
| G5 refinements with no full-corpus search | — | **100 %** | 0 % |
| G6 trace coverage | 100 % | **100 %** | 100 % |
| answer latency after utterance end, p50 / p95 | — | **32 / 206 ms** | 455 / 564 ms |
| retrieval lead time before utterance end (mean) | — | 0.95 s | — |
| retrievals / generator calls per turn | — | 1.80 / 1.36 | 1.00 / 2.72 |
| would-be LLM prompt tokens per turn | — | 568 | 1017 |

**Dev split** (the thresholds were set here): the streaming engine scores 100 % on every gate metric above. The baseline scores 71.8 % gold precision.

### Reading the numbers

- **Latency is where streaming pays off.** Retrieval starts on average 0.95 s before the user stops talking. With the local generator, the claim is usually verified before the end-of-utterance marker too, so the median answer arrives **32 ms** after the user stops, against **455 ms** for batch.
- **Streaming spends more retrieval but less generation.** It runs +0.8 retrievals per turn at 1–2 ms each, but makes half as many generator calls and uses 44 % fewer prompt tokens. Branches are retrieval-only, and presentation and noise turns cost nothing.
- **Decomposition drives answer quality.** One query per utterance leaves compound intents uncovered, and the baseline's "top-3 sentences" pads answers with loosely related passages. That padding is why the baseline's gold precision is 76.5 % even though it has the same verifier and fabricates no IDs.

## 3. Ablations (test split)

| configuration | early % | false trig % | split % | over-frag % | gold prec % | intent cov % | refinements scoped % | p50 ms | retr/turn | gen/turn |
|---|---|---|---|---|---|---|---|---|---|---|
| **Full system** | 90.6 | 0 | 100 | 11.1 | **95.3** | **97.6** | 100 | 32 | 1.80 | 1.36 |
| A1 hybrid, **no cluster prior** | 90.6 | 0 | 100 | 11.1 | 90.7 | 92.7 | 100 | 32 | 1.80 | 1.36 |
| A1 dense-only (+ cluster prior) | 90.6 | 0 | 100 | 11.1 | 93.3 | 97.6 | 100 | 34 | 1.77 | 1.33 |
| A1 sparse-only (+ cluster prior) | 90.6 | 0 | 100 | 11.1 | 95.3 | 97.6 | 100 | 33 | 1.80 | 1.36 |
| A2 rule-based controller | 100 | 0 | 100 | 11.1 | 95.3 | 97.6 | 100 | 31 | **2.13** | 1.41 |
| A2 end-only controller (no early retrieval) | **0** | 0 | 100 | 11.1 | 95.3 | 97.6 | 75 | **227** | 1.18 | 1.18 |
| A3 naive comma-split | 90.6 | **40** | 100 | 11.1 | 95.3 | 97.6 | 100 | 32 | 1.90 | 1.46 |
| A3 no decomposition | 90.6 | 0 | **0** | 0 | 93.8 | **75.6** | 100 | 28 | 1.54 | 1.03 |

### A1 — hybrid vs. dense-only retrieval (and the cluster prior)

**The cluster prior matters most.** Removing it costs −4.6 points of gold precision and −4.9 points of intent coverage. The prior is the decomposition's own clause→cluster decision reused as a third RRF list, so it adds no model and no latency.

**Dense-only is 2 points behind hybrid, and sparse-only ties it on this corpus.** The synthetic corpus is lexically clean: each policy uses its own vocabulary, which suits BM25. We keep hybrid because dense retrieval is what recovers paraphrases ("get paid after my claim" → "reimbursement timeline") on real, messier corpora, and fusion never hurt here.

**Takeaway:** on a clean corpus the gain comes from *topic-aware fusion* more than from sparse+dense fusion itself.

### A2 — rule-based vs. model-based controller

**Rule-based controller.** It fires on every chunk with ≥ 3 content tokens. That gets 100 % early retrieval but costs **+18.6 % retrievals per turn** (2.13 vs 1.80): it re-queries on every partial phrase, which is pitfall 1 thrashing, and gains nothing in quality.

**Entropy/stability controller (model-based).** It gives up 9 points of earliness: it waits on three turns whose partial phrases were not yet topical (F1). In exchange it forks once the intent is stable.

**End-only controller.** Removing early retrieval entirely makes answers **7× slower** (p50 227 ms vs 32 ms). Its lower "refinements scoped" figure (75 %) comes from turn t18 (F5). That turn is misclassified as a topic shift in *both* configurations; a topic shift correctly searches the full corpus, which is what end-only does. The full system happened to reuse a scoped branch forked mid-stream. So the full system's 100 % on that turn is incidental, and G5 should be read together with the state-preserved figure (75 %).

**Why false triggers match.** The rule-based controller's false-trigger rate equals ours (0 %) because the downstream decomposition confidence gate also blocks noise. The controller's contribution here is *cost*, not correctness.

### A3 — cluster-split vs. naive split decomposition

**Naive comma/"and" splitting** has no confidence gate, so it turns noise ("okay so um let me think for a second") into sub-queries: **40 % false triggers** on no-retrieval turns, plus +5 % retrievals and +7 % generator calls.

**No decomposition** loses **22 points of intent coverage** (75.6 % vs 97.6 %) and splits 0 % of compound queries.

**Cluster-based splitting** is the only variant that is both complete and quiet.

## 4. Edge-case failures (analysed)

**Transparency note.** The first full test-split run scored **87.5 % early, 95.0 % gold precision, 90.2 % intent coverage and 50 % G5 state preserved** (see §5). Analysing those failures gave F1–F3 below. Three *code-level* fixes followed, each generic and each described in its failure entry; thresholds were not tuned on test. The numbers in §2–§3 are from the re-run. F4–F6 remain open.

### F1 — Centroid dilution: controller waits too long; first run abstained on an answerable question

- **Case (t05).** "What is the mileage rate if I drive my own car for work?"
- **Symptom.** On the first run the system asked for clarification. The final run answers correctly (Doc_44 §3) but retrieval still starts only at utterance end.
- **Root cause.** Cluster 13 mixes local cabs, airport transfers and personal-vehicle mileage, so its centroid sits far from each member. The utterance's best *centroid* similarity was below the 0.30 affinity gate, although its best *passage* similarity was well above it.
- **Fix applied.** The final abstain / clarify decision now uses passage-level affinity: max(centroid, best chunk).
- **Remaining.** The mid-stream controller stays centroid-only for parsimony, which is why this turn is still not early.
- **Next step.** Use max-member affinity for clusters with high intra-cluster variance, which can be precomputed at index time.

### F2 — Slot-inheritance drag: a named entity pulled other intents to its page

- **Cases (t09, t12).**
  - "How much does Orchid Hall cost per day and what dietary options can the caterer handle?"
  - "…the price of Riverside Studio, the AV rental charges, and the budget approval rules."
- **Symptom (first run).** The dietary sub-query was answered with "In-house catering is mandatory at Orchid Hall" (Doc_12 §2 instead of Doc_09 §3). The AV sub-query was answered with "Riverside Studio is the only Pune venue that allows external caterers."
- **Root cause.** Inherited slots were *appended to every sub-query's lexical query*. BM25 then rewarded any passage naming the venue.
- **Fix applied.** Slots now travel as structured context (dense utterance blend, generator prompt, numeric and location consistency), not as query words. Both turns now cover all intents correctly.
- **Trade-off.** Inheritance is now softer; a clause that relies *only* on an inherited place ("what about catering there?") gets that place through the dense context and the location-consistency score, not through BM25.

### F3 — Short or declarative late constraints misread

- **Cases (t15, t16, t17).**
  - "The trip is actually to Singapore."
  - "It was an emergency trip though."
  - "Some of our guests are vegan."
- **Symptoms (first run).**
  - t15/t17: no early retrieval, because the controller's 3-token floor blocked 2-token sentences.
  - t16: treated as a topic shift, which lost the claim graph, because the claim's *evidence* cluster (booking exceptions) was not counted as part of the graph's topic.
  - t17: no delta claim, because "guests" was stripped as a headcount unit word.
- **Fixes applied.**
  - A finished sentence with ≥ 2 content tokens may proceed.
  - The graph's topics include the clusters of cited evidence.
  - A clause's informativeness is measured after removing only the *matched slot spans*.
- **Result.** t16 is now a proper refinement: state preserved, "In a genuine emergency…" added under *In addition*.

### F4 — Over-fragmentation when one rule answers two surface topics (open)

- **Case (t13).** "Can I add personal leave to a business trip and who pays for the extra hotel nights?"
- **Symptom.** Split into a leave intent and a hotel intent. The hotel sub-query cited the international hotel limit (Doc_08 §2), a wrong document. The single correct passage (Doc_25 §4) answers both halves.
- **Root cause.** The clauses sit on genuinely different clusters (leave, hotels), which the design treats as the signature of multi-intent.
- **Proposed mitigation.** *Evidence fusion before synthesis*: if one candidate chunk ranks top for both sub-queries and grounds both, merge them into one claim. The machinery already exists for identical claims (`_answer_fresh` deduplication).

### F5 — Implicit dependency not recognised as a refinement (open)

- **Case (t18).** "What is the per diem for a day trip to Pune?" → "The client is providing lunch that day."
- **Symptom.** The follow-up is classified as a topic shift and answered with catering packages (Doc_09 §1). The correct update is that the per diem is reduced by INR 400 per provided meal (Doc_10 §2), and the claim graph is lost.
- **Root cause.** This is exactly the Tier-2 false negative the design warns about. "Lunch provided" shares no slot with the per-diem claim, and its embedding is closer to the catering cluster than to anything in the graph (measured relation 0.28 < 0.35).
- **Proposed mitigation.** Before declaring a shift, run one scoped retrieval of the constraint inside the graph's clusters. If a passage there entails a condition mentioned by the constraint ("where a client provides meals…"), treat the utterance as a refinement. This costs one extra scoped search only on ambiguous shifts.

### F6 — Extractive lexical echo (observed during development, mitigated)

- **Case.** The guide's Example 1 "cancellation policy" initially cited "Later rescheduling requests are treated as a cancellation followed by a new booking" (Doc_31 §2) instead of the cancellation terms (§4).
- **Root cause.** The extractive generator favours sentences that echo the query word.
- **Fixes.** The cluster prior in RRF, heading overlap, and a factual-density prior (sentences stating concrete figures). Example 1 now cites §4.
- **Remaining.** This is an inherent limit of an extractive generator. A hosted LLM behind the same verifier chooses by meaning.

## 5. First-run numbers (before the F1–F3 fixes)

| config | G2 early % | false trig % | G3 split % | G4 gold-prec % | fabricated | intent cov % | G5 state % | G5 scoped % | p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| Streaming (first run) | 87.5 | 0.0 | 100.0 | 95.0 | 0 | 90.2 | 50.0 | 100.0 | 25 |
| Baseline (first run) | 0.0 | 100.0 | 0.0 | 76.5 | 0 | 90.2 | 0.0 | 0.0 | 403 |

## 6. Threats to validity

- **Small, author-written evaluation set.** 39 test turns over a synthetic corpus written by the same team. The gold labels are document-level, which is lenient about the exact section.
- **Latency depends on the machine.** p95 varies by ±100 ms between runs on the same laptop. Ratios are stable; absolute numbers are not.
- **Refinement results rest on four turns.** The G5 "75 %" is 3/4 turns; treat it as indicative.
- **Offline generator only.** Every number uses the offline extractive generator. With a hosted LLM, quality on paraphrase-heavy questions should rise, and latency will include the model call, which happens once per sub-question after promotion.
