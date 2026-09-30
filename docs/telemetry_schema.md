# Telemetry & Observability Schema

Every run writes `runs/<timestamp>/events.jsonl`, one JSON object per line. The same events feed three consumers:

- `telemetry/validate.py` (trace coverage, gate G6, and the citation provenance audit, gate G4)
- `telemetry/dashboard.py` (the HTML branch timeline)
- the benchmark scorer

## Common envelope

| field | type | meaning |
|---|---|---|
| `event` | string | event type (below) |
| `session_id` | string | random per conversation; sessions share nothing |
| `turn` | int | user turn within the session (1-based) |
| `t_stream` | float s | stream time since the turn began — same axis as transcript chunk timestamps (0.0, 0.8, …) |
| `t_wall_ms` | float ms | wall-clock offset since the logger started |

## Event types

| event | key fields | emitted when |
|---|---|---|
| `session_start` | `engine`, `llm_backend`, `encoder`, `verifier`, `corpus_chunks`, `clusters`, `config` | new session |
| `turn_start` | — | a user turn begins |
| `chunk_received` | `chunk_index`, `text`, `buffer` | every transcript chunk |
| `controller_decision` | `action` (wait/fork/suppress/patch), `reason`, `clusters`, `top_cluster`, `max_sim`, `entropy`, `patch_mode`, `slots` | every chunk (1:1 with `chunk_received`) |
| `decomposition` | `partial`, `clauses`, `assignments[{clause, cluster, sim, kept, role?}]`, `holistic`, `reason`, `sub_queries` | mid-stream (partial) and at utterance end |
| `branch_forked` | `branch`, `cluster`, `cluster_label`, `trigger` (provisional/multi_intent/hedge/delta), `hypothesis`, `entropy`, `max_sim`, `slots` | a speculative retrieval branch is created |
| `branch_pruned` | `branch`, `reason`, `retrieval_done`, `llm_calls_spent` (always 0) | the hypothesis is invalidated or doesn't match the final utterance |
| `branch_promoted` | `branch`, `grounding_score`, `forked_at_s`, `lead_time_s` | a branch matches a final sub-query |
| `retrieval_started` | `query`, `trigger`, `branch`, `scope_chunks` | any search (provisional, multi_intent, hedge, refresh, final, final_refresh, delta, early_patch, patch:*, fallback_sequential, batch) |
| `retrieval_completed` | `query`, `top_ids`, `latency_ms`, `chunks_searched`, `corpus_size`, `mode` | search finished; `chunks_searched < corpus_size` proves a scoped search (G5) |
| `retrieval_reused` | `branch`, `query`, `speculative_hit` | final/delta step reused mid-stream retrieval (and claim) |
| `llm_call` | `purpose`, `backend`, `model`, `prompt_tokens`, `completion_tokens`, `latency_ms`, `est_cost_usd`, `would_be_prompt_tokens` | every generator call, including speculative ones |
| `claim_verified` | `sub_query`, `grounded`, `citation`, `selected_index`, `candidates`, `overlap`, `missing_critical`, `nli_label`, `confidence`, `reason`, `rejected_index`, `verify_ms`, `speculative` | every cite-then-write verification |
| `early_patch` | `claim_id`, `detail` | a slot change started Tier-1 re-verification mid-utterance |
| `claim_stale` | `claim_id`, `tier` (slot_diff/embedding_proximity/cascade), `detail`, `similarity` | the patching tiers flag a claim |
| `claim_patched` | `claim_id`, `tier`, `outcome` (reconfirmed/patched/retracted), `before{text,citation,version}`, `after{…}`, `chunks_searched`, `corpus_size` | Tier-3 re-verification result |
| `refinement` / `topic_shift` / `followup_question` | `relation`, `delta_sub_queries`, `action` | utterance-end routing when a claim graph exists |
| `fallback_sequential` | `sub_query`, `reason`, `previous` | a promoted branch failed grounding → plain sequential path |
| `retrieval_decision` | `retrieval_required` (false), `reason` (presentation_restructure / low corpus affinity), `style`, `n` | turns answered with no corpus query |
| `utterance_end` | `transcript`, `n_chunks` | end-of-utterance marker |
| `answer_version` | `answer_version`, `parent_version`, `cause` (initial/patch/presentation/restart), `answer`, `citations`, `uncertainty`, `claims[{id,version,status,citation}]` | every new answer version — the version lineage |
| `turn_summary` | `kind`, `answer_version`, `latency{utterance_end_s, first_retrieval_s, early_retrieval, answer_ready_s, post_utterance_ms}`, `cost{llm_calls, prompt_tokens, completion_tokens, est_usd, would_be_prompt_tokens, retrievals}`, `citations`, `sub_queries` | end of every turn |
| `session_end` | `turns`, `answer_versions` (full lineage) | session closed (state discarded) |
| `error` | `where`, `branch` | a branch task raised; the stream continues |

## Trace coverage rule (G6)

`validate_trace` counts a turn as fully traced only when **all** of the following hold:

1. There is a `turn_start`, an `utterance_end` and a `turn_summary` with `latency` and `cost`.
2. Every `chunk_received` has a `controller_decision`.
3. Retrieval completions never outnumber starts. A branch pruned mid-flight may legitimately drop its completion.
4. Every non-clarification turn has an `answer_version`.
5. Every cited ID is in the corpus index; otherwise it counts as a *fabricated citation*.
6. Every cited ID appears in a `retrieval_completed` of that session **and** in a grounded `claim_verified`; otherwise it counts as an *untraceable citation*.
7. An `answer` turn performed or reused a retrieval.

On every run in this repository, the demo examples, the dev split, the test split and all ablations, coverage is 100 % with 0 fabricated and 0 untraceable citations.

## Cost estimation

- **Hosted generator.** `llm_call.est_cost_usd` is computed from reported token usage with the per-model prices in `engine/llm_client.py:PRICE_PER_MTOK` (a default of $0.15 / $0.60 per million input/output tokens).
- **Offline extractive generator.** It spends 0 tokens, but it still logs `would_be_prompt_tokens`, the size of the closed-index prompt a hosted model would have received. On the test split that is **568 tokens per turn** for the streaming engine versus 1017 for the batch baseline, about $0.00009 per turn at the default price.
- **Retrieval and verification.** These are local CPU costs, reported as `latency_ms` and `verify_ms`.
