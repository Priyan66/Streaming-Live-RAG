"""Engine configuration.

Every threshold lives here so ablations and tuning never touch mechanism code.
Any field can be overridden with an environment variable ``RAG_<FIELD_NAME>``
(e.g. ``RAG_LOW_CONF_THRESHOLD=0.3``). Values were tuned on the *dev* split of
the benchmark only (see docs/benchmark_report.md) and are corpus-agnostic
similarity/probability levels, not content.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class EngineConfig:
    # --- models -----------------------------------------------------------
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    nli_model: str = "cross-encoder/nli-MiniLM2-L6-H768"

    # --- fork policy (§9) -------------------------------------------------
    low_conf_threshold: float = 0.30      # max cluster sim below this -> wait
    softmax_temperature: float = 0.05     # for entropy over cluster sims
    stable_entropy: float = 0.55          # normalised entropy below -> single direction
    tie_margin: float = 0.04              # clusters within this of the max are "near-tied"
    max_branches: int = 4
    min_content_tokens: int = 3           # buffers with fewer content tokens -> wait
    confident_sim: float = 0.55           # a buffer this close to one topic is stable on its own

    # --- decomposition (§10) ----------------------------------------------
    clause_min_content_tokens: int = 2
    clause_conf_threshold: float = 0.22   # clause-to-cluster sim needed to count as an intent
    clause_merge_sim: float = 0.80        # same-cluster clauses this similar are one intent

    # --- retrieval --------------------------------------------------------
    retrieval_mode: str = "hybrid"        # hybrid | dense | sparse
    cluster_prior: bool = True            # fuse the sub-query's topic-cluster ranking into RRF
    rrf_k: int = 60
    top_n_per_retriever: int = 20
    top_k: int = 5
    dedup_sim: float = 0.97

    # --- grounding (§11) --------------------------------------------------
    overlap_threshold: float = 0.5
    nli_threshold: float = 0.5
    extract_min_relevance: float = 0.38   # extractive backend: below -> "no supporting passage"
    extract_min_focus: float = 0.20       # ...and the slot-free part of the question must match too
    branch_grounding_theta: float = 0.5   # promoted branch below this -> sequential fallback

    # --- refinement (§12) -------------------------------------------------
    topic_shift_threshold: float = 0.35   # constraint vs promoted topic; below -> new intent
    claim_conflict_threshold: float = 0.45
    presentation_margin: float = 0.05

    # --- runtime ----------------------------------------------------------
    controller: str = "entropy"           # entropy | rule | end_only (ablation)
    decomposer: str = "cluster"           # cluster | comma | none (ablation)
    speculative_synthesis: str = "local_only"   # local_only | always | never

    def with_overrides(self, **kw) -> "EngineConfig":
        return replace(self, **kw)

    @classmethod
    def from_env(cls, **kw) -> "EngineConfig":
        base = cls()
        env = {}
        for f in fields(cls):
            raw = os.environ.get(f"RAG_{f.name.upper()}")
            if raw is None:
                continue
            typ = type(getattr(base, f.name))
            env[f.name] = typ(raw)
        env.update(kw)
        return replace(base, **env)
