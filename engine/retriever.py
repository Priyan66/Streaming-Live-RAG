"""Corpus retrieval & fusion: BM25 + FAISS + Reciprocal Rank Fusion (Phase 3).

``scope`` restricts search to a chunk subset (used by claim patching so a late
constraint never triggers a full-corpus search — gate G5).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from corpus.indexer import CorpusIndex
from corpus.loader import Chunk
from engine.config import EngineConfig
from engine.primitives import stem_tokens


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None
    sparse_rank: int | None


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit]
    chunks_searched: int
    corpus_size: int
    latency_ms: float
    mode: str

    @property
    def ids(self) -> list[str]:
        return [h.chunk.chunk_id for h in self.hits]


class HybridRetriever:
    def __init__(self, index: CorpusIndex, cfg: EngineConfig, mode: str | None = None):
        self.index = index
        self.cfg = cfg
        self.mode = mode or cfg.retrieval_mode
        self.searches = 0

    def _dense(self, qvec: np.ndarray, scope: list[int] | None, n: int) -> list[int]:
        if scope is not None:
            sims = self.index.chunk_emb[scope] @ qvec
            order = np.argsort(-sims)[:n]
            return [scope[i] for i in order]
        _, ids = self.index.faiss_index.search(qvec[None, :].astype(np.float32), min(n, len(self.index.chunks)))
        return [int(i) for i in ids[0] if i >= 0]

    def _sparse(self, query: str, scope: list[int] | None, n: int) -> list[int]:
        toks = stem_tokens(query)
        if not toks:
            return []
        scores = self.index.bm25.get_scores(toks)
        cand = np.array(scope) if scope is not None else np.arange(len(scores))
        cand = cand[scores[cand] > 0]
        order = cand[np.argsort(-scores[cand], kind="stable")]
        return [int(i) for i in order[:n]]

    async def search(self, query: str, scope: list[int] | None = None, k: int | None = None,
                     qvec: np.ndarray | None = None, cluster: int | None = None) -> RetrievalResult:
        """Hybrid search. ``cluster`` (the sub-query's topic cluster from decomposition/fork-seeding)
        adds a third ranked list to the fusion: that cluster's chunks by dense similarity."""
        t0 = time.perf_counter()
        k = k or self.cfg.top_k
        n = self.cfg.top_n_per_retriever
        if qvec is None:
            qvec = await self.index.embedder.embed(query)
        dense = self._dense(qvec, scope, n) if self.mode in ("hybrid", "dense") else []
        sparse = self._sparse(query, scope, n) if self.mode in ("hybrid", "sparse") else []

        prior: list[int] = []
        if cluster is not None and cluster >= 0 and self.cfg.cluster_prior:
            members = self.index.cluster_members[cluster]
            if scope is not None:
                members = [m for m in members if m in set(scope)]
            prior = self._dense(qvec, members, n) if members else []
        fused: dict[int, float] = {}
        for ranks in (dense, sparse, prior):
            for r, i in enumerate(ranks):
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.cfg.rrf_k + r + 1)
        order = sorted(fused, key=lambda i: -fused[i])

        # de-duplicate near-identical chunks so the reranked list stays diverse
        kept: list[int] = []
        for i in order:
            if all(float(self.index.chunk_emb[i] @ self.index.chunk_emb[j]) < self.cfg.dedup_sim for j in kept):
                kept.append(i)
            if len(kept) == k:
                break
        d_rank = {i: r for r, i in enumerate(dense)}
        s_rank = {i: r for r, i in enumerate(sparse)}
        hits = [Hit(self.index.chunks[i], fused[i], d_rank.get(i), s_rank.get(i)) for i in kept]
        self.searches += 1
        return RetrievalResult(query, hits, len(scope) if scope is not None else len(self.index.chunks),
                               len(self.index.chunks), (time.perf_counter() - t0) * 1000, self.mode)
