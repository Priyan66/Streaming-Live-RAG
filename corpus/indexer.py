"""Builds every index structure once per corpus (spec §6, Phase 1).

* BM25 (sparse) and FAISS inner-product (dense) indices over chunks.
* k-means topic clusters over chunk embeddings + normalised centroids — the
  single comparison target reused by fork-seeding, decomposition and patching.
* Per-sentence embeddings (for extractive cite-then-write).
* A corpus-derived gazetteer of proper nouns (for slot extraction).

Embeddings are cached on disk keyed by corpus content + model name. This is
index caching, not answer precomputation: nothing here depends on any query.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from corpus.loader import Chunk, load_corpus
from engine.primitives import Embedder, load_encoder, stem_tokens

log = logging.getLogger(__name__)

_PROPER_RE = re.compile(r"\b[A-Z][a-zA-Z0-9'-]+(?:\s+[A-Z][a-zA-Z0-9'-]+)*")
_LOC_PREP_RE = r"\b(?:in|at|to|from|near)\s+(?:the\s+)?{}\b"


@dataclass
class CorpusIndex:
    chunks: list[Chunk]
    embedder: Embedder
    chunk_emb: np.ndarray
    bm25: BM25Okapi
    faiss_index: faiss.Index
    labels: np.ndarray
    centroids: np.ndarray
    cluster_terms: list[list[str]]
    cluster_members: list[list[int]]
    sent_chunk: np.ndarray          # sentence -> chunk idx
    sent_text: list[str]
    sent_emb: np.ndarray
    gazetteer: dict[str, tuple[str, str]]
    build_seconds: float = 0.0
    valid_ids: set[str] = field(default_factory=set)

    @property
    def n_clusters(self) -> int:
        return len(self.centroids)

    def cluster_label(self, c: int) -> str:
        return " ".join(self.cluster_terms[c])

    def chunk_by_id(self, chunk_id: str) -> Chunk | None:
        for ch in self.chunks:
            if ch.chunk_id == chunk_id:
                return ch
        return None

    def sentences_of(self, chunk_idx: int) -> np.ndarray:
        return np.nonzero(self.sent_chunk == chunk_idx)[0]

    def scope_for_clusters(self, clusters) -> list[int]:
        out: list[int] = []
        for c in sorted(set(clusters)):
            out.extend(self.cluster_members[c])
        return sorted(out)


def _choose_k(emb: np.ndarray) -> int:
    n = len(emb)
    if n < 6:
        return max(1, n // 2)
    lo, hi = max(2, n // 8), max(3, min(n - 1, n // 3))
    best_k, best_s = lo, -1.0
    for k in range(lo, hi + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(emb)
        s = silhouette_score(emb, labels, metric="cosine")
        if s > best_s + 1e-4:
            best_k, best_s = k, s
    return best_k


def _cluster_terms(chunks: list[Chunk], labels: np.ndarray, k: int, n_terms: int = 4) -> list[list[str]]:
    docs = [" ".join(f"{c.heading} {c.heading} {c.text}" for c in chunks if labels[c.idx] == j) for j in range(k)]
    vec = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]+\b", sublinear_tf=True)
    x = vec.fit_transform(docs).toarray()
    vocab = np.array(vec.get_feature_names_out())
    return [list(vocab[np.argsort(-row)[:n_terms]]) for row in x]


def _gazetteer(chunks: list[Chunk]) -> dict[str, tuple[str, str]]:
    body = " ".join(c.text for c in chunks)
    lowered_body = " " + " ".join(re.findall(r"(?<![A-Za-z])[a-z][a-z0-9'-]*", body)) + " "  # lowercase-written words
    mid: set[str] = set()          # phrases seen capitalised mid-sentence
    initial: dict[str, str] = {}
    for c in chunks:
        for sent in c.sentences:
            for m in _PROPER_RE.finditer(sent):
                words = m.group(0).split()
                if m.start() == 0:  # sentence-initial capital is weak evidence: drop common words
                    while words and f" {words[0].lower()} " in lowered_body:
                        words = words[1:]
                    phrase = " ".join(words)
                    if phrase:
                        initial.setdefault(phrase.lower(), phrase)
                    continue
                mid.add(" ".join(words).lower())
                initial.setdefault(" ".join(words).lower(), " ".join(words))
    gaz = {}
    for low, surface in initial.items():
        if low not in mid or surface.isupper() or len(surface) < 3 or f" {low} " in lowered_body:
            continue
        kind = "location" if re.search(_LOC_PREP_RE.format(re.escape(surface)), body) else "entity"
        gaz[low] = (surface, kind)
    return gaz


def build_index(corpus_dir: str | Path, model_name: str, cache_dir: str | Path = ".cache",
                embedder: Embedder | None = None) -> CorpusIndex:
    t0 = time.perf_counter()
    chunks = load_corpus(corpus_dir)
    sent_chunk, sent_text = [], []
    for c in chunks:
        for s in c.sentences:
            sent_chunk.append(c.idx)
            sent_text.append(s)

    if embedder is None:
        embedder = Embedder(load_encoder(model_name, [c.embed_text for c in chunks]))
    enc_name = getattr(embedder.encoder, "name", model_name)
    h = hashlib.sha256(enc_name.encode())
    for c in chunks:
        h.update(c.chunk_id.encode() + b"\0" + c.embed_text.encode())
    cache = Path(cache_dir) / f"emb_{h.hexdigest()[:16]}.npz"
    if cache.exists():
        z = np.load(cache)
        chunk_emb, sent_emb = z["chunk"], z["sent"]
    else:
        chunk_emb = embedder.encoder.encode([c.embed_text for c in chunks])
        sent_emb = embedder.encoder.encode(sent_text)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, chunk=chunk_emb, sent=sent_emb)

    fx = faiss.IndexFlatIP(chunk_emb.shape[1])
    fx.add(chunk_emb)
    bm25 = BM25Okapi([stem_tokens(c.embed_text) or ["_"] for c in chunks])

    k = _choose_k(chunk_emb)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(chunk_emb)
    labels = km.labels_
    centroids = km.cluster_centers_.astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    members = [[int(i) for i in np.nonzero(labels == j)[0]] for j in range(k)]

    idx = CorpusIndex(
        chunks=chunks, embedder=embedder, chunk_emb=chunk_emb, bm25=bm25, faiss_index=fx,
        labels=labels, centroids=centroids, cluster_terms=_cluster_terms(chunks, labels, k),
        cluster_members=members, sent_chunk=np.array(sent_chunk), sent_text=sent_text,
        sent_emb=sent_emb, gazetteer=_gazetteer(chunks),
        valid_ids={c.chunk_id for c in chunks},
    )
    idx.build_seconds = time.perf_counter() - t0
    log.info("index: %d chunks, %d clusters, %d gazetteer terms, %.2fs", len(chunks), k,
             len(idx.gazetteer), idx.build_seconds)
    return idx
