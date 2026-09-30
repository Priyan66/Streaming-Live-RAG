"""The shared primitives (spec §6).

One encoder, one cluster-centroid comparison and one entropy function are used
by every pipeline stage: fork-seeding, decomposition, patching and grounding.
Nothing downstream loads a second embedding model.
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# text utilities (tokenisation shared by BM25, overlap checks and slot logic)
# ---------------------------------------------------------------------------
STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been
    before being below between both but by can could did do does doing down during each few
    for from further had has have having he her here hers herself him himself his how i if
    in into is it its itself just me more most my myself no nor not now of off on once only
    or other our ours ourselves out over own same she should so some such than that the
    their theirs them themselves then there these they this those through to too under
    until up very was we were what when where which while who whom why will with would you
    your yours yourself yourselves please need want like tell know let get got okay ok um uh
    hmm actually also well so just really maybe thing things something anything one""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower().replace(",", ""))


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


_SUFFIXES = ("ations", "ation", "ments", "ment", "ings", "ing", "ers", "er", "ed", "es", "s", "e")


def stem(tok: str) -> str:
    """Tiny suffix stripper so cancel/cancelled/cancellation(s) share a lexical key."""
    if tok.isdigit() or len(tok) <= 4:
        return tok
    if tok.endswith("ies") and len(tok) > 5:
        return tok[:-3] + "y"
    for suf in _SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            tok = tok[: -len(suf)]
            if suf != "e" and tok.endswith("e") and len(tok) > 4:
                tok = tok[:-1]
            break
    if len(tok) > 4 and tok[-1] == tok[-2] and tok[-1] not in "aeiou":
        tok = tok[:-1]
    return tok


def stem_tokens(text: str) -> list[str]:
    return [stem(t) for t in content_tokens(text)]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def approx_tokens(text: str) -> int:
    """Cheap token estimate (≈4 chars/token) used for cost telemetry."""
    return max(1, math.ceil(len(text) / 4))


# ---------------------------------------------------------------------------
# encoders
# ---------------------------------------------------------------------------
class _STEncoder:
    kind = "sentence-transformers"

    def __init__(self, name: str):
        from sentence_transformers import SentenceTransformer

        self.name = name
        try:  # prefer the local HF cache: no network round-trips once models are downloaded
            self.model = SentenceTransformer(name, device="cpu", local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(name, device="cpu")
        self.dim = (getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension)()

    def encode(self, texts: list[str]) -> np.ndarray:
        v = self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                              show_progress_bar=False, convert_to_numpy=True)
        return v.astype(np.float32)


class _LSAEncoder:
    """Offline fallback: TF-IDF + truncated SVD fitted on the corpus itself.

    Used only when the sentence-transformer weights cannot be loaded (e.g. an
    air-gapped machine). Keeps the pipeline runnable; quality is lower.
    """

    kind = "lsa-fallback"

    def __init__(self, corpus_texts: list[str], dim: int = 128):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.name = "tfidf-lsa"
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        x = self.vec.fit_transform(corpus_texts)
        n_comp = max(2, min(dim, x.shape[0] - 1, x.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=0).fit(x)
        self.dim = n_comp

    def encode(self, texts: list[str]) -> np.ndarray:
        v = self.svd.transform(self.vec.transform(texts)).astype(np.float32)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return v / n


def load_encoder(name: str, corpus_texts: list[str]):
    try:
        return _STEncoder(name)
    except Exception as exc:  # pragma: no cover - exercised only offline
        log.warning("sentence-transformer %s unavailable (%s); using LSA fallback", name, exc)
        return _LSAEncoder(corpus_texts)


class Embedder:
    """``embed(text)`` primitive: cached, async, single model thread.

    All model calls go through one worker thread so the asyncio event loop never
    blocks and concurrent branches don't oversubscribe the CPU.
    """

    def __init__(self, encoder, cache_size: int = 4096):
        self.encoder = encoder
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embed")
        self._lock = threading.Lock()
        self.calls = 0
        self.cache_hits = 0

    @property
    def dim(self) -> int:
        return self.encoder.dim

    def clear_cache(self):
        with self._lock:
            self._cache.clear()

    def encode_sync(self, texts: list[str]) -> np.ndarray:
        out: list[np.ndarray | None] = []
        missing = []
        with self._lock:
            for t in texts:
                v = self._cache.get(t)
                if v is not None:
                    self._cache.move_to_end(t)
                    self.cache_hits += 1
                else:
                    missing.append(t)
                out.append(v)
        if missing:
            uniq = list(dict.fromkeys(missing))
            self.calls += 1
            vecs = self.encoder.encode(uniq)
            fresh = dict(zip(uniq, vecs))
            with self._lock:
                for t, v in fresh.items():
                    self._cache[t] = v
                    if len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)
            out = [fresh[t] if v is None else v for t, v in zip(texts, out)]
        return np.stack(out) if out else np.zeros((0, self.dim), np.float32)

    async def embed(self, text: str) -> np.ndarray:
        if text in self._cache:
            return self.encode_sync([text])[0]
        loop = asyncio.get_running_loop()
        return (await loop.run_in_executor(self._pool, self.encode_sync, [text]))[0]

    async def embed_many(self, texts: list[str]) -> np.ndarray:
        if all(t in self._cache for t in texts):
            return self.encode_sync(texts)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, self.encode_sync, texts)


# ---------------------------------------------------------------------------
# similarity / entropy
# ---------------------------------------------------------------------------
def cosine_sim(vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Cosine similarity of one (normalised) vector against rows of ``mat``."""
    if mat.ndim == 1:
        return np.array([float(vec @ mat)])
    return mat @ vec


def softmax_entropy(sims: np.ndarray, temperature: float) -> float:
    """Normalised entropy (0..1) of softmax(sims / T)."""
    if len(sims) <= 1:
        return 0.0
    z = (sims - sims.max()) / temperature
    p = np.exp(z)
    p /= p.sum()
    h = -float(np.sum(p * np.log(p + 1e-12)))
    return h / math.log(len(sims))
