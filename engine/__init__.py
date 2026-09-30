"""Streaming Live RAG engine."""
from __future__ import annotations

from pathlib import Path

from engine.config import EngineConfig

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = ROOT / "corpus" / "documents"


class Runtime:
    """Shared, per-process heavy objects: index (encoder inside) + NLI verifier + LLM.

    Built once and reused by every engine variant (streaming, baseline, ablations);
    sessions themselves stay ephemeral.
    """

    def __init__(self, corpus_dir=DEFAULT_CORPUS, cfg: EngineConfig | None = None, cache_dir=ROOT / ".cache"):
        from corpus.indexer import build_index
        from engine.grounding import Verifier
        from engine.llm_client import make_llm

        self.cfg = cfg or EngineConfig.from_env()
        self.index = build_index(corpus_dir, self.cfg.embed_model, cache_dir)
        self.verifier = Verifier(self.cfg)
        self.llm = make_llm(self.index, self.cfg)

    def engine(self, logger, kind: str = "streaming", **overrides):
        from engine.baseline import BaselineEngine
        from engine.llm_client import ExtractiveBackend
        from engine.pipeline import StreamingEngine

        cfg = self.cfg.with_overrides(**overrides) if overrides else self.cfg
        llm = self.llm
        if isinstance(llm, ExtractiveBackend) and overrides:
            llm = ExtractiveBackend(self.index, cfg)
        cls = BaselineEngine if kind == "baseline" else StreamingEngine
        return cls(self.index, cfg, llm, self.verifier, logger)
