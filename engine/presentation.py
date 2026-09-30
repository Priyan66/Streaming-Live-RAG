"""Presentation-only turns (pitfall 4): reformat / shorten / repeat / translate.

Detection reuses the shared ``embed`` primitive: the utterance is compared to a
handful of *generic, domain-free* descriptions of output-restructuring requests
and to the corpus cluster centroids. If it is closer to "restructure your last
answer" than to anything in the corpus, and it adds no new content slots, no
retrieval happens. The prototypes describe the *kind* of request, never any
test content, so they satisfy the no-hardcoding rule.
"""
from __future__ import annotations

import re

import numpy as np

from engine.primitives import split_sentences

PROTOTYPES = [
    "repeat your last answer",
    "say that again in bullet points",
    "make your answer shorter",
    "summarize what you just told me",
    "rephrase that more simply",
    "translate your answer into another language",
    "format the previous answer as a list",
    "give me that in two sentences",
]
_CUE_RE = re.compile(
    r"\b(repeat|rephrase|reword|reformat|shorten|shorter|briefly|concise|bullets?|bullet points|"
    r"translate|simpler|say (?:that|it) again|(?:your|the) (?:last|previous) (?:answer|reply|response)|"
    r"in (?:\w+ )?(?:points|sentences|lines|bullets))\b", re.I)
_REF_RE = re.compile(r"\b(that|it|this|your|last|previous|above|again)\b", re.I)
_NUMS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "single": 1}
_CONTENT_SLOTS = ("headcount", "amount", "percent", "duration", "date", "location", "entity")


class PresentationDetector:
    def __init__(self, index, cfg):
        self.index = index
        self.cfg = cfg
        self._proto: np.ndarray | None = None

    async def _protos(self) -> np.ndarray:
        if self._proto is None:
            self._proto = await self.index.embedder.embed_many(PROTOTYPES)
        return self._proto

    async def score(self, text: str) -> float:
        vec = await self.index.embedder.embed(text)
        return float(np.max(await self._protos() @ vec))

    async def is_presentation(self, text: str, corpus_max_sim: float, slots: dict) -> bool:
        if any(k in slots for k in _CONTENT_SLOTS):
            return False
        cue = bool(_CUE_RE.search(text)) and bool(_REF_RE.search(text))
        proto = await self.score(text)
        return cue or proto >= corpus_max_sim + self.cfg.presentation_margin


def parse_style(text: str) -> tuple[str, int | None]:
    low = text.lower()
    m = re.search(r"\b(\d+|one|two|three|four|five|single)\s+(bullets?|bullet points|points|sentences?|lines?)\b", low)
    n = None
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else _NUMS[m.group(1)]
    if "translat" in low:
        return "translate", n
    if re.search(r"bullet|points|list", low):
        return "bullets", n
    if re.search(r"sentence|line", low):
        return "sentences", n
    if re.search(r"short|brief|concise|summar|simpl", low):
        return "shorter", n
    return "repeat", n


def restructure(claims: list[dict], style: str, n: int | None) -> tuple[str, list[str]]:
    """Deterministically re-render grounded claims. Never adds content or citations."""
    items = [(c["text"], c["citation"]) for c in claims if c.get("citation")]
    if style == "shorter":
        items = [(split_sentences(t)[0], cit) for t, cit in items]
    groups: list[list[tuple[str, str]]]
    if n and n < len(items):
        size = -(-len(items) // n)
        groups = [items[i:i + size] for i in range(0, len(items), size)]
    else:
        groups = [[it] for it in items]
    rendered = [" ".join(f"{t} [{cit}]" for t, cit in g) for g in groups]
    if style in ("bullets",):
        body = "\n".join(f"- {r}" for r in rendered)
    else:
        body = " ".join(rendered)
    cits = list(dict.fromkeys(cit for _, cit in items))
    return body, cits
