"""Pluggable async LLM clients. Nothing here blocks the event loop.

* ``ExtractiveBackend`` (default): deterministic, offline, zero-token. It answers
  the same closed-index JSON contract as an LLM by selecting the best-supported
  sentence(s) from the numbered candidates, so the whole replay runs unattended
  with no API key (gate G1) and every claim is a verbatim corpus span.
* ``OpenAICompatBackend``: any OpenAI-compatible chat endpoint (OpenAI, Gemini's
  OpenAI-compatible endpoint, Ollama, vLLM ...), selected via environment
  variables. Uses ``httpx.AsyncClient``.

Both return ``LLMResponse`` with token usage for cost telemetry.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

from corpus.loader import Chunk
from engine.config import EngineConfig
from engine.primitives import approx_tokens, content_tokens, stem_tokens

# USD per 1M tokens (input, output); used only for cost *estimates* in telemetry.
PRICE_PER_MTOK = {"default": (0.15, 0.60)}


@dataclass
class CiteContext:
    sub_query: str
    slots: dict[str, str]
    candidates: dict[int, Chunk]
    qvec: object = None          # optional context-blended query vector (numpy)


@dataclass
class LLMResponse:
    data: dict
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    backend: str
    model: str
    est_cost_usd: float = 0.0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
_CAP_RE = re.compile(
    r"(?:up to|groups of|for|accommodates|seats)\s+(?:about\s+)?(\d+)(?:\s+to\s+(\d+))?\s+(?:people|persons|guests|"
    r"participants|attendees)", re.I)


_NUM_TOKEN = re.compile(r"\b\d[\d,.]*\b")


def numeric_compat(sentence: str, slots: dict[str, str]) -> float:
    """Small score adjustment when a sentence's stated capacity contradicts the
    requested headcount (generic numeric-slot check, no domain terms)."""
    if "headcount" not in slots:
        return 0.0
    try:
        want = max(int(v) for v in slots["headcount"].split(", "))
    except ValueError:
        return 0.0
    caps = _CAP_RE.findall(sentence)
    if not caps:
        return -0.04        # the user gave a group size; passages that state none answer it less directly
    ok = any((int(lo) <= want <= int(hi)) if hi else want <= int(lo) for lo, hi in caps)
    return 0.10 if ok else -0.12


def location_compat(passage: str, slots: dict[str, str], gazetteer: dict) -> float:
    """Slot-consistency check: a passage about a *different* place than the one asked about is
    a worse answer ("which Mumbai venue ..." should not be answered from the Pune venue list)."""
    if "location" not in slots:
        return 0.0
    low = passage.lower()
    wanted = [v.lower() for v in slots["location"].split(", ")]
    if any(re.search(r"\b" + re.escape(w) + r"\b", low) for w in wanted):
        return 0.03
    others = [p for p, (_, kind) in gazetteer.items() if kind == "location" and p not in wanted]
    return -0.08 if any(re.search(r"\b" + re.escape(o) + r"\b", low) for o in others) else 0.0


def focus_text(query: str, slots: dict[str, str]) -> str:
    out = query
    for vals in slots.values():
        for v in vals.split(", "):
            out = re.sub(r"\b" + re.escape(v) + r"\b", " ", out, flags=re.I)
    out = re.sub(r"\b(?:people|persons|attendees|guests|participants|pax|inr|usd|rs)\b", " ", out, flags=re.I)
    return " ".join(out.split()) if len(content_tokens(out)) >= 1 else ""


class ExtractiveBackend:
    name = "extractive"
    model = "extractive-v1"
    is_local = True

    def __init__(self, index, cfg: EngineConfig):
        self.index = index
        self.cfg = cfg

    async def generate_json(self, prompt: str, ctx: CiteContext | None = None, task: str = "cite") -> LLMResponse:
        t0 = time.perf_counter()
        if task != "cite" or ctx is None:
            return LLMResponse({"unsupported": True}, 0, 0, 0.0, self.name, self.model)
        # sentence selection scores against the sub-question itself; the conversation-context blend
        # (ctx.qvec) already did its job at retrieval time and would otherwise favour sentences that
        # merely echo the rest of the utterance
        qvec = await self.index.embedder.embed(ctx.sub_query)
        q_toks = {t for t in stem_tokens(ctx.sub_query) if not t.isdigit()}   # numbers: numeric_compat
        # "focus" = what is asked once slot values are removed; guards against a sentence that only
        # matches the place/number ("weather in Pune" must not be answered by "venues in Pune").
        focus = focus_text(ctx.sub_query, ctx.slots)
        fvec = await self.index.embedder.embed(focus) if focus else None
        best: list[tuple[float, int, int, str]] = []  # (score, cand_no, sent_pos, text)
        for no, chunk in ctx.candidates.items():
            sids = self.index.sentences_of(chunk.idx)
            sims = self.index.sent_emb[sids] @ qvec
            fsims = self.index.sent_emb[sids] @ fvec if fvec is not None else sims
            chunk_sim = float(self.index.chunk_emb[chunk.idx] @ qvec)   # passage-level context (heading, title)
            chunk_fit = numeric_compat(chunk.text, ctx.slots)
            chunk_fit = chunk_fit if chunk_fit < -0.05 else 0.0   # a passage whose stated capacities all miss
            head_lex = len(q_toks & set(stem_tokens(chunk.heading))) / max(1, len(q_toks))
            loc_fit = location_compat(f"{chunk.doc_title} {chunk.heading} {chunk.text}", ctx.slots,
                                      self.index.gazetteer)
            for pos, (sid, sim, fsim) in enumerate(zip(sids, sims, fsims)):
                text = self.index.sent_text[sid]
                lex = len(q_toks & set(stem_tokens(text))) / max(1, len(q_toks))
                density = 0.03 * min(2, len(_NUM_TOKEN.findall(text)))   # factual density: concrete figures
                score = (0.65 * float(sim) + 0.35 * chunk_sim + 0.15 * lex + 0.1 * head_lex + density
                         + min(numeric_compat(text, ctx.slots), chunk_fit or 1.0) + loc_fit - 0.015 * (no - 1))
                if float(fsim) < self.cfg.extract_min_focus:
                    score = min(score, self.cfg.extract_min_relevance - 1e-3)
                best.append((score, no, pos, text))
        best.sort(key=lambda b: -b[0])
        if not best or best[0][0] < self.cfg.extract_min_relevance:
            data = {"selected_index": None,
                    "uncertainty_reason": "no retrieved passage addresses this sub-question"}
        else:
            top = best[0]
            picked = [top]
            for b in best[1:3]:  # optionally a second supporting sentence from the same passage
                if b[1] == top[1] and b[0] >= 0.92 * top[0] and b[0] >= self.cfg.extract_min_relevance:
                    picked.append(b)
                    break
            picked.sort(key=lambda b: b[2])
            data = {"selected_index": top[1], "claim_text": " ".join(b[3] for b in picked),
                    "relevance": round(top[0], 4)}
        return LLMResponse(data, 0, 0, (time.perf_counter() - t0) * 1000, self.name, self.model,
                           extra={"would_be_prompt_tokens": approx_tokens(prompt)})


# ---------------------------------------------------------------------------
class OpenAICompatBackend:
    """Async client for any OpenAI-compatible ``/chat/completions`` endpoint.

    Env: ``LLM_BASE_URL`` (default OpenAI), ``LLM_API_KEY``, ``LLM_MODEL``.
    For Gemini use ``LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai``.
    """

    name = "openai-compatible"
    is_local = False

    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None,
                 timeout: float = 30.0):
        import httpx

        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.is_local = self.base_url.startswith(("http://localhost", "http://127.0.0.1"))
        self._client = httpx.AsyncClient(timeout=timeout)

    async def generate_json(self, prompt: str, ctx: CiteContext | None = None, task: str = "cite") -> LLMResponse:
        t0 = time.perf_counter()
        body = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "You answer strictly from the numbered passages provided. "
                                              "Reply with a single JSON object and nothing else."},
                {"role": "user", "content": prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = await self._client.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
        r.raise_for_status()
        payload = r.json()
        text = payload["choices"][0]["message"]["content"] or "{}"
        m = re.search(r"\{.*\}", text, re.S)
        try:
            data = json.loads(m.group(0) if m else text)
        except json.JSONDecodeError:
            data = {"selected_index": None, "uncertainty_reason": "model returned malformed JSON"}
        usage = payload.get("usage") or {}
        pt = usage.get("prompt_tokens", approx_tokens(prompt))
        ct = usage.get("completion_tokens", approx_tokens(text))
        pin, pout = PRICE_PER_MTOK.get(self.model, PRICE_PER_MTOK["default"])
        return LLMResponse(data, pt, ct, (time.perf_counter() - t0) * 1000, self.name, self.model,
                           est_cost_usd=(pt * pin + ct * pout) / 1e6)

    async def aclose(self):
        await self._client.aclose()


def make_llm(index, cfg: EngineConfig):
    backend = os.environ.get("LLM_BACKEND", "extractive").lower()
    if backend in ("openai", "openai-compatible", "gemini", "ollama"):
        return OpenAICompatBackend()
    return ExtractiveBackend(index, cfg)


def est_cost(prompt_tokens: int, completion_tokens: int, model: str = "default") -> float:
    pin, pout = PRICE_PER_MTOK.get(model, PRICE_PER_MTOK["default"])
    return (prompt_tokens * pin + completion_tokens * pout) / 1e6


__all__ = ["CiteContext", "LLMResponse", "ExtractiveBackend", "OpenAICompatBackend", "make_llm",
           "numeric_compat", "est_cost"]
