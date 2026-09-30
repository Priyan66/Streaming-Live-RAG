"""Cite-then-write grounding (spec §11).

1. Closed-set citation: the generator sees passages as throwaway indices [1..k],
   never real Doc IDs, so it cannot fabricate one. Real IDs are resolved here.
2. Selection before prose: JSON contract ``{"selected_index", "claim_text"}``.
3. Independent deterministic verification: keyword/entity overlap AND a local
   NLI cross-encoder must both pass, otherwise the claim becomes an explicit
   uncertainty. The verifier never trusts the generator's own citation.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from corpus.loader import Chunk
from engine.config import EngineConfig
from engine.llm_client import CiteContext, LLMResponse
from engine.primitives import content_tokens, stem_tokens, tokenize

log = logging.getLogger(__name__)

UNCERTAIN_MARKER = "[unverified]"


# ---------------------------------------------------------------------------
# verifier primitive: entails(premise, hypothesis)
# ---------------------------------------------------------------------------
@dataclass
class NLIResult:
    label: str
    entail: float
    contradict: float


class Verifier:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nli")
        self.kind = "cross-encoder"
        self.calls = 0
        self.pairs = 0
        self._pending: list = []
        self._flush_task = None
        try:
            import torch

            torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
        except Exception:  # pragma: no cover
            pass
        try:
            from sentence_transformers import CrossEncoder

            try:
                self.model = CrossEncoder(cfg.nli_model, device="cpu", local_files_only=True)
            except Exception:
                self.model = CrossEncoder(cfg.nli_model, device="cpu")
            id2label = self.model.model.config.id2label
            self.labels = [id2label[i].lower() for i in range(len(id2label))]
        except Exception as exc:  # pragma: no cover - offline only
            log.warning("NLI model unavailable (%s); using lexical-containment verifier", exc)
            self.model = None
            self.kind = "lexical-fallback"

    def _nli_sync(self, pairs: list[tuple[str, str]]) -> list[NLIResult]:
        self.calls += 1
        self.pairs += len(pairs)
        if self.model is None:
            out = []
            for prem, hyp in pairs:
                h = set(content_tokens(hyp))
                cov = len(h & set(content_tokens(prem))) / max(1, len(h))
                out.append(NLIResult("entailment" if cov >= 0.9 else "neutral", cov, 0.0))
            return out
        probs = self.model.predict(pairs, apply_softmax=True, show_progress_bar=False)
        out = []
        for p in probs:
            d = dict(zip(self.labels, map(float, p)))
            label = max(d, key=d.get)
            out.append(NLIResult(label, d.get("entailment", 0.0), d.get("contradiction", 0.0)))
        return out

    async def _run(self, pairs: list[tuple[str, str]]) -> list[NLIResult]:
        """Micro-batching: verifications requested in the same event-loop tick share one forward pass."""
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._pending.append((pairs, fut))
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.ensure_future(self._flush())
        return await fut

    async def _flush(self):
        while self._pending:
            await asyncio.sleep(0)                    # let sibling coroutines enqueue first
            batch, self._pending = self._pending, []
            flat = [p for pairs, _ in batch for p in pairs]
            try:
                res = await asyncio.get_running_loop().run_in_executor(self._pool, self._nli_sync, flat)
            except Exception as exc:  # pragma: no cover
                for _, fut in batch:
                    fut.set_exception(exc)
                continue
            i = 0
            for pairs, fut in batch:
                fut.set_result(res[i:i + len(pairs)])
                i += len(pairs)

    async def entails(self, premise: str, hypothesis: str) -> NLIResult:
        return (await self._run([(premise, hypothesis)]))[0]

    async def entails_best_window(self, chunk: Chunk, hypothesis: str) -> NLIResult:
        """Premise = the chunk sentence that best covers the claim, and the 2-sentence window
        around it; the best entailment wins.

        Small SNLI/MNLI cross-encoders call a hypothesis "neutral" when the premise is a long
        multi-sentence passage, even one containing the hypothesis verbatim, so the premise is
        a contiguous span *inside the cited chunk* chosen by lexical coverage of the claim. The
        check stays strict: whatever the claim asserts must be entailed by that span.
        """
        s = chunk.sentences or [chunk.text]
        h = set(content_tokens(hypothesis))
        cov = [len(h & set(content_tokens(x))) for x in s]
        b = max(range(len(s)), key=lambda i: cov[i])
        premises = [s[b]]
        if len(s) > 1:
            nb = b + 1 if b + 1 < len(s) and (b == 0 or cov[b + 1] >= cov[b - 1]) else b - 1
            lo, hi = min(b, nb), max(b, nb)
            premises.append(" ".join(s[lo:hi + 1]))
        res = await self._run([(p, hypothesis) for p in premises])
        return max(res, key=lambda r: r.entail)


_CRIT_RE = re.compile(r"\b(?:\d[\d,.]*|[A-Z][a-zA-Z]{2,})\b")


def keyword_overlap(claim: str, chunk_text: str) -> tuple[float, list[str]]:
    """(fraction of claim content tokens found in chunk, critical tokens missing).

    Critical tokens = numbers and capitalised names in the claim. Any missing
    critical token is disqualifying: a number or name that isn't in the cited
    passage is exactly what a misattributed citation looks like.
    """
    chunk_toks = set(tokenize(chunk_text))
    chunk_stems = set(stem_tokens(chunk_text))
    claim_stems = stem_tokens(claim)
    ratio = sum(t in chunk_stems for t in claim_stems) / max(1, len(claim_stems))
    first = claim.split()[0] if claim.split() else ""
    missing = []
    for m in _CRIT_RE.finditer(claim):
        tok = m.group(0).rstrip(".,").replace(",", "")
        if m.start() == 0 and tok == first.rstrip(".,"):
            continue  # sentence-initial capital is not a name
        if tok.lower() not in chunk_toks and tok not in chunk_text:
            missing.append(tok)
    return ratio, missing


# ---------------------------------------------------------------------------
@dataclass
class ClaimResult:
    grounded: bool
    text: str
    chunk_id: str | None
    confidence: float
    reason: str | None = None
    overlap: float = 0.0
    missing_critical: list[str] = field(default_factory=list)
    nli_label: str | None = None
    selected_index: int | None = None
    candidate_ids: list[str] = field(default_factory=list)
    llm: LLMResponse | None = None
    verify_ms: float = 0.0
    rejected_index: object = None


def build_closed_index_prompt(sub_query: str, slots: dict[str, str], indexed: dict[int, Chunk]) -> str:
    lines = [f"Question: {sub_query}"]
    if slots:
        lines.append("Details the user gave: " + "; ".join(f"{k}={v}" for k, v in slots.items()))
    lines.append("Passages:")
    for i, c in indexed.items():
        lines.append(f"[{i}] {c.heading}: {c.text}")
    lines.append(
        'Return JSON with keys in exactly this order: {"selected_index": <number of the ONE passage that '
        'answers the question>, "claim_text": "<one or two sentences stating only what that passage says>"}. '
        'If no passage answers it, return {"selected_index": null, "uncertainty_reason": "<short reason>"}. '
        "Do not mention passage numbers or document names inside claim_text."
    )
    return "\n".join(lines)


async def cite_then_write(sub_query: str, slots: dict[str, str], candidates: list[Chunk], llm,
                          verifier: Verifier, cfg: EngineConfig, qvec=None) -> ClaimResult:
    indexed = {i + 1: c for i, c in enumerate(candidates)}
    cand_ids = [c.chunk_id for c in candidates]
    if not indexed:
        return ClaimResult(False, UNCERTAIN_MARKER, None, 0.0, "no candidate passages retrieved",
                           candidate_ids=cand_ids)
    prompt = build_closed_index_prompt(sub_query, slots, indexed)
    resp: LLMResponse = await llm.generate_json(prompt, CiteContext(sub_query, slots, indexed, qvec), task="cite")
    sel = resp.data.get("selected_index")
    if isinstance(sel, str) and sel.strip().strip("[]").isdigit():
        sel = int(sel.strip().strip("[]"))
    if sel is None:
        return ClaimResult(False, UNCERTAIN_MARKER, None, 0.0,
                           resp.data.get("uncertainty_reason") or "no supporting passage selected",
                           candidate_ids=cand_ids, llm=resp)
    if not isinstance(sel, int) or sel not in indexed:
        # structurally impossible to turn into a Doc ID: rejected, never "repaired"
        return ClaimResult(False, UNCERTAIN_MARKER, None, 0.0, f"generator selected invalid index {sel!r}",
                           candidate_ids=cand_ids, llm=resp, rejected_index=sel)

    chunk = indexed[sel]
    claim = str(resp.data.get("claim_text") or "").strip()
    if not claim:
        return ClaimResult(False, UNCERTAIN_MARKER, None, 0.0, "empty claim text", selected_index=sel,
                           candidate_ids=cand_ids, llm=resp)
    t0 = time.perf_counter()
    ratio, missing = keyword_overlap(claim, chunk.text)
    nli = await verifier.entails_best_window(chunk, claim)
    verify_ms = (time.perf_counter() - t0) * 1000
    overlap_ok = ratio >= cfg.overlap_threshold and not missing
    entail_ok = nli.label == "entailment" and nli.entail >= cfg.nli_threshold
    grounded = overlap_ok and entail_ok
    reason = None
    if not grounded:
        why = []
        if not overlap_ok:
            why.append(f"overlap {ratio:.2f}" + (f", missing {missing}" if missing else ""))
        if not entail_ok:
            why.append(f"NLI {nli.label} ({nli.entail:.2f})")
        reason = "verification failed: " + "; ".join(why)
    return ClaimResult(grounded, claim if grounded else UNCERTAIN_MARKER, chunk.chunk_id if grounded else None,
                       nli.entail, reason, ratio, missing, nli.label, sel, cand_ids, resp, verify_ms)
