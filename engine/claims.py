"""Claim graph data structures (spec §8) and claim-level patching (spec §12)."""
from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from engine.grounding import UNCERTAIN_MARKER, ClaimResult, cite_then_write
from engine.primitives import content_tokens
from engine.slots import changed_slots, extract_slots, rewrite_with_slots, strip_slot_spans

if TYPE_CHECKING:
    from engine.pipeline import StreamingEngine
    from engine.retriever import RetrievalResult
    from engine.session import Session

_ids = itertools.count(1)


def new_id(prefix: str) -> str:
    return f"{prefix}{next(_ids)}"


@dataclass
class SubQuery:
    topic: str                       # merged clause text (what the user said)
    slots: dict[str, str]            # inherited *global* slots, not clause-local
    cluster: int = -1
    cluster_sim: float = 0.0
    context: str = ""                # conversation context terms (delta sub-queries of a refinement)
    id: str = field(default_factory=lambda: new_id("sq"))

    @property
    def query(self) -> str:
        """Retrieval query = the clause (+ conversation context terms for refinement deltas).

        Inherited slots travel as structured data (``self.slots``): they reach the dense query
        through the utterance-context blend, the generator prompt ("details the user gave") and
        numeric-compatibility scoring. Appending them lexically to every sub-query was measured
        to drag unrelated intents toward whichever passage names the venue/city (see
        docs/benchmark_report.md, failure F2).
        """
        low = self.topic.lower()
        extra = [t for t in self.context.split() if t.lower() not in low] if self.context else []
        return " ".join([self.topic] + list(dict.fromkeys(extra))).strip()


@dataclass
class Claim:
    id: str
    text: str
    supporting_chunk_id: str | None
    source_sub_query: str
    topic_cluster: int
    slots_used: dict[str, str]
    embedding: np.ndarray
    confidence: float
    version: int = 1
    stale: bool = False
    grounded: bool = True
    uncertainty_reason: str | None = None
    status: str = "new"              # new | preserved | reconfirmed | patched | retracted | added
    history: list[dict] = field(default_factory=list)


@dataclass
class Branch:
    id: str
    hypothesis_cluster: int
    sub_queries: list[SubQuery]
    trigger: str
    created_t: float
    claims: list[Claim] = field(default_factory=list)
    status: Literal["forking", "alive", "promoted", "pruned", "patched"] = "forking"
    answer_version: int = 1
    retrieval: "RetrievalResult | None" = None
    retrieval_query: str | None = None
    scope: list[int] | None = None
    gen: int = 0
    speculative: dict = field(default_factory=dict)

    @property
    def grounding_score(self) -> float:
        live = [c.confidence for c in self.claims if not c.stale]
        return float(np.mean(live)) if live else 0.0

    @property
    def query(self) -> str:
        return self.sub_queries[0].query if self.sub_queries else ""


@dataclass
class AnswerGraph:
    """The promoted branch's claim graph held by the session between turns."""
    branch: Branch
    anchor_text: str                 # the utterance that created the graph
    anchor_vec: np.ndarray
    slots: dict[str, str]
    chunk_cluster: dict[str, int] = field(default_factory=dict)   # chunk id -> topic cluster (from the index)

    @property
    def claims(self) -> list[Claim]:
        return self.branch.claims

    @property
    def clusters(self) -> set[int]:
        """Topics the answer is about: the sub-questions' clusters and the cited evidence's clusters."""
        evidence = {self.chunk_cluster[c.supporting_chunk_id] for c in self.claims
                    if c.supporting_chunk_id in self.chunk_cluster}
        return {c.topic_cluster for c in self.claims} | {sq.cluster for sq in self.branch.sub_queries} | evidence


def claim_from_result(res: ClaimResult, sq: SubQuery, emb: np.ndarray) -> Claim:
    return Claim(
        id=new_id("c"), text=res.text, supporting_chunk_id=res.chunk_id, source_sub_query=sq.topic,
        topic_cluster=sq.cluster, slots_used=dict(sq.slots), embedding=emb, confidence=res.confidence,
        grounded=res.grounded, uncertainty_reason=res.reason,
    )


# ---------------------------------------------------------------------------
# §12 claim-level patching
def patch_subquery(c: Claim, merged_slots: dict[str, str]) -> SubQuery:
    """Delta query for re-verifying a stale claim: its own sub-question with updated slots."""
    topic = rewrite_with_slots(c.source_sub_query, c.slots_used, merged_slots)
    return SubQuery(topic=topic, slots=merged_slots, cluster=c.topic_cluster)


def patch_scope(index, graph: "AnswerGraph") -> list[int]:
    """Re-verification searches only the claim graph's own topic clusters."""
    return index.scope_for_clusters(c for c in graph.clusters if c >= 0)


def slot_stale(graph: "AnswerGraph", new_slots: dict[str, str]) -> dict[str, str]:
    out = {}
    for c in graph.claims:
        diff = changed_slots(c.slots_used, new_slots)
        if diff:
            out[c.id] = f"slot_diff:{','.join(sorted(diff))}"
    return out


# ---------------------------------------------------------------------------
async def apply_constraint(engine: "StreamingEngine", s: "Session", constraint: str,
                           delta_subqs: list[SubQuery], turn_log: dict) -> list[Claim]:
    """Patch only the claims a late constraint invalidates, then add delta claims.

    Returns the claims that were added. Mutates ``s.graph`` in place — the session's
    claim graph is never cleared here.
    """
    idx, cfg, log = engine.index, engine.cfg, engine.logger
    graph = s.graph
    assert graph is not None
    vec = await idx.embedder.embed(constraint)
    new_slots = extract_slots(constraint, idx.gazetteer)
    merged_slots = {**graph.slots, **new_slots}

    # neighbourhood scope: the claim graph's clusters + the constraint's nearest clusters
    sims = idx.centroids @ vec
    scope_clusters = set(graph.clusters) | {int(c) for c in np.argsort(-sims)[:2]}
    scope_clusters |= {sq.cluster for sq in delta_subqs}
    scope = idx.scope_for_clusters(c for c in scope_clusters if c >= 0)

    # Tier 1 (slot diff) + Tier 2 (embedding proximity)
    stale: dict[str, str] = slot_stale(graph, new_slots)
    for c in graph.claims:
        if c.id not in stale and float(vec @ c.embedding) > cfg.claim_conflict_threshold:
            stale[c.id] = "embedding_proximity"
    for c in graph.claims:
        c.stale = c.id in stale
        if c.stale:
            log.emit(s, "claim_stale", claim_id=c.id, tier=stale[c.id].split(":")[0], detail=stale[c.id],
                     similarity=round(float(vec @ c.embedding), 4))

    pscope = patch_scope(idx, graph)

    async def reverify(c, why: str) -> Claim:
        sq = patch_subquery(c, merged_slots)
        res, retr = await engine.retrieve_and_cite(s, sq, scope=pscope, trigger=f"patch:{why}",
                                                   context_vec=graph.anchor_vec)
        before = {"text": c.text, "citation": c.supporting_chunk_id, "version": c.version}
        if res.grounded and res.chunk_id == c.supporting_chunk_id and res.text == c.text:
            c.status = "reconfirmed"
        elif res.grounded:
            c.text, c.supporting_chunk_id, c.status = res.text, res.chunk_id, "patched"
            c.embedding = await idx.embedder.embed(res.text)
        else:
            c.text, c.supporting_chunk_id, c.status = UNCERTAIN_MARKER, None, "retracted"
            c.grounded, c.uncertainty_reason = False, res.reason
        c.confidence, c.slots_used, c.stale = res.confidence, dict(merged_slots), False
        c.version += 1
        after = {"text": c.text, "citation": c.supporting_chunk_id, "version": c.version}
        c.history.append({"answer_version": graph.branch.answer_version + 1, "cause": why,
                          "before": before, "after": after})
        log.emit(s, "claim_patched", claim_id=c.id, tier=why, outcome=c.status, before=before, after=after,
                 chunks_searched=retr.chunks_searched, corpus_size=retr.corpus_size)
        return c

    # Tier 3: real re-verification, only for flagged claims
    flagged = [c for c in graph.claims if c.id in stale]
    patched = await asyncio.gather(*(reverify(c, stale[c.id].split(":")[0]) for c in flagged))
    for c in graph.claims:
        if c.id not in stale:
            c.status = "preserved"

    # one-hop cascade: a claim whose text changed may conflict with an unflagged neighbour
    changed = [c for c in patched if c.status in ("patched", "retracted")]
    cascade = [o for o in graph.claims if o.id not in stale and any(
        float(p.embedding @ o.embedding) > cfg.claim_conflict_threshold for p in changed)]
    for o in cascade:
        log.emit(s, "claim_stale", claim_id=o.id, tier="cascade", detail="one-hop from patched claim")
    await asyncio.gather(*(reverify(o, "cascade") for o in cascade))

    # delta sub-queries: what the constraint adds that the graph doesn't cover yet
    added: list[Claim] = []

    context = " ".join(dict.fromkeys(content_tokens(graph.anchor_text)))

    async def add_delta(sq: SubQuery):
        sq.slots = merged_slots
        sq.context = context
        res, _ = await engine.resolve_delta(s, sq, scope, graph.anchor_vec)
        turn_log["sub_queries"].append(sq.query)
        if res.grounded and any(c.supporting_chunk_id == res.chunk_id and c.text == res.text for c in graph.claims):
            return  # already covered by an existing claim
        emb = await idx.embedder.embed(res.text if res.grounded else sq.topic)
        c = claim_from_result(res, sq, emb)
        c.status = "added"
        added.append(c)

    await asyncio.gather(*(add_delta(sq) for sq in delta_subqs))
    graph.claims.extend(added)
    graph.slots = merged_slots
    graph.branch.status = "patched"
    return added


def informative_clause(sq: SubQuery, gazetteer=None) -> bool:
    """A delta clause worth querying must say more than its slot values
    ("expecting 60 people" is a slot update, "some guests are vegan" is new content)."""
    return len(content_tokens(strip_slot_spans(sq.topic, gazetteer))) >= 2
