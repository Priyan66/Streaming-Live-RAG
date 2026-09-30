"""Multi-intent decomposition (spec §10) — no LLM call.

1. Rule-based syntactic segmentation into clauses (sentence punctuation and
   coordinating conjunctions / commas); fragments too small to be a question are
   merged back into a neighbour.
2. Each clause is embedded and assigned to its nearest corpus cluster (the same
   centroids the fork policy uses).
3. Clauses on the same cluster that are also semantically close are merged
   (pitfall 5: no near-identical sub-queries); clauses on distinct clusters stay
   separate — the structural signature of genuine multi-intent.
4. Slots are extracted once from the whole utterance and inherited by every
   sub-query ("Pune, 30 people" also applies to the catering clause).
5. Confidence gate: if fewer than two confident groups remain, the whole
   utterance becomes one holistic sub-query — never force a split.

Ablation modes: ``comma`` (split on every boundary, no merging, no gate) and
``none`` (always one holistic sub-query).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from engine.claims import SubQuery
from engine.config import EngineConfig
from engine.primitives import content_tokens
from engine.slots import extract_slots, strip_slot_spans

_SENT_SPLIT = re.compile(r"(?<=[.?!;])\s+")
_CLAUSE_SPLIT = re.compile(
    r"\s*,\s*(?:and\s+|but\s+|also\s+|plus\s+|then\s+)?|\s+(?:and also|as well as|and then|along with|and|plus|also)\s+",
    re.I)
_LEADING_FILLER = re.compile(r"^(?:and|but|also|plus|so|then|or)\s+", re.I)
# discourse-level retractions ("forget the trip", "never mind that") withdraw an intent, they don't ask one
RETRACTION_RE = re.compile(r"^(?:actually\s+|oh\s+|ok(?:ay)?\s+)?(?:forget|never\s?mind|ignore|scratch|drop|"
                           r"cancel that|disregard)\b", re.I)


def syntactic_split(text: str, min_content: int = 2) -> list[str]:
    frags: list[str] = []
    for sent in _SENT_SPLIT.split(text.strip()):
        for f in _CLAUSE_SPLIT.split(sent):
            f = _LEADING_FILLER.sub("", f.strip(" .,;!?\"'…")).strip()
            if f:
                frags.append(f)
    clauses: list[str] = []
    for f in frags:
        n = len(content_tokens(f))
        if n == 0 or RETRACTION_RE.match(f):
            continue                      # filler ("and I need") or a retraction carries no new intent
        if n < min_content and clauses:
            clauses[-1] = f"{clauses[-1]} and {f}"
        elif n < min_content:
            clauses.append(f)             # merged forward below
        else:
            if clauses and len(content_tokens(clauses[-1])) < min_content:
                clauses[-1] = f"{clauses[-1]} {f}"
            else:
                clauses.append(f)
    return clauses


@dataclass
class Decomposition:
    sub_queries: list[SubQuery]
    clauses: list[str]
    assignments: list[dict] = field(default_factory=list)
    holistic: bool = False
    reason: str = ""

    def log_fields(self) -> dict:
        return {"clauses": self.clauses, "assignments": self.assignments, "holistic": self.holistic,
                "reason": self.reason, "sub_queries": [sq.query for sq in self.sub_queries]}


async def decompose(utterance: str, index, cfg: EngineConfig, inherited_slots: dict | None = None,
                    mode: str | None = None) -> Decomposition:
    mode = mode or cfg.decomposer
    global_slots = {**(inherited_slots or {}), **extract_slots(utterance, index.gazetteer)}
    uvec = await index.embedder.embed(utterance)
    u_sims = index.centroids @ uvec
    u_top = int(np.argmax(u_sims))

    # corpus affinity for the abstain decision is judged against actual passages: a centroid
    # of a mixed-topic cluster can sit far from each of its members (benchmark failure F1)
    u_affinity = max(float(u_sims[u_top]), float(np.max(index.chunk_emb @ uvec)))

    def holistic(reason: str, clauses: list[str], assignments=None) -> Decomposition:
        if u_affinity < cfg.low_conf_threshold and not any(a.get("kept") for a in (assignments or [])):
            return Decomposition([], clauses, assignments or [], True, "no corpus affinity: " + reason)
        topic = " ".join(clauses) if clauses else utterance.strip()   # retractions already removed
        sq = SubQuery(topic, global_slots, u_top, float(u_sims[u_top]))
        return Decomposition([sq], clauses, assignments or [], True, reason)

    if mode == "none":
        return holistic("decomposition disabled", [utterance])

    clauses = syntactic_split(utterance, cfg.clause_min_content_tokens)
    if mode == "comma":
        subs = []
        for cl in clauses:
            v = await index.embedder.embed(cl)
            s = index.centroids @ v
            subs.append(SubQuery(cl, global_slots, int(np.argmax(s)), float(np.max(s))))
        return Decomposition(subs or holistic("empty", clauses).sub_queries, clauses, [], False, "naive split")

    if len(clauses) < 2:
        return holistic("single clause", clauses)

    vecs = await index.embedder.embed_many(clauses)
    sims = vecs @ index.centroids.T
    assignments, kept = [], []
    for i, cl in enumerate(clauses):
        c = int(np.argmax(sims[i]))
        conf = float(sims[i, c])
        # a clause that only carries slot values ("for a trip to London") is shared context, not an intent
        informative = len(content_tokens(strip_slot_spans(cl, index.gazetteer))) >= cfg.clause_min_content_tokens
        ok = conf >= cfg.clause_conf_threshold and informative
        assignments.append({"clause": cl, "cluster": c, "sim": round(conf, 4), "kept": ok,
                            **({} if informative else {"role": "context"})})
        if ok:
            kept.append((i, c, conf))

    # group: same cluster AND semantically close -> one intent
    groups: list[dict] = []
    for i, c, conf in kept:
        for g in groups:
            if g["cluster"] == c and max(float(vecs[i] @ vecs[j]) for j in g["idx"]) >= cfg.clause_merge_sim:
                g["idx"].append(i)
                g["conf"] = max(g["conf"], conf)
                break
        else:
            groups.append({"cluster": c, "idx": [i], "conf": conf})

    if len(groups) < 2:
        return holistic("fewer than two confident intents", clauses, assignments)

    subs = [SubQuery(" and ".join(clauses[i] for i in g["idx"]), global_slots, g["cluster"], g["conf"])
            for g in groups]
    return Decomposition(subs, clauses, assignments, False, f"{len(subs)} intents on distinct topics")
