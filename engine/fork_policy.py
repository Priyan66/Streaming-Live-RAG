"""Retrieval controller / fork policy (spec §9).

Per transcript chunk: embed the running buffer once, compare it to the cluster
centroids once, and decide:

* ``wait``     — too little signal, or intent not yet stable (pitfall 1: no
                 retrieval on noise).
* ``suppress`` — presentation-only turn, or nothing new relative to the promoted
                 branch (pitfall 4).
* ``fork``     — start retrieval on 1 cluster (low entropy) or hedge across the
                 near-tied top clusters (high entropy), capped at ``max_branches``.

Ablations: ``controller="rule"`` fires on every chunk with enough tokens;
``controller="end_only"`` never fires early (the naive batch baseline).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from engine.config import EngineConfig
from engine.presentation import PresentationDetector
from engine.primitives import content_tokens, softmax_entropy
from engine.slots import changed_slots, extract_slots

CONTENT_SLOT_KEYS = ("headcount", "amount", "percent", "duration", "date", "location", "entity")


@dataclass
class ControllerDecision:
    action: str                      # wait | fork | suppress
    reason: str
    clusters: list[int] = field(default_factory=list)
    top_cluster: int | None = None
    max_sim: float = 0.0
    entropy: float = 1.0
    patch_mode: bool = False
    slots: dict = field(default_factory=dict)
    vec: np.ndarray | None = None

    def log_fields(self) -> dict:
        return {"action": self.action, "reason": self.reason, "clusters": self.clusters,
                "top_cluster": self.top_cluster, "max_sim": round(self.max_sim, 4),
                "entropy": round(self.entropy, 4), "patch_mode": self.patch_mode, "slots": self.slots}


async def fork_policy(buffer: str, session, index, cfg: EngineConfig,
                      presentation: PresentationDetector) -> ControllerDecision:
    toks = content_tokens(buffer)
    if cfg.controller == "end_only":
        return ControllerDecision("wait", "end_only controller (batch baseline)")
    sentence_done = buffer.rstrip().endswith(("?", ".", "!"))
    if len(toks) < cfg.min_content_tokens and not (sentence_done and len(toks) >= 2):
        return ControllerDecision("wait", f"only {len(toks)} content tokens")

    vec = await index.embedder.embed(buffer)
    sims = index.centroids @ vec
    top = int(np.argmax(sims))
    max_sim = float(sims[top])
    ent = softmax_entropy(sims, cfg.softmax_temperature)
    slots = extract_slots(buffer, index.gazetteer)
    base = dict(top_cluster=top, max_sim=max_sim, entropy=ent, slots=slots, vec=vec)

    if cfg.controller == "rule":
        return ControllerDecision("fork", "rule: enough tokens", clusters=[top], **base)

    graph = session.graph
    if graph is not None and await presentation.is_presentation(buffer, max_sim, slots):
        return ControllerDecision("suppress", "presentation_restructure", **base)

    patch_mode = False
    if graph is not None:
        rel = max(float(graph.anchor_vec @ vec), max((float(index.centroids[c] @ vec) for c in graph.clusters if c >= 0),
                                                     default=0.0))
        slot_change = sorted(changed_slots(graph.slots, slots))
        patch_mode = rel >= cfg.topic_shift_threshold or bool(slot_change)
        if slot_change:
            # a changed slot is a late constraint on the promoted answer: re-verify the dependent
            # claims now, while the user is still talking (Tier-1 patching, started early)
            return ControllerDecision("patch", f"slot change {slot_change} on promoted topic", patch_mode=True,
                                      **base)

    if max_sim < cfg.low_conf_threshold:
        return ControllerDecision("wait", "low corpus affinity", **base)

    has_content_slot = any(k in slots for k in CONTENT_SLOT_KEYS)
    # stable = a concrete slot was spoken, the same topic won twice in a row, or the words
    # continue the conversation's existing claim graph (a refinement in progress)
    stable = (has_content_slot or session.prev_top_cluster == top or patch_mode or sentence_done
              or max_sim >= cfg.confident_sim)
    session.prev_top_cluster = top
    if not stable:
        return ControllerDecision("wait", "intent unstable: awaiting a slot or a repeated topic", **base)

    if ent < cfg.stable_entropy:
        clusters = [top]
        if graph is not None and top in graph.clusters:
            new_content = {k: v for k, v in slots.items() if k in CONTENT_SLOT_KEYS and graph.slots.get(k) != v}
            if not new_content:
                return ControllerDecision("suppress", "matches promoted branch; nothing new yet",
                                          clusters=clusters, patch_mode=patch_mode, **base)
        reason = "single stable direction"
    else:
        order = np.argsort(-sims)
        clusters = [int(c) for c in order if sims[c] >= max_sim - cfg.tie_margin][: cfg.max_branches]
        reason = f"ambiguous: hedging across {len(clusters)} clusters"
    return ControllerDecision("fork", reason, clusters=clusters, patch_mode=patch_mode, **base)
