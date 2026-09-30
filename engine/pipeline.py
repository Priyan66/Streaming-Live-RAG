"""Streaming engine: the event-driven orchestrator (spec §7, §13).

Per chunk   -> fork policy -> (fork) partial decomposition -> async branch retrieval
               (+ speculative cite-then-write when the backend is local and cheap)
Utterance end -> presentation? -> refinement (claim patching)? -> fresh answer:
               final decomposition, promote matching branches, prune the rest,
               cite-then-write per sub-query, fallback-sequential safety valve.
"""
from __future__ import annotations

import asyncio
import logging
import re

import numpy as np

from engine.claims import (AnswerGraph, Branch, Claim, SubQuery, apply_constraint, claim_from_result,
                           informative_clause, new_id, patch_scope, patch_subquery, slot_stale)
from engine.config import EngineConfig
from engine.decomposer import RETRACTION_RE, Decomposition, decompose
from engine.fork_policy import ControllerDecision, fork_policy
from engine.grounding import ClaimResult, Verifier, cite_then_write
from engine.presentation import PresentationDetector, parse_style, restructure
from engine.retriever import HybridRetriever, RetrievalResult
from engine.session import Session, StreamClock
from engine.primitives import content_tokens
from engine.slots import changed_slots, extract_slots

log = logging.getLogger(__name__)

_QUESTION_RE = re.compile(r"\?\s*$|^\s*(what|how|which|when|where|who|why|can|could|is|are|does|do|will|"
                          r"should|may|tell me|what about|and what)\b", re.I)
CONTEXT_WEIGHT = 0.3
CLARIFY_TEXT = ("I couldn't match that to anything in the knowledge base. "
                "Could you rephrase it or add a detail such as the topic, place or policy you mean?")


def _clean_chunk(text: str) -> str:
    return text.strip().strip("\"'“”").strip("…").strip().strip("…").strip()


class StreamingEngine:
    name = "streaming"

    def __init__(self, index, cfg: EngineConfig, llm, verifier: Verifier, logger):
        self.index = index
        self.cfg = cfg
        self.llm = llm
        self.verifier = verifier
        self.logger = logger
        self.retriever = HybridRetriever(index, cfg)
        self.presentation = PresentationDetector(index, cfg)

    # ------------------------------------------------------------------ API
    def new_session(self, clock: StreamClock | None = None) -> Session:
        s = Session(clock or StreamClock())
        self.logger.emit(s, "session_start", engine=self.name, llm_backend=self.llm.name,
                         encoder=getattr(self.index.embedder.encoder, "kind", "?"), verifier=self.verifier.kind,
                         corpus_chunks=len(self.index.chunks), clusters=self.index.n_clusters,
                         config={"controller": self.cfg.controller, "decomposer": self.cfg.decomposer,
                                 "retrieval_mode": self.cfg.retrieval_mode})
        return s

    def begin_turn(self, s: Session):
        s.start_turn()
        self.logger.emit(s, "turn_start")

    async def on_chunk(self, s: Session, text: str):
        text = _clean_chunk(text)
        s.buffer = f"{s.buffer} {text}".strip()
        s.n_chunks += 1
        self.logger.emit(s, "chunk_received", chunk_index=s.n_chunks, text=text, buffer=s.buffer)
        dec = await fork_policy(s.buffer, s, self.index, self.cfg, self.presentation)
        self.logger.emit(s, "controller_decision", chunk_index=s.n_chunks, **dec.log_fields())
        if dec.action == "suppress":
            s.suppressed = dec.reason
        if dec.action == "patch":
            self._early_patch(s, dec)
            if dec.max_sim >= self.cfg.low_conf_threshold:
                await self._sync_branches(s, dec)
        if dec.action == "fork":
            await self._sync_branches(s, dec)

    def _early_patch(self, s: Session, dec: ControllerDecision):
        g = s.graph
        merged = {**g.slots, **dec.slots}
        stale = slot_stale(g, dec.slots)
        scope = patch_scope(self.index, g)
        for c in g.claims:
            key = (c.id, tuple(sorted(merged.items())))
            if c.id in stale and key not in s.early_patched:
                s.early_patched.add(key)
                self.logger.emit(s, "early_patch", claim_id=c.id, detail=stale[c.id])
                s.spawn(self.retrieve_and_cite(s, patch_subquery(c, merged), scope=scope, trigger="early_patch",
                                               context_vec=g.anchor_vec))

    async def on_utterance_end(self, s: Session) -> dict:
        t_end = s.clock.now()
        text = s.buffer.strip()
        self.logger.emit(s, "utterance_end", transcript=text, n_chunks=s.n_chunks)
        await s.quiesce()  # in-flight branch work (real-time mode) legitimately gates the answer
        if not text:
            return await self._clarify(s, text, t_end, "empty utterance")

        g = s.graph
        if g is None:
            return await self._answer_fresh(s, text, t_end)

        vec = await self.index.embedder.embed(text)
        max_sim = float(np.max(self.index.centroids @ vec))
        slots = extract_slots(text, self.index.gazetteer)
        if await self.presentation.is_presentation(text, max_sim, slots):
            return await self._present(s, text, t_end)

        rel = self._relation(g, vec)
        slot_update = sorted(changed_slots(g.slots, slots))
        u_top = int(np.argmax(self.index.centroids @ vec))
        d = await decompose(text, self.index, self.cfg, inherited_slots=g.slots)
        retracts = bool(RETRACTION_RE.match(text.strip()))
        if (rel < self.cfg.topic_shift_threshold or retracts) and not slot_update and (u_top not in g.clusters or retracts):
            self.logger.emit(s, "topic_shift", relation=rel, threshold=self.cfg.topic_shift_threshold,
                             action="discard claim graph; re-enter fork policy")
            s.graph = None
            return await self._answer_fresh(s, text, t_end)
        delta = [sq for sq in d.sub_queries
                 if sq.cluster_sim >= self.cfg.clause_conf_threshold and informative_clause(sq, self.index.gazetteer)]
        new_topic = [sq for sq in delta if sq.cluster not in g.clusters]
        if _QUESTION_RE.search(text) and new_topic and len(new_topic) == len(delta):
            self.logger.emit(s, "followup_question", relation=rel, inherited_slots=g.slots,
                             action="new answer graph, session slots and context carried over")
            return await self._answer_fresh(s, text, t_end, inherited=g.slots, context_vec=g.anchor_vec, d=d)
        return await self._refine(s, text, t_end, delta, rel)

    # ------------------------------------------------------- mid-stream work
    async def _sync_branches(self, s: Session, dec: ControllerDecision):
        g = s.graph if dec.patch_mode else None
        d = await decompose(s.buffer, self.index, self.cfg, inherited_slots=g.slots if g else None)
        self.logger.emit(s, "decomposition", partial=True, **d.log_fields())
        targets: dict[int, tuple[SubQuery, str]] = {}
        multi = len(d.sub_queries) > 1
        anchor_ctx = " ".join(dict.fromkeys(content_tokens(g.anchor_text))) if g is not None else ""
        for sq in d.sub_queries:
            if g is not None and not informative_clause(sq, self.index.gazetteer):
                continue
            sq.context = anchor_ctx
            targets[sq.cluster] = (sq, ("delta" if g else "multi_intent" if multi else "provisional"))
        hedging = dec.entropy >= self.cfg.stable_entropy and len(dec.clusters) > 1
        if hedging and g is None:
            for c in dec.clusters:
                if c not in targets and len(targets) < self.cfg.max_branches:
                    sq = SubQuery(self.index.cluster_label(c), dec.slots, c, float(self.index.centroids[c] @ dec.vec))
                    targets[c] = (sq, "hedge")
        scope = None
        if s.graph is not None:
            # mid-conversation: speculative branches search the claim graph's topics plus their own
            # hypothesis clusters, never the whole corpus; a genuine topic shift still gets a
            # full-corpus fallback at utterance end if this evidence doesn't ground
            scope = self.index.scope_for_clusters(c for c in set(s.graph.clusters) | set(targets) if c >= 0)

        for c, (sq, trig) in targets.items():
            br = s.branches.get(c)
            if br is None:
                br = Branch(new_id("b"), c, [sq], trig, s.clock.now(), scope=scope)
                s.branches[c] = br
                s.all_branches.append(br)
                self.logger.emit(s, "branch_forked", branch=br.id, cluster=c,
                                 cluster_label=self.index.cluster_label(c), trigger=trig, hypothesis=sq.query,
                                 entropy=dec.entropy, max_sim=dec.max_sim, slots=sq.slots)
                br.status = "alive"
                self._launch(s, br)
            elif br.retrieval_query != sq.query and br.sub_queries[0].query != sq.query:
                br.sub_queries = [sq]
                if br.trigger == "hedge" and trig != "hedge":
                    br.trigger = trig
                self._launch(s, br)
        for c in [c for c in s.branches if c not in targets]:
            self._prune(s, s.branches.pop(c), "hypothesis invalidated by later words")

    def _launch(self, s: Session, br: Branch):
        br.gen += 1
        s.spawn(self._branch_task(s, br, br.gen))

    async def _branch_task(self, s: Session, br: Branch, gen: int):
        try:
            sq = br.sub_queries[0]
            trigger = br.trigger if gen == 1 else "refresh"
            ctx = s.graph.anchor_vec if (br.scope is not None and s.graph is not None) else s.buffer
            qvec = await self._qvec(sq.query, ctx)
            self._note_retrieval(s, sq.query, trigger, br.id, br.scope)
            res = await self.retriever.search(sq.query, scope=br.scope, qvec=qvec, cluster=sq.cluster)
            if br.gen != gen or br.status == "pruned":
                return
            br.retrieval, br.retrieval_query = res, sq.query
            self._log_retrieval(s, res, trigger, br.id)
            if self._should_speculate(s, sq):
                cr = await self._cite(s, sq, res, qvec, speculative=True)
                if br.gen == gen:
                    br.speculative[(sq.query, tuple(res.ids))] = cr
        except Exception:  # never let one branch kill the stream
            log.exception("branch task failed")
            self.logger.emit(s, "error", where="branch_task", branch=br.id)

    def _should_speculate(self, s: Session, sq: SubQuery) -> bool:
        mode = self.cfg.speculative_synthesis
        if mode == "never" or (mode == "local_only" and not self.llm.is_local):
            return False
        buf = s.buffer.rstrip()
        return buf.endswith((".", "?", "!")) or not buf.lower().endswith(sq.topic.lower()[-12:])

    def _prune(self, s: Session, br: Branch, reason: str):
        br.status = "pruned"
        br.gen += 1
        self.logger.emit(s, "branch_pruned", branch=br.id, cluster=br.hypothesis_cluster, trigger=br.trigger,
                         reason=reason, retrieval_done=br.retrieval is not None, llm_calls_spent=0)

    # ------------------------------------------------ retrieval + grounding
    async def _qvec(self, query: str, context: str | np.ndarray | None) -> np.ndarray:
        q = await self.index.embedder.embed(query)
        if context is None or (isinstance(context, str) and (not context or context == query)):
            return q
        c = await self.index.embedder.embed(context) if isinstance(context, str) else context
        v = q + CONTEXT_WEIGHT * c
        return (v / np.linalg.norm(v)).astype(np.float32)

    def _note_retrieval(self, s: Session, query: str, trigger: str, branch: str | None, scope):
        t = s.clock.now()
        if s.first_retrieval_t is None:
            s.first_retrieval_t = t
        s.turn_cost["retrievals"] += 1
        ev = {"timestamp_s": round(t, 3), "query": query, "trigger": trigger, "branch": branch,
              "scoped": scope is not None}
        s.turn_events.append(ev)
        self.logger.emit(s, "retrieval_started", query=query, trigger=trigger, branch=branch,
                         scope_chunks=len(scope) if scope is not None else len(self.index.chunks))

    def _log_retrieval(self, s: Session, res: RetrievalResult, trigger: str, branch: str | None):
        self.logger.emit(s, "retrieval_completed", branch=branch, trigger=trigger, query=res.query,
                         top_ids=res.ids, latency_ms=res.latency_ms, chunks_searched=res.chunks_searched,
                         corpus_size=res.corpus_size, mode=res.mode)

    async def retrieve_and_cite(self, s: Session, sq: SubQuery, scope=None, trigger="final",
                                context_vec=None, branch: str | None = None) -> tuple[ClaimResult, RetrievalResult]:
        key = (sq.query, tuple(scope) if scope is not None else None)
        pending = s.cite_cache.get(key)
        if pending is not None:            # same delta query already ran (e.g. early patch mid-stream)
            self.logger.emit(s, "retrieval_reused", branch=branch, query=sq.query, trigger=trigger,
                             speculative_hit=True)
            return await pending
        fut = asyncio.get_running_loop().create_future()
        s.cite_cache[key] = fut
        try:
            qvec = await self._qvec(sq.query, context_vec)
            self._note_retrieval(s, sq.query, trigger, branch, scope)
            res = await self.retriever.search(sq.query, scope=scope, qvec=qvec,
                                              cluster=sq.cluster if trigger != "fallback_sequential" else None)
            self._log_retrieval(s, res, trigger, branch)
            out = (await self._cite(s, sq, res, qvec), res)
            fut.set_result(out)
            return out
        except Exception as exc:
            s.cite_cache.pop(key, None)
            fut.set_exception(exc)
            fut.exception()
            raise

    async def _cite(self, s: Session, sq: SubQuery, res: RetrievalResult, qvec, speculative=False) -> ClaimResult:
        cr = await cite_then_write(sq.query, sq.slots, [h.chunk for h in res.hits], self.llm, self.verifier,
                                   self.cfg, qvec=qvec)
        r = cr.llm
        if r is not None:
            s.turn_cost["llm_calls"] += 1
            s.turn_cost["prompt_tokens"] += r.prompt_tokens
            s.turn_cost["completion_tokens"] += r.completion_tokens
            s.turn_cost["est_usd"] += r.est_cost_usd
            s.turn_cost["would_be_prompt_tokens"] += r.extra.get("would_be_prompt_tokens", r.prompt_tokens)
            self.logger.emit(s, "llm_call", purpose="speculative_cite_then_write" if speculative else "cite_then_write",
                             backend=r.backend, model=r.model, prompt_tokens=r.prompt_tokens,
                             completion_tokens=r.completion_tokens, latency_ms=r.latency_ms,
                             est_cost_usd=r.est_cost_usd, would_be_prompt_tokens=r.extra.get("would_be_prompt_tokens"))
        self.logger.emit(s, "claim_verified", sub_query=sq.query, speculative=speculative, grounded=cr.grounded,
                         citation=cr.chunk_id, selected_index=cr.selected_index, candidates=cr.candidate_ids,
                         overlap=cr.overlap, missing_critical=cr.missing_critical, nli_label=cr.nli_label,
                         confidence=cr.confidence, reason=cr.reason, verify_ms=cr.verify_ms,
                         rejected_index=cr.rejected_index, text=cr.text)
        return cr

    # --------------------------------------------------------- utterance end
    async def _answer_fresh(self, s: Session, text: str, t_end: float, inherited: dict | None = None,
                            context_vec=None, d: Decomposition | None = None) -> dict:
        if d is None:
            d = await decompose(text, self.index, self.cfg, inherited_slots=inherited)
        self.logger.emit(s, "decomposition", partial=False, **d.log_fields())
        if not d.sub_queries:
            for c in list(s.branches):
                self._prune(s, s.branches.pop(c), "utterance has no corpus affinity")
            return await self._clarify(s, text, t_end, "low corpus affinity")

        ctx = context_vec if context_vec is not None else text
        final_clusters = {sq.cluster for sq in d.sub_queries}
        for c in [c for c in s.branches if c not in final_clusters]:
            self._prune(s, s.branches.pop(c), "hypothesis mismatch with final utterance")
        results = await asyncio.gather(*(self._resolve(s, sq, s.branches.get(sq.cluster), ctx)
                                         for sq in d.sub_queries))

        claims: list[Claim] = []
        for sq, cr in results:
            emb = await self.index.embedder.embed(cr.text if cr.grounded else sq.topic)
            dup = next((c for c in claims if cr.grounded and c.supporting_chunk_id == cr.chunk_id
                        and float(c.embedding @ emb) > 0.9), None)
            if dup is not None:  # evidence fusion: same passage, same fact -> one claim
                dup.source_sub_query += f" / {sq.topic}"
                continue
            claims.append(claim_from_result(cr, sq, emb))
        answer = Branch(new_id("b"), d.sub_queries[0].cluster, d.sub_queries, "promoted", s.clock.now(),
                        claims=claims, status="promoted")
        for c, br in list(s.branches.items()):
            br.status = "promoted"
            br.claims = [cl for cl in claims if cl.topic_cluster == c]
            self.logger.emit(s, "branch_promoted", branch=br.id, cluster=c, trigger=br.trigger,
                             grounding_score=br.grounding_score, forked_at_s=br.created_t,
                             lead_time_s=t_end - br.created_t)
        anchor = context_vec if context_vec is not None else await self.index.embedder.embed(text)
        s.graph = AnswerGraph(answer, text, anchor, d.sub_queries[0].slots if d.sub_queries else {},
                              {c.chunk_id: int(self.index.labels[c.idx]) for c in self.index.chunks})
        return self._emit_answer(s, text, t_end, "answer" if inherited is None else "followup",
                                 [sq.query for sq in d.sub_queries], cause="initial")

    async def _resolve(self, s: Session, sq: SubQuery, br: Branch | None, ctx) -> tuple[SubQuery, ClaimResult]:
        qvec = await self._qvec(sq.query, ctx)
        how = "sequential"
        if br is not None and br.retrieval is not None and br.retrieval_query == sq.query:
            res = br.retrieval
            spec = br.speculative.get((sq.query, tuple(res.ids)))
            how = "speculative" if spec is not None else "reused"
            self.logger.emit(s, "retrieval_reused", branch=br.id, query=sq.query, speculative_hit=spec is not None)
            cr = spec if spec is not None else await self._cite(s, sq, res, qvec)
        elif br is not None:
            how = "refreshed"
            cr, res = await self.retrieve_and_cite(s, sq, scope=br.scope, trigger="final_refresh",
                                                   context_vec=ctx, branch=br.id)
        else:
            cr, res = await self.retrieve_and_cite(s, sq, trigger="final", context_vec=ctx)
        if not cr.grounded and (how != "sequential" or (br is not None and br.scope is not None)):
            # safety valve: early/cluster-seeded evidence was not good enough -> plain sequential path
            self.logger.emit(s, "fallback_sequential", sub_query=sq.query, reason=cr.reason, previous=how)
            cr2, _ = await self.retrieve_and_cite(s, sq, trigger="fallback_sequential", context_vec=ctx)
            if cr2.grounded or cr.selected_index is None:
                cr = cr2
        return sq, cr

    async def resolve_delta(self, s: Session, sq: SubQuery, scope, context_vec) -> tuple[ClaimResult, RetrievalResult]:
        """Delta sub-query of a refinement: reuse the scoped branch retrieval that already ran
        mid-stream for this clause, otherwise run a scoped search now."""
        br = s.branches.get(sq.cluster)
        if br is not None and br.retrieval is not None and br.retrieval_query == sq.query:
            br.status = "promoted"
            spec = br.speculative.get((sq.query, tuple(br.retrieval.ids)))
            self.logger.emit(s, "retrieval_reused", branch=br.id, query=sq.query, speculative_hit=spec is not None)
            if spec is not None:
                return spec, br.retrieval
            return await self._cite(s, sq, br.retrieval, await self._qvec(sq.query, context_vec)), br.retrieval
        return await self.retrieve_and_cite(s, sq, scope=scope, trigger="delta", context_vec=context_vec)

    async def _refine(self, s: Session, text: str, t_end: float, delta: list[SubQuery], rel: float) -> dict:
        g = s.graph
        self.logger.emit(s, "refinement", relation=rel, delta_sub_queries=[sq.query for sq in delta],
                         claims_before=[c.id for c in g.claims], action="patch claim graph in place")
        for c in [c for c in s.branches if c not in {sq.cluster for sq in delta}]:
            self._prune(s, s.branches.pop(c), "refinement handled by claim patching")
        turn_log = {"sub_queries": []}
        await apply_constraint(self, s, text, delta, turn_log)
        g.branch.answer_version += 1
        return self._emit_answer(s, text, t_end, "refinement", turn_log["sub_queries"], cause="patch")

    async def _present(self, s: Session, text: str, t_end: float) -> dict:
        style, n = parse_style(text)
        for c in list(s.branches):
            self._prune(s, s.branches.pop(c), "presentation-only turn")
        self.logger.emit(s, "retrieval_decision", retrieval_required=False, reason="presentation_restructure",
                         style=style, n=n)
        prev = s.last_answer or {}
        claims = [c for c in prev.get("claims", []) if c.get("citation")]
        uncertainty = prev.get("uncertainty")
        if style == "translate":
            body, cits = prev.get("answer", ""), prev.get("citations", [])
            uncertainty = ("Translation needs a generative LLM backend (LLM_BACKEND); the extractive backend "
                           "never rewrites corpus text, so the previous answer is repeated unchanged.")
            if not self.llm.is_local:
                body, cits, uncertainty = await self._llm_restyle(s, text, prev, body, cits, uncertainty)
        else:
            body, cits = restructure(claims, style, n)
        return self._emit_answer(s, text, t_end, "presentation", [], cause="presentation",
                                 override=(body, cits, uncertainty), retrieval_required=False,
                                 reason="presentation_restructure")

    async def _llm_restyle(self, s, request, prev, body, cits, unc):
        prompt = (f"Rewrite the answer below as requested: \"{request}\". Keep every [Doc ...] citation marker "
                  f"attached to the same statement. Do not add, remove or change any facts.\n\nAnswer:\n"
                  f"{prev.get('answer', '')}\n\nReturn JSON {{\"text\": \"...\"}}.")
        try:
            r = await self.llm.generate_json(prompt, None, task="restyle")
            s.turn_cost["llm_calls"] += 1
            self.logger.emit(s, "llm_call", purpose="presentation_restyle", backend=r.backend, model=r.model,
                             prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
                             latency_ms=r.latency_ms, est_cost_usd=r.est_cost_usd)
            out = str(r.data.get("text", ""))
            used = re.findall(r"\[([^\[\]]+ §[\w.]+)\]", out)
            if out and set(used) <= set(cits) and set(cits) <= set(used):
                return out, cits, None
            unc = "Restyled output changed the citation set, so it was rejected; showing the previous answer."
        except Exception as exc:  # pragma: no cover - network
            unc = f"Restyle call failed ({exc.__class__.__name__}); showing the previous answer."
        return body, cits, unc

    async def _clarify(self, s: Session, text: str, t_end: float, reason: str) -> dict:
        self.logger.emit(s, "retrieval_decision", retrieval_required=False, reason=reason)
        return self._emit_answer(s, text, t_end, "clarification", [], cause="clarification",
                                 override=(CLARIFY_TEXT, [], "No corpus evidence matched this request."),
                                 retrieval_required=False, reason=reason, bump=False)

    def _relation(self, g: AnswerGraph, vec: np.ndarray) -> float:
        cands = [float(g.anchor_vec @ vec)]
        cands += [float(self.index.centroids[c] @ vec) for c in g.clusters if c >= 0]
        cands += [float(c.embedding @ vec) for c in g.claims if c.grounded]
        return max(cands)

    # ------------------------------------------------------------- output
    def _emit_answer(self, s: Session, text: str, t_end: float, kind: str, sub_queries: list[str], cause: str,
                     override=None, retrieval_required=True, reason=None, bump=True) -> dict:
        claims = s.graph.claims if (s.graph is not None and override is None) else []
        if override is None:
            answer, citations, uncertainty = compose_answer(claims, refinement=(kind == "refinement"))
        else:
            answer, citations, uncertainty = override
        parent = (s.answer_version or None) if bump else None
        if bump:
            s.answer_version += 1
        claim_view = [{"id": c.id, "text": c.text, "citation": c.supporting_chunk_id, "confidence": c.confidence,
                       "version": c.version, "status": c.status, "sub_query": c.source_sub_query}
                      for c in claims]
        if override is not None and kind == "presentation":
            claim_view = (s.last_answer or {}).get("claims", [])
        if bump:
            s.versions.append({"answer_version": s.answer_version, "parent_version": parent, "cause": cause,
                               "citations": citations, "claims": [(c["id"], c["version"]) for c in claim_view]})
            self.logger.emit(s, "answer_version", answer_version=s.answer_version, parent_version=parent,
                             cause=cause, answer=answer, citations=citations, uncertainty=uncertainty,
                             claims=[{k: c[k] for k in ("id", "version", "status", "citation")} for c in claim_view])
        t_ready = s.clock.now()
        latency = {
            "utterance_end_s": round(t_end, 3),
            "first_retrieval_s": None if s.first_retrieval_t is None else round(s.first_retrieval_t, 3),
            "early_retrieval": s.first_retrieval_t is not None and s.first_retrieval_t < t_end,
            "answer_ready_s": round(t_ready, 3),
            "post_utterance_ms": round((t_ready - t_end) * 1000, 1),
        }
        seeds = s.all_branches
        out = {
            "session_id": s.id, "turn": s.turn, "kind": kind, "transcript": text,
            "retrieval_required": retrieval_required, "reason": reason,
            "retrieval_events": list(s.turn_events),
            "sub_queries": sub_queries,
            "answer": answer, "citations": citations, "uncertainty": uncertainty,
            "answer_version": s.answer_version, "parent_version": parent,
            "claims": claim_view, "latency": latency, "cost": dict(s.turn_cost),
            "branches": {"forked": len(seeds), "pruned": sum(b.status == "pruned" for b in seeds),
                         "promoted": sum(b.status == "promoted" for b in seeds)},
        }
        self.logger.emit(s, "turn_summary", kind=kind, answer_version=s.answer_version, latency=latency,
                         cost=s.turn_cost, retrieval_events=len(s.turn_events), citations=citations,
                         sub_queries=sub_queries)
        s.all_branches = []
        s.last_answer = out
        return out


def compose_answer(claims: list[Claim], refinement: bool = False) -> tuple[str, list[str], str | None]:
    def render(cs):
        return " ".join(f"{c.text} [{c.supporting_chunk_id}]" for c in cs)

    grounded = [c for c in claims if c.grounded and c.supporting_chunk_id]
    if refinement:
        kept = [c for c in grounded if c.status in ("preserved", "reconfirmed")]
        changed = [c for c in grounded if c.status == "patched"]
        added = [c for c in grounded if c.status == "added"]
        parts = []
        if kept:
            parts.append("Still applies: " + render(kept))
        if changed:
            parts.append("Updated: " + render(changed))
        if added:
            parts.append("In addition: " + render(added))
        answer = "\n".join(parts)
    else:
        answer = render(grounded)
    citations = list(dict.fromkeys(c.supporting_chunk_id for c in grounded))
    unc = [c for c in claims if not c.grounded]
    uncertainty = None
    if unc:
        uncertainty = " ".join(
            f"Could not verify “{c.source_sub_query}” from the corpus"
            + (" (previous claim retracted)" if c.status == "retracted" else "") + "." for c in unc)
    if not grounded:
        answer = "I could not find corpus evidence to answer this reliably."
    return answer, citations, uncertainty
