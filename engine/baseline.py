"""Naive batch RAG baseline used for the benchmark comparison.

Waits for the utterance to end, retrieves once for the whole utterance (no
decomposition), answers with up to three cited sentences, restarts from scratch
on follow-ups (re-querying the concatenated conversation over the full corpus)
and never suppresses presentation-only turns. It uses the *same* retriever,
generator and verifier as the streaming engine so the comparison isolates the
architecture rather than the models.
"""
from __future__ import annotations

from engine.claims import SubQuery, claim_from_result, new_id
from engine.pipeline import StreamingEngine, compose_answer, _clean_chunk
from engine.claims import AnswerGraph, Branch
from engine.session import Session
from engine.slots import extract_slots


class BaselineEngine(StreamingEngine):
    name = "baseline"

    async def on_chunk(self, s: Session, text: str):
        s.buffer = f"{s.buffer} {_clean_chunk(text)}".strip()
        s.n_chunks += 1
        self.logger.emit(s, "chunk_received", chunk_index=s.n_chunks, text=text, buffer=s.buffer)
        self.logger.emit(s, "controller_decision", chunk_index=s.n_chunks, action="wait",
                         reason="batch baseline waits for utterance end")

    async def on_utterance_end(self, s: Session) -> dict:
        t_end = s.clock.now()
        text = s.buffer.strip()
        self.logger.emit(s, "utterance_end", transcript=text, n_chunks=s.n_chunks)
        history = getattr(s, "_history", [])
        history.append(text)
        s._history = history
        query = " ".join(history)             # restart: whole conversation, full corpus
        sq = SubQuery(query, extract_slots(query, self.index.gazetteer), -1, 0.0)
        qvec = await self._qvec(query, None)
        self._note_retrieval(s, query, "batch", None, None)
        res = await self.retriever.search(query, qvec=qvec)
        self._log_retrieval(s, res, "batch", None)
        claims, remaining = [], list(res.hits)
        for _ in range(3):                    # up to three cited sentences from distinct passages
            if not remaining:
                break
            cand = type(res)(res.query, remaining, res.chunks_searched, res.corpus_size, 0.0, res.mode)
            cr = await self._cite(s, sq, cand, qvec)
            if not cr.grounded:
                break
            claims.append(claim_from_result(cr, sq, await self.index.embedder.embed(cr.text)))
            remaining = [h for h in remaining if h.chunk.chunk_id != cr.chunk_id]
        if not claims:
            claims.append(claim_from_result(cr, sq, qvec))
        s.graph = AnswerGraph(Branch(new_id("b"), -1, [sq], "batch", t_end, claims=claims, status="promoted"),
                              query, qvec, sq.slots)  # baseline never patches, so no cluster map needed
        return self._emit_answer(s, text, t_end, "answer", [query], cause="restart")


__all__ = ["BaselineEngine", "compose_answer"]
