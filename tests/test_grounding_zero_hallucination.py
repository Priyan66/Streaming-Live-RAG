"""G4: fabricated IDs are structurally impossible; misattributed claims are caught."""
import asyncio

import pytest

from engine.grounding import UNCERTAIN_MARKER, cite_then_write, keyword_overlap
from engine.llm_client import LLMResponse
from simulator.stream_replay import replay
from telemetry.logger import TraceLogger
from telemetry.validate import validate_trace


class ScriptedLLM:
    """Stands in for a generator that misbehaves in a chosen way."""
    name = "scripted"
    model = "scripted"
    is_local = True

    def __init__(self, data):
        self.data = data

    async def generate_json(self, prompt, ctx=None, task="cite"):
        assert "Doc_" not in prompt, "real Doc IDs must never be shown to the generator"
        return LLMResponse(dict(self.data), 10, 10, 0.0, self.name, self.model)


def _chunks(rt, *ids):
    return [rt.index.chunk_by_id(i) for i in ids]


def _cite(rt, data, ids=("Doc_12 §2", "Doc_09 §1")):
    return asyncio.run(cite_then_write("How many people fit in Orchid Hall?", {}, _chunks(rt, *ids),
                                       ScriptedLLM(data), rt.verifier, rt.cfg))


@pytest.mark.parametrize("bad_index", [99, 0, -1, "Doc_999", "[7]", 1.5])
def test_invalid_indices_are_rejected_not_repaired(rt, bad_index):
    res = _cite(rt, {"selected_index": bad_index, "claim_text": "Orchid Hall seats up to 40 people."})
    assert not res.grounded and res.chunk_id is None and res.text == UNCERTAIN_MARKER


def test_wrong_number_is_caught_by_overlap_check(rt):
    res = _cite(rt, {"selected_index": 1, "claim_text": "Orchid Hall seats up to 400 people in a classroom layout."})
    assert not res.grounded
    assert "400" in res.missing_critical


def test_contradiction_is_caught_by_nli(rt):
    res = _cite(rt, {"selected_index": 1, "claim_text": "In-house catering is not required at Orchid Hall."})
    assert not res.grounded


def test_misattributed_but_real_citation_is_caught(rt):
    # true statement, but passage [2] (catering prices) does not say it
    res = _cite(rt, {"selected_index": 2, "claim_text": "Orchid Hall seats up to 40 people in a classroom layout."})
    assert not res.grounded


def test_faithful_paraphrase_is_grounded(rt):
    res = _cite(rt, {"selected_index": 1, "claim_text": "Orchid Hall can seat 40 people in a classroom layout."})
    assert res.grounded and res.chunk_id == "Doc_12 §2"


def test_explicit_null_becomes_uncertainty(rt):
    res = _cite(rt, {"selected_index": None, "uncertainty_reason": "not in passages"})
    assert not res.grounded and res.reason == "not in passages"


def test_keyword_overlap_flags_foreign_names():
    ratio, missing = keyword_overlap("Venue Zeta seats 40 people.", "Orchid Hall seats up to 40 people.")
    assert "Zeta" in missing


def test_out_of_corpus_question_abstains(rt):
    logger = TraceLogger()
    eng = rt.engine(logger)
    from simulator.stream_replay import chunk_text
    sc = {"id": "ooc", "turns": [chunk_text("What is the policy on bringing pets to the office?")]}
    out = asyncio.run(replay(eng, sc))[0]
    assert out["citations"] == []
    assert out["uncertainty"]


def test_every_citation_in_every_example_is_real_and_traceable(rt, run):
    for name in ("example1_multi_intent", "example2_late_constraint", "example3_query_suppression",
                 "example4_topic_shift", "example5_slot_change"):
        outs, events = run(name)
        for out in outs:
            assert set(out["citations"]) <= rt.index.valid_ids
        report = validate_trace(events, rt.index.valid_ids)
        assert report["fabricated_citations"] == 0
        assert report["untraceable_citations"] == 0
        assert report["coverage_pct"] == 100.0, report["failures"]
