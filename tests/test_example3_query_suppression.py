"""Guide Example 3: presentation-only turn — no corpus query, no new citations."""
from tests.conftest import events_of


def test_presentation_turn_runs_no_retrieval(run):
    outs, ev = run("example3_query_suppression")
    v1, v2 = outs
    assert v2["kind"] == "presentation"
    assert v2["retrieval_required"] is False
    assert v2["reason"] == "presentation_restructure"
    assert not events_of(ev, 2, "retrieval_started")
    assert v2["cost"]["llm_calls"] == 0 and v2["cost"]["retrievals"] == 0
    assert any(e["action"] == "suppress" for e in events_of(ev, 2, "controller_decision"))


def test_presentation_keeps_prior_citations_and_formats_bullets(run):
    v1, v2 = run("example3_query_suppression")[0]
    assert v2["citations"] == v1["citations"]
    bullets = [line for line in v2["answer"].splitlines() if line.startswith("- ")]
    assert len(bullets) == 2
    for cid in v1["citations"]:
        assert f"[{cid}]" in v2["answer"]
    assert v2["answer_version"] == v1["answer_version"] + 1
