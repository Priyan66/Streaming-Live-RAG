"""Guide Example 2: a late-arriving detail refines the answer instead of restarting."""
from tests.conftest import events_of


def test_refines_in_place_and_preserves_prior_claims(run):
    outs, ev = run("example2_late_constraint")
    v1, v2 = outs
    assert v1["kind"] == "answer" and v2["kind"] == "refinement"
    assert v2["answer_version"] == 2 and v2["parent_version"] == 1
    before = {c["id"] for c in v1["claims"]}
    after = {c["id"] for c in v2["claims"]}
    assert before <= after                                       # session state not cleared
    assert set(v1["citations"]) <= set(v2["citations"])          # prior citations preserved
    assert any(c["status"] in ("preserved", "reconfirmed") for c in v2["claims"])


def test_delta_queries_add_the_new_constraints(run):
    outs, _ = run("example2_late_constraint")
    v2 = outs[1]
    added = [c for c in v2["claims"] if c["status"] == "added" and c["citation"]]
    assert added, "late detail should add at least one delta claim"
    assert "Doc_07" in {c["citation"].split()[0] for c in added}   # post-travel booking exception


def test_refinement_never_searches_the_full_corpus(run):
    outs, ev = run("example2_late_constraint")
    done = events_of(ev, 2, "retrieval_completed")
    assert done, "refinement should run scoped delta retrieval"
    assert all(e["chunks_searched"] < e["corpus_size"] for e in done)
    assert not events_of(ev, 2, "topic_shift")


def test_slot_change_patches_only_dependent_claims(run):
    outs, ev = run("example5_slot_change")
    v1, v2 = outs
    assert v2["kind"] == "refinement"
    statuses = {c["status"] for c in v2["claims"]}
    assert "patched" in statuses                                 # the 30-person venue claim was re-grounded
    stale = events_of(ev, 2, "claim_stale")
    assert any(e["tier"] == "slot_diff" for e in stale)
    assert events_of(ev, 2, "early_patch"), "slot change should start re-verification mid-utterance"
    assert v2["latency"]["early_retrieval"]


def test_topic_shift_discards_graph_and_reforks(run):
    outs, ev = run("example4_topic_shift")
    assert outs[1]["kind"] == "answer"
    assert events_of(ev, 2, "topic_shift")
    assert {c.split()[0] for c in outs[1]["citations"]} == {"Doc_33"}
