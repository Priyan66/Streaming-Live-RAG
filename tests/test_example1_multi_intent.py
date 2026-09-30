"""Guide Example 1: incremental multi-intent utterance."""
from tests.conftest import events_of


def test_waits_on_unstable_prefix_then_retrieves_early(run):
    outs, ev = run("example1_multi_intent")
    decisions = events_of(ev, 1, "controller_decision")
    assert decisions[0]["action"] == "wait"                      # 0.0 s: intent incomplete
    assert decisions[1]["action"] == "fork"                      # 0.8 s: slots Pune / 30 people
    starts = events_of(ev, 1, "retrieval_started")
    assert starts[0]["trigger"] == "provisional"
    assert 0.8 <= starts[0]["t_stream"] < 1.6
    out = outs[0]
    assert out["latency"]["early_retrieval"]
    assert out["latency"]["first_retrieval_s"] < out["latency"]["utterance_end_s"]


def test_decomposes_into_three_parallel_sub_queries(run):
    outs, ev = run("example1_multi_intent")
    out = outs[0]
    assert len(out["sub_queries"]) == 3
    triggers = [e["trigger"] for e in events_of(ev, 1, "retrieval_started")]
    assert triggers.count("multi_intent") == 2                   # cancellation + catering forked at 1.6 s
    assert {c.split()[0] for c in out["citations"]} >= {"Doc_12", "Doc_31", "Doc_09"}
    assert out["branches"]["promoted"] == 3


def test_answer_is_ready_quickly_because_retrieval_was_already_done(run):
    outs, ev = run("example1_multi_intent")
    assert outs[0]["latency"]["post_utterance_ms"] < 500
    assert any(e["event"] == "retrieval_reused" for e in events_of(ev, 1))
    assert not any(e["event"] == "fallback_sequential" for e in events_of(ev, 1))


def test_output_matches_guide_event_record_shape(run):
    out = run("example1_multi_intent")[0][0]
    for key in ("retrieval_events", "sub_queries", "answer", "citations", "uncertainty"):
        assert key in out
    for e in out["retrieval_events"]:
        assert {"timestamp_s", "query", "trigger"} <= set(e)
