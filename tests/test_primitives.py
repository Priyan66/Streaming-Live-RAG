"""Unit tests for the shared primitives and corpus handling."""
import asyncio
import json

from corpus.loader import load_corpus
from engine.decomposer import syntactic_split
from engine.primitives import softmax_entropy, stem
from engine.slots import changed_slots, extract_slots, rewrite_with_slots
import numpy as np


def test_slots_are_literal_spans(rt):
    s = extract_slots("I need a workshop in Pune for thirty people on 12 March, budget INR 50,000", rt.index.gazetteer)
    assert s["headcount"] == "30"
    assert s["location"] == "Pune"
    assert "12 march" in s["date"]
    assert s["amount"].startswith("inr")


def test_slot_diff_and_rewrite():
    old, new = {"headcount": "30"}, {"headcount": "60"}
    assert changed_slots(old, new) == {"headcount"}
    assert rewrite_with_slots("venue for 30 people", old, new) == "venue for 60 people"


def test_syntactic_split_drops_filler_and_retractions():
    assert syntactic_split("Pune for 30 people, and I need") == ["Pune for 30 people"]
    assert syntactic_split("Forget the trip, the TV shows no picture.") == ["the TV shows no picture"]
    assert len(syntactic_split("the cancellation policy and the catering options")) == 2


def test_stemmer_groups_word_forms():
    assert stem("cancellation") == stem("cancelled") == stem("cancellations")
    assert stem("policies") == stem("policy")


def test_entropy_bounds():
    assert softmax_entropy(np.array([0.9, 0.1, 0.1]), 0.05) < 0.1
    assert softmax_entropy(np.array([0.5, 0.5, 0.5]), 0.05) > 0.99


def test_loader_handles_plain_text_and_jsonl(tmp_path):
    (tmp_path / "Handbook.txt").write_text("First paragraph here.\n\nSecond paragraph here.", encoding="utf-8")
    (tmp_path / "extra.jsonl").write_text(json.dumps({"doc_id": "Doc_77", "section": "3", "text": "Hi."}),
                                          encoding="utf-8")
    ids = {c.chunk_id for c in load_corpus(tmp_path)}
    assert ids == {"Handbook §1", "Handbook §2", "Doc_77 §3"}


def test_index_clusters_cover_every_chunk(rt):
    idx = rt.index
    assert sorted(i for m in idx.cluster_members for i in m) == list(range(len(idx.chunks)))
    assert np.allclose(np.linalg.norm(idx.centroids, axis=1), 1.0, atol=1e-4)


def test_embed_is_cached(rt):
    e = rt.index.embedder
    asyncio.run(e.embed("a sentence for the cache test"))
    hits = e.cache_hits
    asyncio.run(e.embed("a sentence for the cache test"))
    assert e.cache_hits == hits + 1
