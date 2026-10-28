"""Offline tests for Milestone 1. No model server, no spaCy model needed:
a tiny fake embedder and a fake chat client stand in."""
import os
import sys
from types import SimpleNamespace
import zlib

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from copilot import llm, rag  # noqa: E402
from copilot.chunking import load_chunks  # noqa: E402
from copilot.retrievers import EmbeddingRetriever, Hit, HybridRetriever, KeywordRetriever  # noqa: E402
from eval.run_eval import is_match, load_questions, summarise  # noqa: E402


class BagOfWordsEmbedder:
    """Deterministic stand-in for a real embedder: hashed bag of words."""
    name = "fake"

    def embed(self, texts):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, zlib.crc32(w.encode()) % 64] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return out / norms


@pytest.fixture(scope="module")
def chunks():
    return load_chunks()


def test_chunks_have_titles_and_sources(chunks):
    assert len(chunks) == 86
    first = chunks[0]
    assert first.source.endswith(".md") and first.title and first.text
    assert first.title in first.indexed_text


def test_questions_file_is_consistent(chunks):
    questions = load_questions()
    assert len(questions) == 30
    sources = {c.source for c in chunks}
    for q in questions:
        if q["type"] == "unanswerable":
            assert q["doc"] is None
        else:  # every expected answer really exists in the documents
            assert q["doc"] + ".md" in sources
            assert any(is_match(Hit(c, 0, 1), q) for c in chunks), q["id"]


def test_keyword_finds_literal_question(chunks):
    hits = KeywordRetriever(chunks).search("What temperature must freezers be kept at?")
    assert hits[0].chunk.source == "cold_chain.md"


def test_hybrid_fuses_both_rankings(chunks):
    keyword = KeywordRetriever(chunks)
    embedding = EmbeddingRetriever(chunks, BagOfWordsEmbedder())
    hits = HybridRetriever(keyword, embedding).search("How many loyalty points for a voucher?")
    assert hits[0].chunk.source == "loyalty_card.md"
    assert [h.rank for h in hits] == [1, 2, 3]


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, model, messages, **kwargs):
        self.calls.append(messages)
        msg = SimpleNamespace(content="Same day only [returns_policy.md]")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


@pytest.fixture
def fake_llm(monkeypatch):
    completions = FakeCompletions()
    monkeypatch.setattr(llm, "client", SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    return completions


def test_rag_refuses_without_calling_model(chunks, fake_llm):
    retriever = KeywordRetriever(chunks, min_score=999)  # never confident
    result = rag.answer("anything", retriever)
    assert result["refused"] and fake_llm.calls == []


def test_rag_sends_context_and_returns_sources(chunks, fake_llm):
    result = rag.answer("How many days do customers have to return non-perishable items?",
                        KeywordRetriever(chunks))
    assert not result["refused"]
    assert "returns_policy.md" in result["sources"]
    assert "[returns_policy.md]" in fake_llm.calls[0][0]["content"]


def test_summary_maths():
    rows = [{"type": "literal", "rank": 1, "confident": True},
            {"type": "paraphrase", "rank": 2, "confident": True},
            {"type": "paraphrase", "rank": None, "confident": False},
            {"type": "unanswerable", "rank": None, "confident": False}]
    s = summarise(rows)
    assert s["all"]["hit@1"] == pytest.approx(1 / 3)
    assert s["all"]["hit@3"] == pytest.approx(2 / 3)
    assert s["all"]["mrr"] == pytest.approx((1 + 0.5) / 3)
    assert s["refused"] == 1 and s["false_refusals"] == 1
