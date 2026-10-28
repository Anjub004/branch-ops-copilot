"""Three ways to find the right paragraph — and to admit when there isn't one.

Every retriever returns a list of Hit(chunk, score, rank) and has a `confident(hits)` check.
If it isn't confident, the RAG layer refuses instead of letting the model guess.
"""
import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


@dataclass
class Hit:
    chunk: object
    score: float
    rank: int


def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in ENGLISH_STOP_WORDS]


def _top(chunks, scores, k):
    order = np.argsort(scores)[::-1][:k]
    return [Hit(chunks[i], float(scores[i]), r + 1) for r, i in enumerate(order)]


class KeywordRetriever:
    """BM25: the classic keyword ranking used by search engines. Fast, free, but blind to synonyms."""
    name = "keyword (BM25)"

    def __init__(self, chunks, min_score=4.0):
        self.chunks, self.min_score = chunks, min_score
        self.bm25 = BM25Okapi([tokenize(c.indexed_text) for c in chunks])

    def search(self, query, k=3):
        return _top(self.chunks, self.bm25.get_scores(tokenize(query)), k)

    def confident(self, hits):
        return bool(hits) and hits[0].score >= self.min_score


class EmbeddingRetriever:
    """Compares meaning instead of exact words, so 'bread' can find 'bakery'."""

    def __init__(self, chunks, embedder, min_score=0.55):
        self.chunks, self.embedder, self.min_score = chunks, embedder, min_score
        self.name = f"embeddings ({embedder.name})"
        self.matrix = embedder.embed([c.indexed_text for c in chunks])

    def search(self, query, k=3):
        query_vector = self.embedder.embed([query])[0]
        return _top(self.chunks, self.matrix @ query_vector, k)

    def confident(self, hits):
        return bool(hits) and hits[0].score >= self.min_score


class HybridRetriever:
    """Reciprocal Rank Fusion: run both, then reward chunks that rank well in either list.

    RRF score = sum over retrievers of 1 / (k + rank). It ignores raw scores, so BM25 numbers
    and cosine similarities never need to be put on the same scale.
    """

    def __init__(self, keyword, embedding, rrf_k=60, depth=20):
        self.keyword, self.embedding, self.rrf_k, self.depth = keyword, embedding, rrf_k, depth
        self.name = "hybrid (BM25 + embeddings, RRF)"
        self._last = {}

    def search(self, query, k=3):
        kw = self.keyword.search(query, self.depth)
        em = self.embedding.search(query, self.depth)
        self._last = {"keyword": kw, "embedding": em}
        fused = {}
        for hits in (kw, em):
            for h in hits:
                fused[h.chunk] = fused.get(h.chunk, 0.0) + 1.0 / (self.rrf_k + h.rank)
        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]
        return [Hit(chunk, score, r + 1) for r, (chunk, score) in enumerate(ranked)]

    def confident(self, hits):
        # Confident if either underlying retriever is confident about this query.
        return bool(hits) and (self.keyword.confident(self._last.get("keyword", []))
                               or self.embedding.confident(self._last.get("embedding", [])))
