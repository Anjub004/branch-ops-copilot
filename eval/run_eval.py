"""Measure retrieval before touching prompts.

    python -m eval.run_eval                 # offline: BM25 vs spaCy vectors vs hybrid
    EMBEDDER=ollama python -m eval.run_eval # use a real embedding model via Ollama

Metrics (answerable questions):
  hit@1 / hit@3 : the expected paragraph is the top result / in the top 3
  MRR           : mean of 1/rank of the expected paragraph (0 if not in the top 10)
Refusals (unanswerable questions):
  refused       : the retriever was not confident, so RAG would refuse instead of guessing
  false refusals: answerable questions it wrongly refused
"""
import csv
import json
from pathlib import Path

from copilot.chunking import load_chunks
from copilot.embedders import get_embedder
from copilot.retrievers import EmbeddingRetriever, HybridRetriever, KeywordRetriever

HERE = Path(__file__).parent


def load_questions():
    return [json.loads(line) for line in (HERE / "questions.jsonl").read_text().splitlines() if line.strip()]


def is_match(hit, q):
    return hit.chunk.source == q["doc"] + ".md" and q["evidence"].lower() in hit.chunk.text.lower()


def evaluate(retriever, questions, depth=10):
    rows = []
    for q in questions:
        hits = retriever.search(q["question"], depth)
        rank = next((h.rank for h in hits if q["doc"] and is_match(h, q)), None)
        rows.append({"id": q["id"], "type": q["type"], "rank": rank,
                     "confident": retriever.confident(hits[:3]),
                     "top_source": hits[0].chunk.source, "top_score": round(hits[0].score, 3)})
    return rows


def summarise(rows):
    def stats(subset):
        n = len(subset) or 1
        return {"n": len(subset),
                "hit@1": sum(r["rank"] == 1 for r in subset) / n,
                "hit@3": sum(r["rank"] is not None and r["rank"] <= 3 for r in subset) / n,
                "mrr": sum(1 / r["rank"] for r in subset if r["rank"]) / n}
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    return {
        "all": stats(answerable),
        "literal": stats([r for r in answerable if r["type"] == "literal"]),
        "paraphrase": stats([r for r in answerable if r["type"] == "paraphrase"]),
        "refused": sum(not r["confident"] for r in unanswerable),
        "n_unanswerable": len(unanswerable),
        "false_refusals": sum(not r["confident"] for r in answerable),
    }


def main():
    chunks, questions = load_chunks(), load_questions()
    keyword = KeywordRetriever(chunks)
    embedding = EmbeddingRetriever(chunks, get_embedder())
    retrievers = [keyword, embedding, HybridRetriever(keyword, embedding)]

    lines = [f"Evaluated on {len(questions)} questions over {len(chunks)} chunks.", "",
             "| Retriever | hit@1 | hit@3 | MRR | hit@3 literal | hit@3 paraphrase | Refused unanswerable | False refusals |",
             "|---|---|---|---|---|---|---|---|"]
    details = []
    for r in retrievers:
        rows = evaluate(r, questions)
        s = summarise(rows)
        lines.append(f"| {r.name} | {s['all']['hit@1']:.0%} | {s['all']['hit@3']:.0%} | {s['all']['mrr']:.2f} | "
                     f"{s['literal']['hit@3']:.0%} | {s['paraphrase']['hit@3']:.0%} | "
                     f"{s['refused']}/{s['n_unanswerable']} | {s['false_refusals']}/{s['all']['n']} |")
        details += [{"retriever": r.name, **row} for row in rows]

    report = "\n".join(lines)
    print(report)
    (HERE / "results.md").write_text(report + "\n")
    with open(HERE / "results_detail.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)


if __name__ == "__main__":
    main()
