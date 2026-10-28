"""Ask the SOPs a question from the command line.

    python ask.py "Can a customer return bread bought yesterday?"
    python ask.py --retriever keyword "What is the cashier float?"
"""
import argparse

from copilot.chunking import load_chunks
from copilot.embedders import get_embedder
from copilot.rag import answer
from copilot.retrievers import EmbeddingRetriever, HybridRetriever, KeywordRetriever


def build(kind):
    chunks = load_chunks()
    keyword = KeywordRetriever(chunks)
    if kind == "keyword":
        return keyword
    embedding = EmbeddingRetriever(chunks, get_embedder())
    return embedding if kind == "embedding" else HybridRetriever(keyword, embedding)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--retriever", choices=["keyword", "embedding", "hybrid"], default="hybrid")
    args = parser.parse_args()
    result = answer(args.question, build(args.retriever))
    print(result["answer"])
    if result["sources"]:
        print("\nSources:", ", ".join(result["sources"]))
