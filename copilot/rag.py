"""Rung 2: retrieve, then answer only from what was retrieved — or refuse.

Two refusal checks:
  1. Before the model: if retrieval isn't confident, refuse without spending a single token.
  2. Inside the model call: the model must reply NOT_FOUND when the context doesn't answer
     the question (added in M3, because M1 showed retrieval scores alone can't spot "no answer").
"""
from copilot.llm import chat

SYSTEM = (
    "You are a retail operations assistant. Answer ONLY from the context below. "
    "Cite the source file in square brackets, e.g. [returns_policy.md]. "
    "If the context does not answer the question, reply with exactly: NOT_FOUND\n\nCONTEXT:\n{context}"
)
REFUSAL = "I couldn't find this in the SOP documents. Please check with your supervisor."


def answer(question, retriever, k=3):
    hits = retriever.search(question, k)
    if not retriever.confident(hits):
        return {"answer": REFUSAL, "sources": [], "refused": True, "refused_by": "retrieval"}
    context = "\n".join(f"[{h.chunk.source}] {h.chunk.text}" for h in hits)
    reply = chat([
        {"role": "system", "content": SYSTEM.format(context=context)},
        {"role": "user", "content": question},
    ])
    text = (reply.content or "").strip()
    if not text or "NOT_FOUND" in text:
        return {"answer": REFUSAL, "sources": [], "refused": True, "refused_by": "model"}
    return {"answer": text, "sources": sorted({h.chunk.source for h in hits}), "refused": False}
