"""Rung 2: retrieve, then answer only from what was retrieved — or refuse."""
from copilot.llm import chat

SYSTEM = (
    "You are a retail operations assistant. Answer ONLY from the context below. "
    "Cite the source file in square brackets, e.g. [returns_policy.md]. "
    "If the context does not contain the answer, say you don't know.\n\nCONTEXT:\n{context}"
)
REFUSAL = "I couldn't find this in the SOP documents. Please check with your supervisor."


def answer(question, retriever, k=3):
    hits = retriever.search(question, k)
    if not retriever.confident(hits):
        return {"answer": REFUSAL, "sources": [], "refused": True}
    context = "\n".join(f"[{h.chunk.source}] {h.chunk.text}" for h in hits)
    reply = chat([
        {"role": "system", "content": SYSTEM.format(context=context)},
        {"role": "user", "content": question},
    ])
    return {"answer": reply.content, "sources": sorted({h.chunk.source for h in hits}), "refused": False}
