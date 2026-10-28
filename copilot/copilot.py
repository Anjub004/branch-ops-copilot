"""The front door: route the question, run the cheapest rung that can answer it, report the cost."""
import time
from functools import lru_cache

from copilot import llm, rag
from copilot.agent import SYSTEM as AGENT_SYSTEM
from copilot.agent import run_agent
from copilot.costs import usd
from copilot.router import CascadeRouter
from copilot.tools import as_of

OUT_OF_SCOPE = ("I can help with store policies, stock, sales and purchase orders. "
                "That question is outside what I know about.")

BRIEFING_SYSTEM = AGENT_SYSTEM + (
    "\nThe user wants a short branch briefing. Cover: items at stockout risk (stockout_risk), "
    "overdue purchase orders (open_purchase_orders) and last 7 days' sales (sales_summary). "
    "Use bullet points and keep it under 150 words."
)


@lru_cache(maxsize=1)
def sop_retriever():
    from copilot.tools import _sop_retriever
    return _sop_retriever()


def ask(question, router=None, force_route=None):
    """Returns {"route", "decision", "answer", "sources", "cost": {calls, prompt_tokens, completion_tokens, usd, seconds}}."""
    llm.reset_stats()
    start = time.perf_counter()
    decision = None
    if force_route:
        route = force_route
    else:
        decision = (router or CascadeRouter()).decide(question)
        route = decision.route

    sources = []
    if route == "out_of_scope":
        answer = OUT_OF_SCOPE
    elif route == "rag":
        result = rag.answer(question, sop_retriever())
        answer, sources = result["answer"], result["sources"]
    elif route == "briefing":  # M4 will replace this with planner -> specialists -> reviewer
        answer = run_agent(question, system=BRIEFING_SYSTEM.format(as_of=as_of()), max_steps=10)["answer"]
    else:
        answer = run_agent(question)["answer"]

    stats = dict(llm.STATS)
    return {
        "route": route,
        "decision": decision.__dict__ if decision else {"route": route, "by": "forced"},
        "answer": answer,
        "sources": sources,
        "cost": {**stats, "usd": usd(stats["prompt_tokens"], stats["completion_tokens"]),
                 "seconds": round(time.perf_counter() - start, 2)},
    }
