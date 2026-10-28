"""The front door: route the question, run the cheapest rung that can answer it, report the cost."""
import os
import time
from functools import lru_cache

from copilot import llm, rag
from copilot.agent import run_agent
from copilot.briefing import run_briefing
from copilot.costs import usd
from copilot.router import CascadeRouter, _branch_pattern

OUT_OF_SCOPE = ("I can help with store policies, stock, sales and purchase orders. "
                "That question is outside what I know about.")
WHICH_BRANCH = "Which branch should the briefing cover? For example: 'Morning briefing for Kitwe'."


@lru_cache(maxsize=1)
def sop_retriever():
    from copilot.tools import _sop_retriever
    return _sop_retriever()


def find_branch(question):
    match = _branch_pattern().search(question)
    return match.group(1) if match else None


def ask(question, router=None, force_route=None, briefing_mode=None):
    """Returns {"route", "decision", "answer", "sources", "cost", "briefing"?}."""
    llm.reset_stats()
    start = time.perf_counter()
    decision = None
    if force_route:
        route = force_route
    else:
        decision = (router or CascadeRouter()).decide(question)
        route = decision.route

    sources, extra = [], {}
    if route == "out_of_scope":
        answer = OUT_OF_SCOPE
    elif route == "rag":
        result = rag.answer(question, sop_retriever())
        answer, sources = result["answer"], result["sources"]
    elif route == "briefing":
        branch = find_branch(question)
        if not branch:
            answer = WHICH_BRANCH  # asking costs nothing; guessing a branch costs a whole team run
        else:
            result = run_briefing(branch, briefing_mode or os.getenv("BRIEFING_MODE", "team"), reset_stats=False)
            answer = result["briefing"]
            extra["briefing"] = {k: result[k] for k in ("mode", "unsupported_numbers", "coverage")}
            if result["unsupported_numbers"]:
                answer += ("\n\n⚠️ Fact-check: these numbers could not be traced to store data: "
                           + ", ".join(f"{n:,.10g}" for n in result["unsupported_numbers"]))
    else:
        answer = run_agent(question)["answer"]

    stats = dict(llm.STATS)  # includes the router's own model call, if it made one
    return {
        "route": route,
        "decision": decision.__dict__ if decision else {"route": route, "by": "forced"},
        "answer": answer,
        "sources": sources,
        **extra,
        "cost": {**stats, "usd": usd(stats["prompt_tokens"], stats["completion_tokens"]),
                 "seconds": round(time.perf_counter() - start, 2)},
    }
