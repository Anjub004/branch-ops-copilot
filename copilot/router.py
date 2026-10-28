"""Milestone 3: decide which rung a question deserves.

Routes
  rag           policy / procedure questions        -> retrieve SOPs, answer with citations
  agent         questions that need store numbers    -> tool-calling stock agent
  briefing      multi-part reports for a branch      -> agent team (M4); for now a briefing-mode agent
  out_of_scope  nothing to do with store operations  -> polite refusal, zero model calls

Three routers:
  RuleRouter     free and instant; says how sure it is
  LLMRouter      one small model call returning JSON; validated, with a safe fallback
  CascadeRouter  rules first, and only asks the model when the rules are unsure
"""
import json
import re
from dataclasses import dataclass
from functools import lru_cache

from copilot.llm import chat

ROUTES = ("rag", "agent", "briefing", "out_of_scope")


@dataclass
class RouteDecision:
    route: str
    confident: bool
    reason: str
    by: str   # "rules", "llm" or "fallback"


# ---------------------------------------------------------------- rules

BRIEFING = re.compile(
    r"\b(briefing|morning update|daily report|status report|situation at|summari[sz]e .*(today|branch)|"
    r"report (on|for)|overview (of|for))\b", re.I)

DATA = re.compile(
    r"\b(stock|in stock|on hand|left|sold|sell|selling|sales|units?|worth|value|run(ning)? (out|low)|"
    r"stock-?out|days of cover|purchase orders?|open orders?|an order|overdue|cost|price of|top \d+|"
    r"compare|how much .* (have|has|hold)|how many .* (sold|sell|left|have|has))\b|\bPO\d+\b|\b\d{4}\b", re.I)

POLICY = re.compile(
    r"\b(policy|policies|procedure|sop|allowed|permitted|rule|rules|approve|approval|sign off|"
    r"can (a|i|we|staff|employees|the)|should|must|what (do|should) (i|we)|how do (i|we)|"
    r"what happens|what is the process|who (is|can|must|needs)|when (is|are|do|must)|do (we|employees)|"
    r"accept|return|refund|complaint|uniform|leave|discount|temperature|expiry|sticker)\b", re.I)

DESTRUCTIVE = re.compile(r"\b(delete|remove|drop|wipe|clear|change|update|edit)\b.*\b(stock|records?|data|orders?|prices?)\b", re.I)


@lru_cache(maxsize=1)
def _branch_pattern():
    from copilot.tools import _query
    names = [r["name"] for r in _query("SELECT name FROM branches")]
    return re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b", re.I)


@lru_cache(maxsize=1)
def _sop_retriever():
    from copilot.tools import _sop_retriever as get
    return get()


class RuleRouter:
    """Cheap first pass. Returns confident=False whenever the signals are mixed or missing."""
    name = "rules"

    def decide(self, question):
        q = question.strip()
        branch = bool(_branch_pattern().search(q))
        data = bool(DATA.search(q))
        policy = bool(POLICY.search(q))

        if BRIEFING.search(q):
            return RouteDecision("briefing", True, "asks for a report/briefing", "rules")
        if DESTRUCTIVE.search(q):
            return RouteDecision("agent", True, "asks to change data: the read-only agent will decline", "rules")
        if (data or branch) and not policy:
            return RouteDecision("agent", True, "needs store numbers" + (" for a named branch" if branch else ""), "rules")
        if policy and not (data or branch):
            return RouteDecision("rag", True, "policy/procedure wording", "rules")
        if policy and (data or branch):
            return RouteDecision("agent", False, "mixed: policy wording plus data/branch signals", "rules")

        # No clear signal: does the question even look like our SOPs?
        retriever = _sop_retriever()
        if retriever.confident(retriever.search(q, 1)):
            return RouteDecision("rag", False, "no keywords, but the SOPs look relevant", "rules")
        return RouteDecision("out_of_scope", False, "no store-related signals", "rules")


# ---------------------------------------------------------------- LLM

LLM_PROMPT = """You route questions for a supermarket operations assistant. Reply with JSON only:
{"route": "<rag|agent|briefing|out_of_scope>", "reason": "<few words>"}

rag          = questions about store policies, rules or procedures (answered from SOP documents)
agent        = questions that need live store data: stock, sales, prices, purchase orders, branches;
               also requests to change data (the agent is read-only and will decline)
briefing     = requests for a report, summary or briefing about a branch with several parts
out_of_scope = anything unrelated to running the store (weather, sport, jokes, IT, share price)

Examples:
"Can a customer return bread bought yesterday?" -> {"route": "rag", "reason": "returns policy"}
"How much milk does Kitwe have?" -> {"route": "agent", "reason": "stock level"}
"Morning briefing for Ndola" -> {"route": "briefing", "reason": "branch report"}
"Tell me a joke" -> {"route": "out_of_scope", "reason": "not store related"}"""


class LLMRouter:
    name = "llm"

    def decide(self, question):
        try:
            reply = chat([{"role": "system", "content": LLM_PROMPT}, {"role": "user", "content": question}],
                         response_format={"type": "json_object"})
            data = json.loads(reply.content)
            route = str(data.get("route", "")).strip().lower()
            if route not in ROUTES:
                raise ValueError(f"invalid route {route!r}")
            return RouteDecision(route, True, str(data.get("reason", ""))[:120], "llm")
        except Exception as e:  # bad JSON, unknown route, server down: never crash the copilot
            return RouteDecision("rag", False, f"LLM router failed ({type(e).__name__}); defaulting to rag", "fallback")


class CascadeRouter:
    """Rules first. Pay for a model call only when the rules aren't sure."""
    name = "cascade"

    def __init__(self, rules=None, llm=None):
        self.rules, self.llm = rules or RuleRouter(), llm or LLMRouter()

    def decide(self, question):
        decision = self.rules.decide(question)
        if decision.confident:
            return decision
        second = self.llm.decide(question)
        if second.by == "fallback":  # model failed: keep the rules' best guess rather than a blind default
            return RouteDecision(decision.route, False, decision.reason + "; LLM unavailable", "rules")
        return second
