"""Offline tests for Milestone 3: routing, the copilot front door, NOT_FOUND refusals and costs."""
import json
import os
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from copilot import llm  # noqa: E402
from copilot.copilot import OUT_OF_SCOPE, ask  # noqa: E402
from copilot.costs import usd  # noqa: E402
from copilot.router import ROUTES, CascadeRouter, LLMRouter, RouteDecision, RuleRouter  # noqa: E402
from eval.run_router_eval import evaluate, load  # noqa: E402


class FakeCompletions:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def create(self, model, messages, **kwargs):
        self.calls += 1
        content = self.replies.pop(0)
        if isinstance(content, Exception):
            raise content
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None,
                                                                                model_dump=lambda **k: {"role": "assistant", "content": content}))],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))


def use(monkeypatch, replies):
    fake = FakeCompletions(replies)
    monkeypatch.setattr(llm, "client", SimpleNamespace(chat=SimpleNamespace(completions=fake)))
    return fake


@pytest.mark.parametrize("question, route", [
    ("Which items will run out at Kitwe within 3 days?", "agent"),
    ("How much cooking oil does Chipata have left?", "agent"),
    ("Who is allowed to do a price override at the till?", "rag"),
    ("Prepare a morning briefing for Kitwe.", "briefing"),
    ("Delete all the bread stock records for Kitwe.", "agent"),
])
def test_rules_confident_cases(question, route):
    d = RuleRouter().decide(question)
    assert (d.route, d.confident, d.by) == (route, True, "rules")


def test_rules_admit_uncertainty():
    d = RuleRouter().decide("Write me a poem about Mondays.")
    assert d.route == "out_of_scope" and not d.confident


def test_llm_router_validates_output(monkeypatch):
    use(monkeypatch, ['{"route": "agent", "reason": "stock"}', '{"route": "astrology"}', "not json",
                      ConnectionError("server down")])
    router = LLMRouter()
    assert router.decide("q").route == "agent"
    for _ in range(3):  # invalid route, bad JSON, server error -> safe fallback, never a crash
        d = router.decide("q")
        assert d.by == "fallback" and d.route == "rag" and not d.confident


def test_cascade_only_pays_when_unsure(monkeypatch):
    fake = use(monkeypatch, ['{"route": "out_of_scope", "reason": "poetry"}'])
    router = CascadeRouter()
    assert router.decide("Which items will run out at Kitwe within 3 days?").by == "rules"
    assert fake.calls == 0
    d = router.decide("Write me a poem about Mondays.")
    assert (d.route, d.by) == ("out_of_scope", "llm") and fake.calls == 1


def test_cascade_keeps_rule_guess_if_llm_is_down(monkeypatch):
    use(monkeypatch, [ConnectionError("down")])
    d = CascadeRouter().decide("Write me a poem about Mondays.")
    assert (d.route, d.by) == ("out_of_scope", "rules")


def test_out_of_scope_costs_nothing(monkeypatch):
    fake = use(monkeypatch, [])
    r = ask("anything", router=SimpleNamespace(decide=lambda q: RouteDecision("out_of_scope", True, "x", "rules")))
    assert r["answer"] == OUT_OF_SCOPE and r["cost"]["calls"] == 0 and fake.calls == 0


def test_rag_route_with_citation_and_cost(monkeypatch):
    monkeypatch.setenv("PRICE_INPUT_PER_MTOK", "1.0")
    monkeypatch.setenv("PRICE_OUTPUT_PER_MTOK", "5.0")
    use(monkeypatch, ["Same day only, if spoiled [returns_policy.md]"])
    r = ask("Who must approve refunds above 500 kwacha?", router=RuleRouter())
    assert r["route"] == "rag" and "refund_approval.md" in r["sources"]
    assert r["cost"]["calls"] == 1 and r["cost"]["usd"] == pytest.approx(100 / 1e6 + 20 * 5 / 1e6)


def test_model_not_found_becomes_refusal(monkeypatch):
    use(monkeypatch, ["NOT_FOUND"])
    r = ask("Who must approve refunds above 500 kwacha?", force_route="rag")
    assert r["sources"] == [] and "couldn't find" in r["answer"]


def test_usd_defaults_to_free(monkeypatch):
    monkeypatch.delenv("PRICE_INPUT_PER_MTOK", raising=False)
    monkeypatch.delenv("PRICE_OUTPUT_PER_MTOK", raising=False)
    assert usd(1_000_000, 1_000_000) == 0


def test_route_eval_set_and_rule_baseline():
    questions = load()
    assert len(questions) == 55 and {q["route"] for q in questions} == set(ROUTES)
    rows = evaluate(RuleRouter(), questions)
    sure = [r for r in rows if r["confident"]]
    # Regression guard for the README numbers: confident on 80%, right 95% of those.
    assert len(sure) == 44 and sum(r["chosen"] == r["route"] for r in sure) == 42
