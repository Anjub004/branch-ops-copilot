"""Offline tests for Milestone 4: facts, template briefing, the fact-checker and the agent team."""
import json
import os
import sys
import threading
from types import SimpleNamespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from copilot import briefing, llm  # noqa: E402
from copilot.copilot import WHICH_BRANCH, ask  # noqa: E402


# ------------------------------------------------------------------ facts, template, checker

def test_facts_for_kitwe():
    f = briefing.gather_facts("kitwe")
    assert f["branch"] == "Kitwe"
    assert [i["name"] for i in f["stockout_risk"]][:2] == ["White bread 700g", "Long-life milk 1L"]
    assert {o["po_id"] for o in f["overdue_orders"]} == {"PO0005", "PO0025"}
    assert f["sales_prev_7d_total"] > 0 and f["sales_change_pct"] is not None


@pytest.mark.parametrize("branch", ["Lusaka Central", "Kitwe", "Ndola", "Livingstone", "Kabwe", "Chipata"])
def test_template_is_fully_supported_and_covered(branch):
    f = briefing.gather_facts(branch)
    text = briefing.template_briefing(f)
    assert briefing.verify(text, f) == []
    assert briefing.coverage(text, f) == 1.0


def test_verify_catches_invented_numbers():
    f = briefing.gather_facts("Kitwe")
    text = ("Sales were K99,999.00 (-1.7%). Milk: 12 units, 0.3 days. PO0025 for 300 units is late "
            "since 2026-10-24. Bread sold 4,512 loaves.")
    assert briefing.verify(text, f) == [4512.0, 99999.0]


def test_coverage_counts_missing_items():
    f = briefing.gather_facts("Kitwe")   # 3 at-risk products + 2 overdue POs = 5 must-mention items
    assert briefing.coverage("White bread is out. PO0025 is late.", f) == pytest.approx(2 / 5)


# ------------------------------------------------------------------ team with a thread-safe fake model

class FakeModel:
    """Answers by role: specialists reply at once with findings; the reviewer writes the briefing."""

    def __init__(self, review_text):
        self.review_text, self.lock, self.calls = review_text, threading.Lock(), []

    def create(self, model, messages, **kwargs):
        system = messages[0]["content"]
        tool_names = sorted(t["function"]["name"] for t in kwargs.get("tools", []))
        with self.lock:
            self.calls.append({"system": system, "tools": tool_names, "user": messages[-1]["content"]})
        if system.startswith("You are the reviewer"):
            content = self.review_text
        else:
            role = next((r for r in briefing.SPECIALISTS
                         if f"the {'purchasing' if r == 'orders' else r} specialist" in system.lower()
                         or (r == "policy" and "sop specialist" in system.lower())), "agent")
            content = f"{role} findings"
        msg = SimpleNamespace(content=content, tool_calls=None,
                              model_dump=lambda **k: {"role": "assistant", "content": content})
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(prompt_tokens=50, completion_tokens=10))


def use(monkeypatch, review_text):
    fake = FakeModel(review_text)
    monkeypatch.setattr(llm, "client", SimpleNamespace(chat=SimpleNamespace(completions=fake)))
    return fake


def test_team_runs_four_specialists_then_reviewer(monkeypatch):
    fake = use(monkeypatch, "Stock at risk: White bread 700g (0), Long-life milk 1L (12), Cola 2L. "
                            "Overdue: PO0005, PO0025.")
    r = briefing.run_briefing("Kitwe", "team")
    assert r["cost"]["calls"] == 5
    specialists = [c for c in fake.calls if not c["system"].startswith("You are the reviewer")]
    assert {tuple(c["tools"]) for c in specialists} == {
        tuple(sorted(t)) for t, _ in briefing.SPECIALISTS.values()}          # each got only its own tools
    reviewer = fake.calls[-1]
    assert reviewer["system"].startswith("You are the reviewer")
    assert '"PO0025"' in reviewer["user"] and "policy findings" in reviewer["user"]  # facts + all findings
    assert r["coverage"] == 1.0 and r["unsupported_numbers"] == []


def test_team_hallucination_is_flagged(monkeypatch):
    use(monkeypatch, "Kitwe sold K1,234,567 this week. White bread 700g is out.")
    r = briefing.run_briefing("Kitwe", "team")
    assert 1234567.0 in r["unsupported_numbers"]
    assert r["coverage"] < 1.0


def test_policy_specialist_is_told_about_problems(monkeypatch):
    fake = use(monkeypatch, "ok")
    briefing.run_briefing("Kitwe", "team")
    policy = next(c for c in fake.calls if "sop specialist" in c["system"].lower())
    assert "White bread 700g nearly out of stock" in policy["system"] and "PO0025" in policy["system"]
    assert policy["tools"] == ["search_sops"]


def test_single_mode_uses_one_agent(monkeypatch):
    fake = use(monkeypatch, "unused")
    r = briefing.run_briefing("Kitwe", "single")
    assert r["cost"]["calls"] == 1 and len(fake.calls[0]["tools"]) == 8


# ------------------------------------------------------------------ copilot front door

def test_briefing_without_branch_asks_instead_of_guessing(monkeypatch):
    fake = use(monkeypatch, "unused")
    r = ask("Prepare a morning briefing please.", force_route="briefing")
    assert r["answer"] == WHICH_BRANCH and r["cost"]["calls"] == 0 and fake.calls == []


def test_briefing_route_template_mode_is_free(monkeypatch):
    use(monkeypatch, "unused")
    r = ask("Prepare a morning briefing for Kitwe.", briefing_mode="template")
    assert r["route"] == "briefing" and r["cost"]["calls"] == 0
    assert "PO0025" in r["answer"] and r["briefing"]["unsupported_numbers"] == []


def test_fact_check_warning_is_shown(monkeypatch):
    use(monkeypatch, "Kitwe sold K1,234,567 this week.")
    r = ask("Morning briefing for Kitwe", force_route="briefing", briefing_mode="team")
    assert "Fact-check" in r["answer"] and "1,234,567" in r["answer"]
