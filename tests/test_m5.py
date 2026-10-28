"""Offline tests for Milestone 5: the Streamlit app and the scorecard (no model server needed)."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def app():
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    return at


def test_app_loads_with_four_tabs():
    at = app()
    assert [t.label for t in at.tabs] == ["💬 Ask", "📋 Briefing", "📊 Scorecard", "🧾 Tool log"]


def test_out_of_scope_question_needs_no_model():
    at = app()
    at.selectbox(key="router").set_value("rules").run()
    at.chat_input[0].set_value("Write me a poem about Mondays.").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "outside what I know about" in text
    assert any("0 model calls" in c.value for c in at.caption)


def test_template_briefing_in_the_app():
    at = app()
    at.selectbox(key="brief_branch").set_value("Kitwe")
    at.selectbox(key="brief_mode").set_value("template")
    at.button[0].click().run()
    assert not at.exception
    assert any("PO0025" in m.value for m in at.markdown)
    assert {m.label: m.value for m in at.metric}["Unsupported numbers"] == "0"


def test_scorecard_marks_unmeasured_sections(tmp_path, monkeypatch):
    from eval import scorecard
    monkeypatch.setattr(scorecard, "ROOT", tmp_path)
    scorecard.main()
    text = (tmp_path / "SCORECARD.md").read_text()
    assert "The ladder, in one table" in text
    assert "M1 · Retrieval" in text and "hit@3" in text                   # measured: included
    if not (scorecard.HERE / "agent_results.md").exists():
        assert "_Not measured yet._ Run `python -m eval.run_agent_eval`" in text  # never invents numbers
