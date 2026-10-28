"""Branch Ops Copilot — Streamlit app (Milestone 5).

    pip install streamlit
    streamlit run app.py

Four tabs:
  Ask        chat with the copilot; every answer shows its route, sources, fact-check and cost
  Briefing   morning briefing for any branch in template / single / team mode
  Scorecard  the measured results from every milestone (files in eval/)
  Tool log   every tool call the agents made (logs/tool_calls.jsonl)
"""
import json
from pathlib import Path

import streamlit as st

from copilot.briefing import run_briefing
from copilot.copilot import ask
from copilot.router import CascadeRouter, RuleRouter
from copilot.tools import _query, as_of

ROOT = Path(__file__).parent
ROUTE_BADGE = {"rag": "📄 RAG", "agent": "🛠️ Agent", "briefing": "🧑‍🤝‍🧑 Briefing", "out_of_scope": "🚫 Out of scope"}

st.set_page_config(page_title="Branch Ops Copilot", page_icon="🛒", layout="wide")

with st.sidebar:
    st.title("🛒 Branch Ops Copilot")
    st.caption(f"Fictional store data as of {as_of()}. Picks the cheapest rung that can answer.")
    router_name = st.selectbox("Router", ["cascade", "rules"], key="router",
                               help="cascade = free rules first, a model call only when the rules are unsure")
    briefing_mode = st.selectbox("Briefing mode", ["template", "single", "team"], key="briefing_mode",
                                 help="template = no AI (0 calls); team = 4 specialists + reviewer")
    st.markdown("**Try:**\n- Can a customer return bread bought yesterday?\n"
                "- Which items will run out at Kitwe in 3 days?\n- Morning briefing for Ndola\n- Tell me a joke")


def receipt(cost):
    tokens = cost["prompt_tokens"] + cost["completion_tokens"]
    return f"{cost['calls']} model calls · {tokens} tokens · ${cost['usd']:.4f} · {cost['seconds']}s"


tab_ask, tab_brief, tab_score, tab_log = st.tabs(["💬 Ask", "📋 Briefing", "📊 Scorecard", "🧾 Tool log"])

# ---------------------------------------------------------------- Ask
with tab_ask:
    st.session_state.setdefault("history", [])
    for item in st.session_state.history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.caption(f"{ROUTE_BADGE[item['route']]} · chosen by {item['decision'].get('by')} — "
                       f"{item['decision'].get('reason', '')}")
            st.markdown(item["answer"])
            if item["sources"]:
                st.caption("Sources: " + ", ".join(item["sources"]))
            st.caption("🧾 " + receipt(item["cost"]))

    question = st.chat_input("Ask about policies, stock, sales or orders…")
    if question:
        router = CascadeRouter() if router_name == "cascade" else RuleRouter()
        try:
            result = ask(question, router=router, briefing_mode=briefing_mode)
        except Exception as e:  # e.g. no model server running
            result = {"route": "out_of_scope", "decision": {"by": "error", "reason": type(e).__name__},
                      "answer": f"⚠️ Could not reach the model server ({e}). Rules-only and template features still work.",
                      "sources": [], "cost": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "usd": 0, "seconds": 0}}
        st.session_state.history.append({"question": question, **result})
        st.rerun()

# ---------------------------------------------------------------- Briefing
with tab_brief:
    branches = [r["name"] for r in _query("SELECT name FROM branches ORDER BY branch_id")]
    col1, col2 = st.columns([2, 1])
    branch = col1.selectbox("Branch", branches, key="brief_branch")
    mode = col2.selectbox("Mode", ["template", "single", "team"], key="brief_mode")
    if st.button("Write briefing", type="primary"):
        try:
            r = run_briefing(branch, mode)
            st.markdown(r["briefing"])
            m1, m2, m3 = st.columns(3)
            m1.metric("Coverage", f"{r['coverage']:.0%}")
            m2.metric("Unsupported numbers", len(r["unsupported_numbers"]))
            m3.metric("Model calls", r["cost"]["calls"])
            if r["unsupported_numbers"]:
                st.warning("Fact-check: could not trace " + ", ".join(f"{n:,.10g}" for n in r["unsupported_numbers"]))
            for f in r["findings"]:
                with st.expander(f"Specialist: {f['specialist']} ({f['tool_calls']} tool calls)"):
                    st.markdown(f["findings"])
            st.caption("🧾 " + receipt(r["cost"]))
        except Exception as e:
            st.error(f"Briefing failed: {e}. Template mode works without a model server.")

# ---------------------------------------------------------------- Scorecard
with tab_score:
    scorecard = ROOT / "SCORECARD.md"
    if scorecard.exists():
        st.markdown(scorecard.read_text())
    else:
        st.info("Run `python -m eval.scorecard` to build SCORECARD.md from the eval results.")

# ---------------------------------------------------------------- Tool log
with tab_log:
    log = ROOT / "logs" / "tool_calls.jsonl"
    if log.exists():
        rows = [json.loads(line) for line in log.read_text().splitlines()[-200:]]
        st.caption(f"Last {len(rows)} tool calls (newest last)")
        st.dataframe([{k: r.get(k) for k in ("run_id", "step", "tool", "args", "ok", "ms", "task")} for r in rows],
                     use_container_width=True)
    else:
        st.info("No tool calls yet. Ask the agent a stock question first.")
