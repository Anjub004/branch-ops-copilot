"""Collect every milestone's measured results into SCORECARD.md.

    python -m eval.scorecard

Sections whose results haven't been produced yet say so, with the command that produces them —
the scorecard never shows a number that wasn't measured.
"""
from datetime import date
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent

SECTIONS = [
    ("M1 · Retrieval: keyword vs embeddings vs hybrid", ["results.md"], "python -m eval.run_eval"),
    ("M2 · Stock agent: 10 scenarios with known answers", ["agent_results.md"], "python -m eval.run_agent_eval"),
    ("M3 · Router accuracy", ["router_results_rules.md", "router_results_cascade.md", "router_results_llm.md"],
     "ROUTER=cascade python -m eval.run_router_eval"),
    ("M3 · Cost: routed vs always-agent", ["cost_report.md"], "python -m eval.run_cost_report"),
    ("M4 · Briefing: template vs single agent vs team",
     ["briefing_results_template_single_team.md", "briefing_results_template.md"], "python -m eval.run_briefing_eval"),
]

SUMMARY = """## The ladder, in one table

| Rung | Used for | Model calls per answer | Main risk | Guard in this repo |
|---|---|---|---|---|
| No AI (rules, templates) | Routing obvious questions, fixed-format briefings | 0 | Can't handle unexpected wording | Rules report when they're unsure |
| RAG | Policy and procedure questions | 1 | Wrong paragraph retrieved; answering without evidence | Retrieval eval, refuse-before-calling, `NOT_FOUND` |
| Agent | Questions that need store numbers | 1–8 | Wrong tool or arguments; unsafe actions | Read-only DB + authorizer, step limit, tool log |
| Agent team | Multi-part briefings with judgement | 5+ | Cost, invented numbers, omissions | Code-built facts, fact-checker, coverage check |
"""


def main():
    parts = [f"# Scorecard — Branch Ops Copilot", "",
             f"_Generated {date.today().isoformat()} from the files in `eval/`. All data is fictional._", "", SUMMARY]
    for title, files, command in SECTIONS:
        parts += [f"## {title}", ""]
        found = [f for f in files if (HERE / f).exists()]
        if not found:
            parts += [f"_Not measured yet._ Run `{command}` (needs a model server unless noted in the README).", ""]
            continue
        for f in found:
            if len(found) > 1:
                parts += [f"**{f}**", ""]
            parts += [(HERE / f).read_text().strip(), ""]
            if "briefing_results_template_single_team.md" in found:
                break  # the full comparison already includes the template rows
    (ROOT / "SCORECARD.md").write_text("\n".join(parts))
    print("Wrote SCORECARD.md")


if __name__ == "__main__":
    main()
