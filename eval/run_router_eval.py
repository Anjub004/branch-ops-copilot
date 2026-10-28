"""How often does each router pick the right rung?

    python -m eval.run_router_eval                  # rules only: free, offline
    ROUTER=cascade python -m eval.run_router_eval   # rules + LLM for unsure cases (needs a chat model)
    ROUTER=llm python -m eval.run_router_eval       # LLM for everything

Reports accuracy, accuracy on the questions the rules were confident about, how many questions
would need a model call, and a confusion matrix (rows = correct route, columns = chosen route).
"""
import json
import os
from collections import Counter
from pathlib import Path

from copilot import llm
from copilot.router import ROUTES, CascadeRouter, LLMRouter, RuleRouter

HERE = Path(__file__).parent


def load():
    return [json.loads(l) for l in (HERE / "route_questions.jsonl").read_text().splitlines() if l.strip()]


def evaluate(router, questions):
    rows = []
    for q in questions:
        d = router.decide(q["question"])
        rows.append({**q, "chosen": d.route, "confident": d.confident, "by": d.by, "reason": d.reason})
    return rows


def report(rows, router_name):
    n = len(rows)
    correct = sum(r["chosen"] == r["route"] for r in rows)
    sure = [r for r in rows if r["confident"]]
    sure_ok = sum(r["chosen"] == r["route"] for r in sure)
    llm_calls = sum(r["by"] in ("llm", "fallback") for r in rows)
    lines = [f"Router: **{router_name}** on {n} labelled questions.", "",
             "| Metric | Value |", "|---|---|",
             f"| Accuracy (all questions) | {correct}/{n} = {correct / n:.0%} |",
             f"| Confident decisions | {len(sure)}/{n} = {len(sure) / n:.0%} |",
             f"| Accuracy when confident | {sure_ok}/{len(sure) or 1} = {sure_ok / (len(sure) or 1):.0%} |",
             f"| Router model calls | {llm_calls} |", "",
             "Confusion matrix (rows = correct, columns = chosen):", "",
             "| correct \\ chosen | " + " | ".join(ROUTES) + " |", "|---" * (len(ROUTES) + 1) + "|"]
    pairs = Counter((r["route"], r["chosen"]) for r in rows)
    for truth in ROUTES:
        lines.append(f"| {truth} | " + " | ".join(str(pairs.get((truth, c), 0)) for c in ROUTES) + " |")
    wrong = [r for r in rows if r["chosen"] != r["route"]]
    if wrong:
        lines += ["", "Mistakes:", ""] + [f"- `{r['id']}` {r['question']} → chose **{r['chosen']}**, "
                                          f"expected **{r['route']}** ({'sure' if r['confident'] else 'unsure'}: {r['reason']})"
                                          for r in wrong]
    return "\n".join(lines)


def main():
    kind = os.getenv("ROUTER", "rules")
    router = {"rules": RuleRouter, "llm": LLMRouter, "cascade": CascadeRouter}[kind]()
    llm.reset_stats()
    text = report(evaluate(router, load()), kind)
    print(text)
    (HERE / f"router_results_{kind}.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
