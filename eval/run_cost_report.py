"""Does routing actually save money? Run each question twice — routed, and forced through the agent —
and compare model calls, tokens, dollars and seconds. Needs a chat model with tool calling.

    python -m eval.run_cost_report
    PRICE_INPUT_PER_MTOK=0.25 PRICE_OUTPUT_PER_MTOK=1.25 python -m eval.run_cost_report
"""
from pathlib import Path

from copilot.copilot import ask

HERE = Path(__file__).parent

QUESTIONS = [
    "Can a customer return bread they bought yesterday?",
    "Who must approve refunds above 500 kwacha?",
    "What temperature must freezers be kept at?",
    "Which items will run out at Kitwe within the next 3 days?",
    "Are there any overdue purchase orders at Kitwe?",
    "What is the company's share price today?",
]


def main():
    lines = ["| Question | Route | Routed: calls / tokens / $ / s | Always-agent: calls / tokens / $ / s |",
             "|---|---|---|---|"]
    totals = {"routed": [0, 0, 0.0, 0.0], "agent": [0, 0, 0.0, 0.0]}
    for q in QUESTIONS:
        routed, forced = ask(q), ask(q, force_route="agent")
        cells = []
        for key, r in (("routed", routed), ("agent", forced)):
            c = r["cost"]
            tok = c["prompt_tokens"] + c["completion_tokens"]
            for i, v in enumerate((c["calls"], tok, c["usd"], c["seconds"])):
                totals[key][i] += v
            cells.append(f"{c['calls']} / {tok} / {c['usd']:.4f} / {c['seconds']}")
        lines.append(f"| {q[:50]} | {routed['route']} | {cells[0]} | {cells[1]} |")
    t_r, t_a = totals["routed"], totals["agent"]
    lines.append(f"| **Total** | | **{t_r[0]} / {t_r[1]} / {t_r[2]:.4f} / {t_r[3]:.1f}** | "
                 f"**{t_a[0]} / {t_a[1]} / {t_a[2]:.4f} / {t_a[3]:.1f}** |")
    report = "\n".join(lines)
    print(report)
    (HERE / "cost_report.md").write_text(report + "\n")


if __name__ == "__main__":
    main()
