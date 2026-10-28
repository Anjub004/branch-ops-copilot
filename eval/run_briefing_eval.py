"""Is the agent team worth its bill? Compare three briefing modes on every branch.

    python -m eval.run_briefing_eval                     # all modes (single/team need a chat model)
    MODES=template python -m eval.run_briefing_eval      # offline baseline only

Columns:
  coverage     share of must-mention items (at-risk products + overdue POs) that the briefing mentions
  unsupported  numbers in the briefing that can't be traced to the facts (possible hallucinations)
"""
import os
from pathlib import Path

from copilot.briefing import run_briefing
from copilot.tools import _query

HERE = Path(__file__).parent


def main():
    modes = os.getenv("MODES", "template,single,team").split(",")
    branches = [r["name"] for r in _query("SELECT name FROM branches ORDER BY branch_id")]
    lines = ["| Mode | Branch | Coverage | Unsupported numbers | Model calls | Tokens | $ | Seconds |",
             "|---|---|---|---|---|---|---|---|"]
    summary = {}
    for mode in modes:
        for branch in branches:
            try:
                r = run_briefing(branch, mode)
            except Exception as e:  # keep going: one failed run shouldn't hide the others
                lines.append(f"| {mode} | {branch} | ERROR: {type(e).__name__} | | | | | |")
                continue
            c = r["cost"]
            tokens = c["prompt_tokens"] + c["completion_tokens"]
            lines.append(f"| {mode} | {branch} | {r['coverage']:.0%} | {len(r['unsupported_numbers'])} | "
                         f"{c['calls']} | {tokens} | {c['usd']:.4f} | {c['seconds']} |")
            s = summary.setdefault(mode, [0, 0.0, 0, 0, 0, 0.0])
            s[0] += 1; s[1] += r["coverage"]; s[2] += len(r["unsupported_numbers"]); s[3] += c["calls"]
            s[4] += tokens; s[5] += c["seconds"]
    lines += ["", "| Mode | Runs | Avg coverage | Total unsupported numbers | Avg calls | Avg tokens | Avg seconds |",
              "|---|---|---|---|---|---|---|"]
    for mode, (n, cov, bad, calls, tok, sec) in summary.items():
        lines.append(f"| {mode} | {n} | {cov / n:.0%} | {bad} | {calls / n:.1f} | {tok / n:.0f} | {sec / n:.1f} |")
    report = "\n".join(lines)
    print(report)
    (HERE / f"briefing_results_{'_'.join(modes)}.md").write_text(report + "\n")


if __name__ == "__main__":
    main()
