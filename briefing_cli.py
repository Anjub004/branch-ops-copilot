"""Morning briefing for a branch, in three modes.

    python briefing_cli.py Kitwe                  # team (default): 4 specialists in parallel + reviewer
    python briefing_cli.py Kitwe --mode template  # no AI: facts in a fixed template, 0 model calls
    python briefing_cli.py Kitwe --mode single    # one agent does everything
"""
import argparse

from copilot.briefing import run_briefing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("branch")
    parser.add_argument("--mode", choices=["template", "single", "team"], default="team")
    args = parser.parse_args()
    r = run_briefing(args.branch, args.mode)
    print(r["briefing"])
    c = r["cost"]
    print(f"\n[mode: {r['mode']} | coverage {r['coverage']:.0%} | unsupported numbers: "
          f"{', '.join(f'{n:,.10g}' for n in r['unsupported_numbers']) or 'none'} | "
          f"{c['calls']} model calls, {c['prompt_tokens'] + c['completion_tokens']} tokens, ${c['usd']:.4f}, {c['seconds']}s]")
