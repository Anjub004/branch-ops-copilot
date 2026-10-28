"""Ask Branch Ops Copilot anything. It picks the cheapest rung that can answer.

    python copilot_cli.py "Can a customer return bread bought yesterday?"
    python copilot_cli.py "Which items will run out at Kitwe in the next 3 days?"
    python copilot_cli.py --route agent "..."     # force a rung, to compare costs
"""
import argparse

from copilot.copilot import ask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--route", choices=["rag", "agent", "briefing", "out_of_scope"])
    args = parser.parse_args()
    r = ask(args.question, force_route=args.route)
    d, c = r["decision"], r["cost"]
    print(f"[route: {r['route']} | by {d.get('by')} | {d.get('reason', '')}]\n")
    print(r["answer"])
    if r["sources"]:
        print("\nSources:", ", ".join(r["sources"]))
    print(f"\n[cost: {c['calls']} model calls, {c['prompt_tokens'] + c['completion_tokens']} tokens, "
          f"${c['usd']:.4f}, {c['seconds']}s]")
