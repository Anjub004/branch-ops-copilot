"""Ask the stock agent a question.

    python agent_cli.py "Which bakery items will run out at Kitwe in the next 3 days?"
    python agent_cli.py --trace "What's the stock value of long-life milk at Kitwe?"
"""
import argparse

from copilot.agent import run_agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--trace", action="store_true", help="print every tool call")
    args = parser.parse_args()
    result = run_agent(args.question)
    if args.trace:
        for i, c in enumerate(result["tool_calls"], 1):
            print(f"  {i}. {c['tool']}({c['args']}) {'ok' if c['ok'] else 'ERROR'} {c['ms']}ms")
        print()
    print(result["answer"])
