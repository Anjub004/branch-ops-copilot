"""Score the stock agent on 10 scenarios with known answers (needs a chat model with tool calling).

    python -m eval.run_agent_eval
    LLM_MODEL=qwen2.5:7b python -m eval.run_agent_eval

A scenario passes when:
  - the answer contains the expected values ("all") or at least one of them ("any"), ignoring case and commas
  - at least one of the expected tools was called (if any are listed)
  - the database is unchanged afterwards (it is opened read-only, this double-checks)
"""
import hashlib
import json
from pathlib import Path

from copilot import llm
from copilot.agent import run_agent
from copilot.tools import DB_PATH

HERE = Path(__file__).parent


def db_fingerprint():
    return hashlib.sha256(DB_PATH.read_bytes()).hexdigest()


def check(answer, expect):
    text = (answer or "").lower().replace(",", "")
    if "all" in expect:
        return all(v.lower().replace(",", "") in text for v in expect["all"])
    return any(v.lower().replace(",", "") in text for v in expect["any"])


def main():
    scenarios = [json.loads(l) for l in (HERE / "agent_scenarios.jsonl").read_text().splitlines() if l.strip()]
    before = db_fingerprint()
    lines = ["| ID | Task | Pass | Tools used | Steps | Tokens |", "|---|---|---|---|---|---|"]
    passed = 0
    for s in scenarios:
        llm.reset_stats()
        result = run_agent(s["task"])
        used = [c["tool"] for c in result["tool_calls"]]
        ok = check(result["answer"], s["expect"]) and (not s["tools_any"] or any(t in used for t in s["tools_any"]))
        passed += ok
        tokens = llm.STATS["prompt_tokens"] + llm.STATS["completion_tokens"]
        lines.append(f"| {s['id']} | {s['task'][:55]} | {'✅' if ok else '❌'} | {', '.join(used) or '-'} | "
                     f"{result['steps']} | {tokens} |")
        print(f"{s['id']} {'PASS' if ok else 'FAIL'} :: {result['answer'][:150]!r}")
    unchanged = db_fingerprint() == before
    report = "\n".join([f"Model: {llm.MODEL}. Passed {passed}/{len(scenarios)}. Database unchanged: {unchanged}.", ""] + lines)
    print("\n" + report)
    (HERE / "agent_results.md").write_text(report + "\n")


if __name__ == "__main__":
    main()
