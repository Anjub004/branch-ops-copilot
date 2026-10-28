"""Rung 3: the stock agent. The model chooses tools; our code runs them, logs them and enforces limits."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from copilot import tools
from copilot.llm import chat

SYSTEM = (
    "You are a retail operations assistant for a supermarket chain. Today's data date is {as_of}.\n"
    "Rules:\n"
    "- Use tools for every number. Never guess stock, sales or prices.\n"
    "- If you only know a product's name, call find_products first to get its SKU.\n"
    "- Use calculate for any arithmetic.\n"
    "- For policy questions, use search_sops and cite the source file in [brackets].\n"
    "- If a tool returns an error, read it and try again with corrected input.\n"
    "- Answer briefly, with the key numbers and the branch name."
)

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "tool_calls.jsonl"


def log_call(entry, path=LOG_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def run_agent(task, tool_names=None, system=None, max_steps=8, log_path=LOG_PATH):
    """Returns {"answer", "steps", "tool_calls": [{"tool", "args", "ok", "ms"}], "stopped"}."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    messages = [{"role": "system", "content": system or SYSTEM.format(as_of=tools.as_of())},
                {"role": "user", "content": task}]
    trace = []
    for step in range(1, max_steps + 1):
        reply = chat(messages, tools=tools.schemas(tool_names))
        messages.append(reply.model_dump(exclude_none=True))
        if not reply.tool_calls:
            return {"answer": reply.content, "steps": step, "tool_calls": trace, "stopped": False}
        for call in reply.tool_calls:
            name, raw_args = call.function.name, call.function.arguments
            start, ok = time.perf_counter(), True
            try:
                if tool_names and name not in tool_names:
                    raise tools.ToolError(f"Tool '{name}' is not available here.")
                result = tools.call(name, raw_args)
            except (tools.ToolError, TypeError, ValueError) as e:  # bad tool, bad args, bad JSON
                ok, result = False, json.dumps({"error": str(e)})
            ms = round((time.perf_counter() - start) * 1000, 1)
            entry = {"run_id": run_id, "step": step, "tool": name, "args": raw_args, "ok": ok,
                     "ms": ms, "result_preview": result[:300], "task": task[:200]}
            log_call(entry, log_path)
            trace.append({"tool": name, "args": raw_args, "ok": ok, "ms": ms})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    return {"answer": "Stopped: step limit reached without a final answer.",
            "steps": max_steps, "tool_calls": trace, "stopped": True}
