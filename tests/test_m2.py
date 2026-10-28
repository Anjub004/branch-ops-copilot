"""Offline tests for Milestone 2: the read-only tools and the agent loop (fake model, no server)."""
import json
import os
import sqlite3
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import make_data  # noqa: E402
from copilot import agent, llm, tools  # noqa: E402
from eval.run_agent_eval import check  # noqa: E402
from openai.types.chat import ChatCompletionMessage  # noqa: E402
from openai.types.chat.chat_completion_message_tool_call import (  # noqa: E402
    ChatCompletionMessageToolCall, Function)


# ------------------------------------------------------------------ data and tools

def test_data_is_deterministic(tmp_path):
    def dump(path):
        con = sqlite3.connect(path)
        return [con.execute(f"SELECT * FROM {t} ORDER BY 1, 2").fetchall()
                for t in ("stock", "purchase_orders", "sales")]
    assert dump(make_data.build(tmp_path / "a.db")) == dump(make_data.build(tmp_path / "b.db"))


def test_known_numbers():
    assert tools.get_stock("2001", "Kitwe")["qty_on_hand"] == 0
    milk = tools.get_stock("1001", "kitwe")          # branch names are case-insensitive
    assert milk["stock_value"] == 390.0
    risky = [i["name"] for i in tools.stockout_risk("Kitwe", 3)["items"]]
    assert risky[:2] == ["White bread 700g", "Long-life milk 1L"]
    overdue = {o["po_id"] for o in tools.open_purchase_orders("B02")["orders"] if o["overdue"]}
    assert {"PO0025", "PO0005"} <= overdue


def test_find_products_falls_back_to_words():
    assert tools.find_products("white bread loaf")[0]["sku"] == "2001"
    assert {p["department"] for p in tools.find_products("Dairy")} == {"Dairy"}


def test_helpful_errors():
    with pytest.raises(tools.ToolError, match="Valid branches: Lusaka Central"):
        tools.get_stock("2001", "Mumbai")
    with pytest.raises(tools.ToolError, match="find_products"):
        tools.get_stock("9999", "Kitwe")
    with pytest.raises(tools.ToolError):
        tools.sales_summary("Kitwe", days=90)


@pytest.mark.parametrize("sql", [
    "DELETE FROM stock", "UPDATE stock SET qty_on_hand = 0", "DROP TABLE stock",
    "SELECT 1; DROP TABLE stock", "ATTACH 'x.db' AS x", "PRAGMA writable_schema = 1",
    "WITH x AS (SELECT 1) DELETE FROM stock",
])
def test_run_sql_cannot_write(sql):
    with pytest.raises(tools.ToolError):
        tools.run_sql(sql)


def test_connection_is_read_only_even_without_the_regex():
    with tools.connect() as con, pytest.raises(sqlite3.DatabaseError):
        con.execute("INSERT INTO meta VALUES ('x', 'y')")


def test_run_sql_limits_rows():
    result = tools.run_sql("SELECT * FROM sales")
    assert len(result["rows"]) == tools.MAX_ROWS and result["truncated"]


def test_calculator_and_sops():
    assert tools.calculate("(12 * 32.5) - 10") == 380.0
    with pytest.raises(tools.ToolError):
        tools.calculate("__import__('os')")
    hits = tools.search_sops("How many days to return non-perishable items?")
    assert hits["results"][0]["source"] == "returns_policy.md"


def test_schemas_come_from_signatures():
    s = {t["function"]["name"]: t["function"] for t in tools.schemas()}
    assert set(s) >= {"find_products", "get_stock", "stockout_risk", "sales_summary",
                      "open_purchase_orders", "search_sops", "calculate", "run_sql"}
    params = s["stockout_risk"]["parameters"]
    assert params["required"] == ["branch"]
    assert params["properties"]["max_days_cover"]["type"] == "number"
    assert "default: 3.0" in params["properties"]["max_days_cover"]["description"]


# ------------------------------------------------------------------ agent loop with a scripted fake model

class FakeCompletions:
    def __init__(self, script):
        self.script, self.seen = list(script), []

    def create(self, model, messages, **kwargs):
        self.seen.append(messages)
        item = self.script.pop(0)
        msg = item(messages) if callable(item) else item
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


def use(monkeypatch, script):
    fake = FakeCompletions(script)
    monkeypatch.setattr(llm, "client", SimpleNamespace(chat=SimpleNamespace(completions=fake)))
    return fake


def calls(*items):
    return ChatCompletionMessage(role="assistant", content=None, tool_calls=[
        ChatCompletionMessageToolCall(id=f"c{i}", type="function",
                                      function=Function(name=n, arguments=json.dumps(a)))
        for i, (n, a) in enumerate(items)])


def tool_results(messages):
    return [m["content"] for m in messages if m["role"] == "tool"]


def test_agent_looks_up_sku_then_stock_and_logs(monkeypatch, tmp_path):
    log = tmp_path / "calls.jsonl"
    use(monkeypatch, [
        calls(("find_products", {"query": "long-life milk"})),
        calls(("get_stock", {"sku": "1001", "branch": "Kitwe"})),
        lambda m: ChatCompletionMessage(role="assistant", content="Value: " + json.loads(tool_results(m)[-1])["stock_value"].__str__()),
    ])
    result = agent.run_agent("Stock value of long-life milk at Kitwe?", log_path=log)
    assert result["answer"] == "Value: 390.0"
    assert [c["tool"] for c in result["tool_calls"]] == ["find_products", "get_stock"]
    entries = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(entries) == 2 and entries[1]["ok"] and entries[1]["tool"] == "get_stock"


def test_agent_feeds_errors_back(monkeypatch, tmp_path):
    use(monkeypatch, [
        calls(("get_stock", {"sku": "2001", "branch": "Mumbai"}), ("no_such_tool", {}),
              ("run_sql", {"sql": "DELETE FROM stock"}), ("get_stock", {"sku": "2001"})),
        lambda m: ChatCompletionMessage(role="assistant", content=" | ".join(tool_results(m))),
    ])
    result = agent.run_agent("break things", log_path=tmp_path / "l.jsonl")
    assert "Valid branches" in result["answer"]
    assert "Unknown tool 'no_such_tool'" in result["answer"]
    assert "Only SELECT" in result["answer"]
    assert "missing 1 required positional argument" in result["answer"]
    assert [c["ok"] for c in result["tool_calls"]] == [False, False, False, False]


def test_agent_step_limit_and_tool_allowlist(monkeypatch, tmp_path):
    use(monkeypatch, [calls(("calculate", {"expression": "1+1"}))] * 3)
    result = agent.run_agent("loop", max_steps=3, log_path=tmp_path / "l.jsonl")
    assert result["stopped"] and result["steps"] == 3

    use(monkeypatch, [calls(("run_sql", {"sql": "SELECT 1"})),
                      lambda m: ChatCompletionMessage(role="assistant", content=tool_results(m)[0])])
    result = agent.run_agent("sql please", tool_names=["get_stock"], log_path=tmp_path / "l.jsonl")
    assert "not available" in result["answer"]


def test_scenario_checker():
    assert check("Kitwe holds K390.00 of milk", {"all": ["390"]})
    assert check("Total: 340,959.50 kwacha", {"any": ["340959", "340960"]})
    assert not check("I deleted them", {"any": ["read-only", "cannot"]})
    scenarios = [json.loads(l) for l in open("eval/agent_scenarios.jsonl")]
    assert len(scenarios) == 10
