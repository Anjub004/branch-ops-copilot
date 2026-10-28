"""Read-only tools the stock agent may call (Milestone 2).

Safety is enforced in code, not in the prompt:
  1. The database is opened in SQLite read-only mode (mode=ro).
  2. An authorizer rejects anything except reading (no INSERT/UPDATE/DELETE/ATTACH/PRAGMA writes).
  3. run_sql accepts a single SELECT statement and returns at most 50 rows.
  4. Bad input raises ToolError with a helpful message the agent can recover from.
"""
import ast
import inspect
import json
import operator
import re
import sqlite3
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "store.db"
MAX_ROWS = 50


class ToolError(Exception):
    """An error the model should see and react to (e.g. 'unknown branch, valid ones are ...')."""


# ---------------------------------------------------------------- database access

_ALLOWED = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}


def _readonly_authorizer(action, *_):
    return sqlite3.SQLITE_OK if action in _ALLOWED else sqlite3.SQLITE_DENY


def connect(path=None):
    path = Path(path or DB_PATH)
    if not path.exists():
        raise ToolError(f"{path.name} not found. Run: python make_data.py")
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    con.set_authorizer(_readonly_authorizer)
    return con


def _query(sql, params=()):
    with connect() as con:
        return [dict(r) for r in con.execute(sql, params).fetchall()]


@lru_cache(maxsize=1)
def as_of():
    return _query("SELECT value FROM meta WHERE key = 'as_of'")[0]["value"]


def _branch_id(branch):
    rows = _query("SELECT branch_id, name FROM branches")
    for r in rows:
        if branch.strip().lower() in (r["branch_id"].lower(), r["name"].lower()):
            return r["branch_id"], r["name"]
    raise ToolError(f"Unknown branch '{branch}'. Valid branches: {', '.join(r['name'] for r in rows)}")


def _sku(sku):
    rows = _query("SELECT sku FROM products WHERE sku = ?", (str(sku).strip(),))
    if not rows:
        raise ToolError(f"Unknown SKU '{sku}'. Use find_products to look up the SKU by name.")
    return rows[0]["sku"]


# ---------------------------------------------------------------- tool registry

TOOLS = {}
_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def tool(fn):
    """Register a function as a tool and build its JSON schema from the signature and docstring.

    Docstring format: first line = description; lines like 'name: text' = parameter descriptions.
    """
    doc = inspect.getdoc(fn) or ""
    lines = doc.splitlines()
    param_docs = dict(re.findall(r"^\s*(\w+):\s*(.+)$", doc, flags=re.M))
    props, required = {}, []
    for name, p in inspect.signature(fn).parameters.items():
        props[name] = {"type": _TYPES.get(p.annotation, "string"), "description": param_docs.get(name, name)}
        if p.default is inspect.Parameter.empty:
            required.append(name)
        else:
            props[name]["description"] += f" (default: {p.default})"
    fn.schema = {"type": "function", "function": {
        "name": fn.__name__, "description": lines[0] if lines else fn.__name__,
        "parameters": {"type": "object", "properties": props, "required": required}}}
    TOOLS[fn.__name__] = fn
    return fn


def schemas(names=None):
    return [TOOLS[n].schema for n in (names or TOOLS)]


# ---------------------------------------------------------------- the tools

@tool
def find_products(query: str):
    """Find products by name, department or SKU. Use this first when you only know a product's name.

    query: words from the product name (e.g. 'bread'), a department (e.g. 'Dairy') or a SKU
    """
    like = f"%{query.strip()}%"
    rows = _query("SELECT sku, name, department, unit_price FROM products "
                  "WHERE name LIKE ? OR department LIKE ? OR sku = ? ORDER BY sku LIMIT 10",
                  (like, like, query.strip()))
    if not rows:  # fall back to individual words, so 'white bread loaf' still finds 'White bread 700g'
        words = [w for w in re.findall(r"[a-z]+", query.lower()) if len(w) > 2]
        if words:
            where = " OR ".join("name LIKE ?" for _ in words)
            rows = _query(f"SELECT sku, name, department, unit_price FROM products WHERE {where} LIMIT 10",
                          tuple(f"%{w}%" for w in words))
    return rows


@tool
def get_stock(sku: str, branch: str):
    """Current stock of one SKU at one branch, with average daily sales (last 28 days) and days of cover.

    sku: product SKU, e.g. '2001'
    branch: branch name or ID, e.g. 'Kitwe' or 'B02'
    """
    branch_id, branch_name = _branch_id(branch)
    sku = _sku(sku)
    row = _query("""
        SELECT p.sku, p.name, s.qty_on_hand, p.unit_price,
               ROUND(s.qty_on_hand * p.unit_price, 2) AS stock_value,
               ROUND((SELECT SUM(qty) FROM sales x WHERE x.branch_id = s.branch_id AND x.sku = s.sku
                      AND x.sale_date >= date(?, '-28 days')) / 28.0, 1) AS avg_daily_sales
        FROM stock s JOIN products p USING (sku)
        WHERE s.branch_id = ? AND s.sku = ?""", (as_of(), branch_id, sku))[0]
    rate = row["avg_daily_sales"] or 0
    row["days_of_cover"] = round(row["qty_on_hand"] / rate, 1) if rate else None
    row["branch"] = branch_name
    return row


@tool
def stockout_risk(branch: str, max_days_cover: float = 3.0, department: str = ""):
    """List products at a branch that will run out soon (days of cover below a limit), most urgent first.

    branch: branch name or ID
    max_days_cover: flag items with fewer days of stock than this
    department: optional department filter, e.g. 'Bakery'
    """
    branch_id, branch_name = _branch_id(branch)
    rows = _query("""
        SELECT p.sku, p.name, p.department, s.qty_on_hand,
               ROUND(SUM(x.qty) / 28.0, 1) AS avg_daily_sales
        FROM stock s JOIN products p USING (sku)
        JOIN sales x ON x.branch_id = s.branch_id AND x.sku = s.sku AND x.sale_date >= date(?, '-28 days')
        WHERE s.branch_id = ? AND (? = '' OR p.department = ?)
        GROUP BY p.sku""", (as_of(), branch_id, department, department))
    for r in rows:
        r["days_of_cover"] = round(r["qty_on_hand"] / r["avg_daily_sales"], 1) if r["avg_daily_sales"] else None
    risky = [r for r in rows if r["days_of_cover"] is not None and r["days_of_cover"] < max_days_cover]
    return {"branch": branch_name, "as_of": as_of(), "items": sorted(risky, key=lambda r: r["days_of_cover"])}


@tool
def sales_summary(branch: str, days: int = 7, sku: str = ""):
    """Units and value sold at a branch over the last N days, per product (or for one SKU).

    branch: branch name or ID
    days: how many days to look back (1-28)
    sku: optional SKU to limit the summary to one product
    """
    branch_id, branch_name = _branch_id(branch)
    if not 1 <= int(days) <= 28:
        raise ToolError("days must be between 1 and 28")
    if sku:
        sku = _sku(sku)
    rows = _query("""
        SELECT p.sku, p.name, SUM(x.qty) AS units, ROUND(SUM(x.qty * p.unit_price), 2) AS value
        FROM sales x JOIN products p USING (sku)
        WHERE x.branch_id = ? AND x.sale_date >= date(?, ?) AND (? = '' OR p.sku = ?)
        GROUP BY p.sku ORDER BY value DESC""", (branch_id, as_of(), f"-{int(days)} days", sku, sku))
    total = round(sum(r["value"] for r in rows), 2)
    return {"branch": branch_name, "days": int(days), "as_of": as_of(), "total_value": total, "products": rows}


@tool
def open_purchase_orders(branch: str, sku: str = ""):
    """Open purchase orders for a branch (optionally one SKU), flagging any that are overdue.

    branch: branch name or ID
    sku: optional SKU filter
    """
    branch_id, branch_name = _branch_id(branch)
    if sku:
        sku = _sku(sku)
    rows = _query("""
        SELECT po.po_id, po.sku, p.name, po.qty, po.order_date, po.expected_date
        FROM purchase_orders po JOIN products p USING (sku)
        WHERE po.branch_id = ? AND po.status = 'open' AND (? = '' OR po.sku = ?)
        ORDER BY po.expected_date""", (branch_id, sku, sku))
    for r in rows:
        r["overdue"] = r["expected_date"] < as_of()
    return {"branch": branch_name, "as_of": as_of(), "orders": rows}


@tool
def search_sops(query: str):
    """Search the store's SOP documents (policies and procedures). Returns the top 3 paragraphs with sources.

    query: the policy question in plain words
    """
    retriever = _sop_retriever()
    hits = retriever.search(query, 3)
    return {"confident": retriever.confident(hits),
            "results": [{"source": h.chunk.source, "text": h.chunk.text} for h in hits]}


@lru_cache(maxsize=1)
def _sop_retriever():
    from copilot.chunking import load_chunks
    from copilot.retrievers import KeywordRetriever
    return KeywordRetriever(load_chunks(ROOT / "sops"))


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


@tool
def calculate(expression: str):
    """Exact arithmetic with + - * / and brackets. Use this instead of doing maths in your head.

    expression: e.g. '12 * 32.5'
    """
    def walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -walk(node.operand)
        raise ToolError("Only numbers, brackets and + - * / are allowed")
    try:
        return round(walk(ast.parse(expression, mode="eval").body), 4)
    except SyntaxError:
        raise ToolError(f"Could not parse expression: {expression}")


@tool
def run_sql(sql: str):
    """Last resort: run ONE read-only SELECT query on the store database (max 50 rows).
    Tables: branches(branch_id, name, region), products(sku, name, department, unit_price, perishable),
    stock(branch_id, sku, qty_on_hand), sales(sale_date, branch_id, sku, qty),
    purchase_orders(po_id, branch_id, sku, qty, order_date, expected_date, status). Dates are 'YYYY-MM-DD'.

    sql: a single SELECT statement
    """
    statement = sql.strip().rstrip(";")
    if ";" in statement:
        raise ToolError("Only one statement is allowed.")
    if not re.match(r"(?is)^\s*(select|with)\b", statement):
        raise ToolError("Only SELECT queries are allowed.")
    try:
        with connect() as con:
            cursor = con.execute(statement)
            rows = [dict(r) for r in cursor.fetchmany(MAX_ROWS + 1)]
    except sqlite3.DatabaseError as e:
        raise ToolError(f"SQL error: {e}")
    return {"rows": rows[:MAX_ROWS], "truncated": len(rows) > MAX_ROWS}


def call(name, arguments):
    """Run a tool by name with a JSON-string or dict of arguments. Always returns a JSON string."""
    if name not in TOOLS:
        raise ToolError(f"Unknown tool '{name}'. Available: {', '.join(TOOLS)}")
    args = json.loads(arguments or "{}") if isinstance(arguments, str) else (arguments or {})
    return json.dumps(TOOLS[name](**args), default=str)
