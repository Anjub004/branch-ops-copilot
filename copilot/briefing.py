"""Milestone 4: the morning briefing, three ways — and a fact-checker that doesn't trust any of them.

  template  no AI at all: facts from the tools, poured into a fixed markdown template (0 model calls)
  single    one agent with every tool writes the whole briefing (what M3 did)
  team      four specialist agents work in parallel, then a reviewer writes the final briefing

Every mode starts from the same `facts` gathered directly by code, and every result is checked by
`verify()`: any number in the briefing that doesn't appear in the facts is flagged as unsupported.
"""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

from copilot import llm, tools
from copilot.agent import SYSTEM as AGENT_SYSTEM
from copilot.agent import run_agent
from copilot.costs import usd

# ---------------------------------------------------------------- facts (plain code, no model)


def gather_facts(branch, risk_days=3.0):
    """Everything a briefing may mention, fetched directly from the tools."""
    risk = tools.stockout_risk(branch, risk_days)
    orders = tools.open_purchase_orders(branch)
    this_week = tools.sales_summary(branch, 7)
    last_14 = tools.sales_summary(branch, 14)
    prev_week_total = round(last_14["total_value"] - this_week["total_value"], 2)
    change = round((this_week["total_value"] - prev_week_total) / prev_week_total * 100, 1) if prev_week_total else None
    return {
        "branch": risk["branch"],
        "as_of": risk["as_of"],
        "risk_days": risk_days,
        "stockout_risk": risk["items"],
        "overdue_orders": [o for o in orders["orders"] if o["overdue"]],
        "open_orders": [o for o in orders["orders"] if not o["overdue"]],
        "sales_7d_total": this_week["total_value"],
        "sales_prev_7d_total": prev_week_total,
        "sales_change_pct": change,
        "top_products_7d": this_week["products"][:5],
    }


# ---------------------------------------------------------------- mode 1: template (zero model calls)


def template_briefing(facts):
    f = facts
    lines = [f"**Morning briefing — {f['branch']} ({f['as_of']})**", ""]
    lines.append(f"**Stock at risk (under {f['risk_days']:g} days of cover):**")
    lines += [f"- {i['name']} ({i['sku']}): {i['qty_on_hand']} on hand, {i['days_of_cover']} days of cover"
              for i in f["stockout_risk"]] or ["- Nothing at risk."]
    lines += ["", "**Overdue purchase orders:**"]
    lines += [f"- {o['po_id']}: {o['qty']} × {o['name']}, expected {o['expected_date']}"
              for o in f["overdue_orders"]] or ["- None."]
    trend = "n/a" if f["sales_change_pct"] is None else f"{f['sales_change_pct']:+g}% vs previous 7 days"
    lines += ["", f"**Sales, last 7 days:** K{f['sales_7d_total']:,.2f} ({trend})", "",
              "**Top sellers:** " + ", ".join(f"{p['name']} (K{p['value']:,.2f})" for p in f["top_products_7d"])]
    return "\n".join(lines)


# ---------------------------------------------------------------- mode 2: one agent does everything

SINGLE_SYSTEM = AGENT_SYSTEM + (
    "\nWrite a short morning briefing for the branch manager: items at stockout risk (under 3 days of cover), "
    "overdue purchase orders, and last 7 days' sales. Bullet points, under 150 words, numbers only from tools."
)


def single_agent_briefing(branch):
    return run_agent(f"Prepare the morning briefing for {branch}.",
                     system=SINGLE_SYSTEM.format(as_of=tools.as_of()), max_steps=10)["answer"]


# ---------------------------------------------------------------- mode 3: the team

SPECIALISTS = {
    "stock": (["stockout_risk", "get_stock", "find_products", "calculate"],
              "You are the stock specialist. Find items at {branch} with under 3 days of cover. "
              "For each: name, quantity on hand, days of cover. Use tools; report facts only, no advice."),
    "orders": (["open_purchase_orders", "get_stock"],
               "You are the purchasing specialist. List overdue and upcoming purchase orders at {branch} "
               "with PO number, product, quantity and expected date. Use tools; facts only."),
    "sales": (["sales_summary", "calculate"],
              "You are the sales specialist. Report {branch}'s total sales value for the last 7 days, the change "
              "versus the 7 days before (use sales_summary with days=14 and subtract), and the top 3 products. "
              "Use calculate for maths; facts only."),
    "policy": (["search_sops"],
               "You are the SOP specialist. The branch {branch} has these problems today: {problems}. "
               "Search the SOPs and list at most 3 procedures staff must follow because of them, citing sources "
               "in [brackets]. If nothing relevant, say so."),
}

REVIEWER_SYSTEM = (
    "You are the reviewer. Combine the specialists' findings into ONE morning briefing for the branch manager.\n"
    "Rules: use only numbers that appear in the FACTS block (it is the source of truth). If a specialist's number "
    "disagrees with FACTS, use the FACTS number. Do not invent anything. Sections: Stock at risk, Overdue orders, "
    "Sales, What staff must do. Bullet points, under 180 words."
)


def _run_specialist(name, branch, problems):
    tool_names, prompt = SPECIALISTS[name]
    system = AGENT_SYSTEM.format(as_of=tools.as_of()) + "\n" + prompt.format(branch=branch, problems=problems)
    result = run_agent(f"Report for {branch}.", tool_names=tool_names, system=system, max_steps=6)
    return {"specialist": name, "findings": result["answer"], "tool_calls": len(result["tool_calls"])}


def team_briefing(branch, facts, parallel=True):
    problems = "; ".join(
        [f"{i['name']} nearly out of stock" for i in facts["stockout_risk"]][:3]
        + [f"purchase order {o['po_id']} overdue" for o in facts["overdue_orders"]][:2]) or "none"
    names = list(SPECIALISTS)
    if parallel:  # the specialists don't depend on each other, so run them at the same time
        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            findings = list(pool.map(lambda n: _run_specialist(n, branch, problems), names))
    else:
        findings = [_run_specialist(n, branch, problems) for n in names]
    review = llm.chat([
        {"role": "system", "content": REVIEWER_SYSTEM},
        {"role": "user", "content": f"FACTS:\n{json.dumps(facts, indent=1)}\n\nFINDINGS:\n{json.dumps(findings, indent=1)}"},
    ])
    return review.content, findings


# ---------------------------------------------------------------- the fact-checker

_NUMBER = re.compile(r"(?<![\d.,])\d[\d,]*(?:\.\d+)?")


def _numbers(text):
    out = set()
    for raw in _NUMBER.findall(text or ""):
        try:
            out.add(round(abs(float(raw.replace(",", ""))), 1))
        except ValueError:
            pass
    return out


def verify(briefing, facts):
    """Return numbers in the briefing that can't be traced to the facts.

    Dates are split into parts, small counts (0-10) are ignored because they're usually list positions
    or 'last 7 days', and a number also counts as supported if it matches a fact after rounding.
    """
    allowed = _numbers(json.dumps(facts).replace("-", " "))
    allowed |= {round(a) for a in allowed} | {float(n) for n in range(0, 11)}
    text = re.sub(r"\d{4}-\d{2}-\d{2}", lambda m: m.group(0).replace("-", " "), briefing or "")
    text = re.sub(r"\b(PO|SKU)\s?\d+", " ", text)  # PO numbers / SKUs are identifiers, checked by name
    return sorted(n for n in _numbers(text) if n not in allowed and round(n) not in allowed)


def coverage(briefing, facts):
    """Share of must-mention items (at-risk products, overdue POs) that the briefing actually mentions."""
    text = (briefing or "").lower()
    must = [i["name"].lower() for i in facts["stockout_risk"]] + [o["po_id"].lower() for o in facts["overdue_orders"]]
    if not must:
        return 1.0
    return round(sum(m in text or m.split()[0] in text for m in must) / len(must), 2)


# ---------------------------------------------------------------- entry point


def run_briefing(branch, mode="team", reset_stats=True):
    """Returns {"mode", "branch", "briefing", "unsupported_numbers", "coverage", "cost", "findings"}."""
    if reset_stats:
        llm.reset_stats()
    start = time.perf_counter()
    facts = gather_facts(branch)
    findings = []
    if mode == "template":
        text = template_briefing(facts)
    elif mode == "single":
        text = single_agent_briefing(facts["branch"])
    elif mode == "team":
        text, findings = team_briefing(facts["branch"], facts)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    stats = dict(llm.STATS)
    return {
        "mode": mode, "branch": facts["branch"], "briefing": text, "findings": findings,
        "unsupported_numbers": verify(text, facts), "coverage": coverage(text, facts),
        "cost": {**stats, "usd": usd(stats["prompt_tokens"], stats["completion_tokens"]),
                 "seconds": round(time.perf_counter() - start, 2)},
    }
