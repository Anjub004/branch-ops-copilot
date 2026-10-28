# Branch Ops Copilot

A retail operations assistant that answers store questions from SOP documents, looks up stock numbers, and writes a daily branch briefing — while always picking **the cheapest approach that works** (RAG → agent → agent team).

It's the capstone from my Medium article **[LLM, RAG, AI Agent or Agentic AI? Climb the Ladder One Rung at a Time](https://medium.com/@anjubarai004/llm-rag-ai-agent-or-agentic-ai-climb-the-ladder-one-rung-at-a-time-c43367483b7f)**. The four-rung teaching version lives in **[ai-ladder](https://github.com/Anjub004/ai-ladder)**.

> All documents and numbers in this repo are made up for teaching. No real company data.

## Roadmap

| Milestone | What it adds | Status |
|---|---|---|
| **M1 · ask-my-sop** | RAG over 26 SOPs, a 30-question eval set, keyword vs embeddings vs hybrid search | ✅ done |
| **M2 · stock-agent** | SQLite stock / sales / PO data, 8 read-only tools, logged tool calls, 10-scenario agent eval | ✅ done |
| M3 · router | Sends each question to the cheapest rung, with a cost report | ⏳ next |
| M4 · morning briefing | Planner → specialist agents → reviewer write a daily branch report | planned |
| M5 · app | Streamlit UI, architecture diagram, final write-up | planned |

---

## Milestone 1 — ask-my-sop

**Question:** a cashier asks *"Can a customer bring back a loaf of bread they bought yesterday?"* The policy talks about *perishable* and *bakery* items, not *bread* or *loaf*. Does search still find it?

Before tuning any prompt, M1 measures **retrieval** — because if the right paragraph never reaches the model, no prompt can save the answer.

### What's inside

```
sops/                 26 fictional retail SOPs (returns, refunds, cold chain, cash handling, …) → 86 paragraphs
eval/questions.jsonl  30 questions: 12 literal, 13 paraphrased, 5 with no answer in the docs
eval/run_eval.py      Runs every retriever, writes eval/results.md and eval/results_detail.csv
copilot/chunking.py   One paragraph = one chunk, with the document title attached for context
copilot/retrievers.py BM25 keyword search, embedding search, hybrid (Reciprocal Rank Fusion)
copilot/embedders.py  Offline spaCy word vectors, or a real embedding model through Ollama
copilot/rag.py        Retrieve → refuse if not confident → answer only from context, with citations
ask.py                Ask a question from the command line
tests/                Offline tests (fake model + fake embedder, no downloads)
```

### Results (measured, offline)

Retriever settings: top-3, BM25 confidence threshold 4.0, cosine threshold 0.55 (set by hand before the run, not tuned on this set). Embeddings here are **spaCy `en_core_web_md` averaged word vectors**, the only embedding model available offline in my build environment.

| Retriever | hit@1 | hit@3 | MRR | hit@3 literal | hit@3 paraphrase | Refused unanswerable | False refusals |
|---|---|---|---|---|---|---|---|
| keyword (BM25) | 64% | 76% | 0.69 | 100% | 54% | 1/5 | 6/25 |
| embeddings (spaCy word vectors) | 36% | 60% | 0.49 | 92% | 31% | 1/5 | 0/25 |
| hybrid (BM25 + embeddings, RRF) | 64% | 76% | 0.70 | 100% | 54% | 1/5 | 0/25 |

*hit@3 = the expected paragraph is in the top 3 results. MRR = mean of 1/rank. Full per-question output: [`eval/results_detail.csv`](eval/results_detail.csv).*

### What the numbers say

1. **Keyword search is perfect on literal questions and weak on paraphrases (100% → 54%).** "Loaf of bread" never finds "bakery". This is the classic RAG failure from the article, now measured.
2. **Cheap embeddings are not magic.** Averaged word vectors did *worse* than BM25 (60% vs 76% hit@3). Averaging blurs a sentence into "generic retail text", so many paragraphs look similar.
3. **Hybrid search kept BM25's accuracy and removed every false refusal (6 → 0).** Fusing two rankings is cheap insurance.
4. **Score thresholds can't spot "no answer".** Only 1 of 5 unanswerable questions was refused by any retriever: their top scores overlap with hard paraphrases. Refusing reliably needs a second check — the model must also be allowed to say "I don't know" (already in `rag.py`'s prompt), and M3 will add an explicit relevance check.

**Next experiment:** run the same eval with a real sentence-embedding model:

```bash
ollama pull nomic-embed-text
EMBEDDER=ollama python -m eval.run_eval
```

I'll add those numbers here once measured. Expect a big jump on paraphrases — that's what sentence embeddings are built for.

### Run it

```bash
git clone https://github.com/Anjub004/branch-ops-copilot.git
cd branch-ops-copilot
pip install -r requirements.txt

# Offline embeddings (used for the table above)
pip install spacy && python -m spacy download en_core_web_md
python -m eval.run_eval

# Ask a question (needs a chat model; any OpenAI-compatible server works)
ollama pull llama3.1:8b
python ask.py "Can a customer bring back a loaf of bread they bought yesterday?"

# Tests (no downloads, no API key)
pytest -q
```

Environment variables: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` (chat), `EMBEDDER=spacy|ollama`, `EMBED_MODEL` (default `nomic-embed-text`).

### Design choices

- **Measure retrieval separately from generation.** It's cheaper and tells you *where* a wrong answer came from.
- **Chunk = paragraph + document title.** Small enough to cite, with enough context to be found.
- **Refuse before calling the model** when retrieval isn't confident — no tokens spent on a guess.
- **Reciprocal Rank Fusion** combines rankings without having to compare BM25 scores with cosine similarities.
- **The eval set includes questions with no answer.** A system that never says "I don't know" will look great until it meets a real user.

---

## Milestone 2 — stock-agent

RAG can *read* about bread. M2 lets the copilot *count* it.

> *"Which items will run out at Kitwe in the next 3 days, and is anything already on order?"*

Answering that needs a stock lookup, sales history, a calculation and the purchase-order list — in an order nobody can fix in advance. That's the job of an **agent**: the model chooses the tools, our code runs them.

### What's inside

```
make_data.py          Builds store.db: 6 fictional branches, 27 products, 28 days of sales, stock, purchase orders
store.db              The generated database (deterministic, seed 42) — committed so you can start immediately
copilot/tools.py      8 read-only tools + a tiny registry that builds JSON schemas from function signatures
copilot/agent.py      Tool-calling loop: step limit, errors fed back to the model, every call logged to JSONL
agent_cli.py          Ask the agent a question (add --trace to see each tool call)
eval/agent_scenarios.jsonl  10 scenarios with known answers (incl. an unknown branch and a delete request)
eval/run_agent_eval.py      Scores a real model on those scenarios and checks the database is unchanged
tests/test_m2.py      19 offline tests with a scripted fake model
```

### The tools

| Tool | What it answers |
|---|---|
| `find_products(query)` | "What's the SKU for long-life milk?" |
| `get_stock(sku, branch)` | Stock on hand, value, average daily sales, **days of cover** |
| `stockout_risk(branch, max_days_cover, department)` | What runs out soon, most urgent first |
| `sales_summary(branch, days, sku)` | Units and value sold in the last N days |
| `open_purchase_orders(branch, sku)` | Open POs, with an **overdue** flag |
| `search_sops(query)` | The M1 retriever as a tool — the agent decides when policy matters |
| `calculate(expression)` | Exact arithmetic (no `eval()`) |
| `run_sql(sql)` | Last resort: one read-only `SELECT`, max 50 rows |

### Read-only by construction, not by politeness

Telling a model "please don't delete anything" is a suggestion. M2 makes writes impossible in code:

1. **The database is opened with SQLite `mode=ro`.**
2. **An authorizer callback denies every action except reading** — so even `WITH x AS (SELECT 1) DELETE FROM stock` fails.
3. **`run_sql` accepts one `SELECT` and returns at most 50 rows.**
4. **Errors are helpful.** Ask for "Mumbai" and the tool replies with the list of valid branches, so the agent can correct itself.

The tests try `DELETE`, `UPDATE`, `DROP`, `ATTACH`, `PRAGMA`, stacked statements and a CTE-wrapped delete. All are blocked.

### Every move is logged

Each tool call is appended to `logs/tool_calls.jsonl`:

```json
{"run_id": "20261028T101500123456", "step": 2, "tool": "get_stock",
 "args": "{\"sku\": \"1001\", \"branch\": \"Kitwe\"}", "ok": true, "ms": 1.8,
 "result_preview": "{\"sku\": \"1001\", \"name\": \"Long-life milk 1L\", \"qty_on_hand\": 12, ...}"}
```

When an answer is wrong, the log tells you whether the model picked the wrong tool, passed the wrong arguments, or misread a correct result.

### Story in the data

The generated data has a few planted situations for the agent to find. At Kitwe, white bread is at **0** and long-life milk has **12 units (0.3 days of cover)**. The milk order **PO0025** was due on 24 October and is **overdue**.

### Run it

```bash
python make_data.py                      # optional: store.db is already committed
python agent_cli.py --trace "Which items will run out at Kitwe in the next 3 days?"
python -m eval.run_agent_eval            # scores your model on the 10 scenarios
pytest -q                                # 26 offline tests (M1 + M2)
```

Tool calling needs a model that supports it, e.g. `llama3.1:8b` or `qwen2.5:7b` in Ollama.

**Agent eval results:** not measured yet. My build environment has no access to a model server, so the logic is covered by offline tests with a scripted fake model. Run `python -m eval.run_agent_eval` with your model, and the table in `eval/agent_results.md` will show passes, tools used, steps and tokens.

## License

MIT © Anju Barai
