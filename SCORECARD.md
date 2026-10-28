# Scorecard — Branch Ops Copilot

_Generated 2026-10-28 from the files in `eval/`. All data is fictional._

## The ladder, in one table

| Rung | Used for | Model calls per answer | Main risk | Guard in this repo |
|---|---|---|---|---|
| No AI (rules, templates) | Routing obvious questions, fixed-format briefings | 0 | Can't handle unexpected wording | Rules report when they're unsure |
| RAG | Policy and procedure questions | 1 | Wrong paragraph retrieved; answering without evidence | Retrieval eval, refuse-before-calling, `NOT_FOUND` |
| Agent | Questions that need store numbers | 1–8 | Wrong tool or arguments; unsafe actions | Read-only DB + authorizer, step limit, tool log |
| Agent team | Multi-part briefings with judgement | 5+ | Cost, invented numbers, omissions | Code-built facts, fact-checker, coverage check |

## M1 · Retrieval: keyword vs embeddings vs hybrid

Evaluated on 30 questions over 86 chunks.

| Retriever | hit@1 | hit@3 | MRR | hit@3 literal | hit@3 paraphrase | Refused unanswerable | False refusals |
|---|---|---|---|---|---|---|---|
| keyword (BM25) | 64% | 76% | 0.69 | 100% | 54% | 1/5 | 6/25 |
| embeddings (spacy-word-vectors) | 36% | 60% | 0.49 | 92% | 31% | 1/5 | 0/25 |
| hybrid (BM25 + embeddings, RRF) | 64% | 76% | 0.70 | 100% | 54% | 1/5 | 0/25 |

## M2 · Stock agent: 10 scenarios with known answers

_Not measured yet._ Run `python -m eval.run_agent_eval` (needs a model server unless noted in the README).

## M3 · Router accuracy

Router: **rules** on 55 labelled questions.

| Metric | Value |
|---|---|
| Accuracy (all questions) | 48/55 = 87% |
| Confident decisions | 44/55 = 80% |
| Accuracy when confident | 42/44 = 95% |
| Router model calls | 0 |

Confusion matrix (rows = correct, columns = chosen):

| correct \ chosen | rag | agent | briefing | out_of_scope |
|---|---|---|---|---|
| rag | 26 | 4 | 0 | 0 |
| agent | 0 | 15 | 0 | 0 |
| briefing | 0 | 0 | 5 | 0 |
| out_of_scope | 3 | 0 | 0 | 2 |

Mistakes:

- `r01` How many days do customers have to return non-perishable items? → chose **agent**, expected **rag** (unsure: mixed: policy wording plus data/branch signals)
- `r08` How often is a full stock take done? → chose **agent**, expected **rag** (sure: needs store numbers)
- `r09` Which purchase orders need head office buyer approval? → chose **agent**, expected **rag** (unsure: mixed: policy wording plus data/branch signals)
- `r19` The yoghurt fridge has been warm all afternoon. What should we do with the stock? → chose **agent**, expected **rag** (unsure: mixed: policy wording plus data/branch signals)
- `r51` What is the company's share price today? → chose **rag**, expected **out_of_scope** (unsure: no keywords, but the SOPs look relevant)
- `r52` How do I reset the Wi-Fi password in the back office? → chose **rag**, expected **out_of_scope** (sure: policy/procedure wording)
- `r55` Who won the football match last night? → chose **rag**, expected **out_of_scope** (unsure: no keywords, but the SOPs look relevant)

## M3 · Cost: routed vs always-agent

_Not measured yet._ Run `python -m eval.run_cost_report` (needs a model server unless noted in the README).

## M4 · Briefing: template vs single agent vs team

| Mode | Branch | Coverage | Unsupported numbers | Model calls | Tokens | $ | Seconds |
|---|---|---|---|---|---|---|---|
| template | Lusaka Central | 100% | 0 | 0 | 0 | 0.0000 | 0.0 |
| template | Kitwe | 100% | 0 | 0 | 0 | 0.0000 | 0.0 |
| template | Ndola | 100% | 0 | 0 | 0 | 0.0000 | 0.0 |
| template | Livingstone | 100% | 0 | 0 | 0 | 0.0000 | 0.0 |
| template | Kabwe | 100% | 0 | 0 | 0 | 0.0000 | 0.0 |
| template | Chipata | 100% | 0 | 0 | 0 | 0.0000 | 0.0 |

| Mode | Runs | Avg coverage | Total unsupported numbers | Avg calls | Avg tokens | Avg seconds |
|---|---|---|---|---|---|---|
| template | 6 | 100% | 0 | 0.0 | 0 | 0.0 |
