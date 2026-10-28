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
