Evaluated on 30 questions over 86 chunks.

| Retriever | hit@1 | hit@3 | MRR | hit@3 literal | hit@3 paraphrase | Refused unanswerable | False refusals |
|---|---|---|---|---|---|---|---|
| keyword (BM25) | 64% | 76% | 0.69 | 100% | 54% | 1/5 | 6/25 |
| embeddings (spacy-word-vectors) | 36% | 60% | 0.49 | 92% | 31% | 1/5 | 0/25 |
| hybrid (BM25 + embeddings, RRF) | 64% | 76% | 0.70 | 100% | 54% | 1/5 | 0/25 |
