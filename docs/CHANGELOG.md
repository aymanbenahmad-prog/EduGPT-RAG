# Changelog

## 1.0.0 (manuscript revision V9)

Replaces the six-file repository of 29 June 2026 (`edugpt_rag_complete.py`, `qasper_evaluate.py`,
`qasper_scoring.py`, `requirements_qasper.txt`, `README.md`, `LICENSE`).

Fixed relative to the previous code:
1. SQuAD 2.0 corpus: contexts were keyed by question id, so each paragraph was indexed once per
   question (142,192 entries). Paragraphs are now de-duplicated (20,233 unique, 20,238 chunks).
2. Configuration now equals the manuscript (all-mpnet-base-v2, ms-marco-MiniLM-L-12-v2,
   DeBERTa-v3-large NLI, IVF-PQ, GPT-3.5-turbo, full development split); previous defaults were
   MiniLM-L6, nli-deberta-v3-small, IVFFlat, a local Flan-T5 generator and 200 evaluation questions.
3. NLI scoring: the previous code called `float()` on a 3-logit array (crash) and used no softmax
   and a hard-coded label index. Probabilities are now softmax outputs and the entailment index is
   read from the model configuration.
4. Re-ranker training: the previous code trained on 1,000 SQuAD training pairs although the
   manuscript states MS MARCO; `training/train_reranker_msmarco.py` implements the stated setup.
5. QASPER: the previous script used the first 93 validation papers as "dev" and the remaining
   validation papers as the index. The new protocol uses the official v0.3 split.
6. README: removed pre-filled "expected results", the "replace with your actual measured
   results" note and reviewer-facing wording.
7. Added: baselines, ablations, sensitivity studies, latency, contamination probes, significance
   tests, independent metric check, human-evaluation kit, figure scripts, consistency checker,
   pinned environment, Dockerfile, tests.
