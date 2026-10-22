# EduGPT-RAG: reproducibility package

Code, configuration and documentation for the manuscript *"A Retrieval-Augmented with
Anti-Hallucination Verification for Academic Question Answering Using EduGPT-RAG Large
Language Model"* (revised version V9).

## Status

| Item | State |
|---|---|
| Pipeline (Algorithm 1): chunking, BM25, SBERT + FAISS IVF-PQ, RRF, cross-encoder, GPT-3.5-turbo with the Appendix A prompts, NLI verification with one constrained regeneration | implemented in `src/edugpt_rag/`, unit-tested with stub models (`pytest tests`) |
| SQuAD 2.0 retrieval corpus (Section 4.1) | built and verified: `data/processed/squad2/corpus_stats.json` |
| MS MARCO re-ranker fine-tuning | `training/train_reranker_msmarco.py` |
| Experiments for Tables 1 to 10 and Figures 3 to 9 | scripts in `experiments/`, `evaluation/`, `figures/` (map below) |
| Per-seed predictions and result files of the manuscript | **not available**: the outputs of the original runs were not archived |
| End-to-end execution of this package at full scale | **not yet performed**; it needs a CUDA GPU and an OpenAI API key |

The values in the manuscript were produced by the authors' original runs. Until this package is
executed end to end, it has not been confirmed that it reproduces those values. The manuscript
tables are transcribed in `results/reported/` (clearly separated from any re-run output), and
`tools/export_tables.py` prints the difference between a re-run and the manuscript.
`docs/KNOWN_ISSUES.md` lists every open point.

## Quick start

```bash
conda env create -f environment.yml && conda activate edugpt-rag     # or: docker build -t edugpt-rag .
export PYTHONPATH=src OPENAI_API_KEY=...
bash data/download.sh
python data/build_corpus_squad.py        # prints the corpus statistics used in Section 4.1
pytest -q tests                          # 17 tests, CPU only, no downloads
bash run_all.sh                          # every table and figure (GPU + API)
```

## Results map (manuscript item -> command -> output)

| Manuscript | Command | Output |
|---|---|---|
| Sec. 4.1 corpus counts | `python data/build_corpus_squad.py` | `data/processed/squad2/corpus_stats.json` |
| Re-ranker (Sec. 4.1) | `python training/train_reranker_msmarco.py --seed {42,123,456}` | `checkpoints/reranker_seed*/` |
| Table 1, EduGPT-RAG | `python experiments/run_squad.py --run_name edugpt_full` then `python evaluation/squad_eval.py --run edugpt_full` | `results/tables/edugpt_full.json` |
| Table 1, baselines | `python experiments/run_baselines.py --baseline <name>` (+ `squad_eval.py`) | `results/tables/<name>.json` |
| Table 2 (same backbone) | `python experiments/run_ablation.py --table 2` | `results/tables/t2_*.json` |
| Table 3 (retrieval) | from the Table 2 runs: `Recall@K_fused` (pre re-ranking), `Recall@K_context` (post) | same files |
| Table 4 (leave-one-out) | `python experiments/run_ablation.py --table 4` | `results/tables/t4_*.json` |
| Table 5 (hallucination) | `python evaluation/run_hallucination.py --run <name>` then `squad_eval.py` | `HallucinationRate`, `CitationAccuracy`, `Abstention_*` |
| Table 6 (chunking) | `python experiments/sensitivity.py --study chunk` | `results/tables/chunk_w*_s*.json` |
| Table 7 (embeddings) | `python experiments/sensitivity.py --study embedding` | `results/tables/emb_*.json` |
| Table 8 (latency) | `python experiments/latency_benchmark.py` | `results/tables/latency.json` |
| Tables 9 and 10 (tau) | `python experiments/sensitivity.py --study tau` | `results/tables/tau_*.json` |
| Sec. 4.2 significance | `python evaluation/significance.py --a edugpt_full --b self_rag` | `results/tables/significance_*.json` |
| Contamination probes | `python experiments/contamination_tests.py --probe {paraphrase,entity,numeric}` | `results/tables/contam_*.json` |
| Human evaluation (Sec. 4.5) | `evaluation/human_eval/` (protocol, sheet generator, analysis) | annotation data must be collected again |
| Independent metric check (R1-C2) | `python evaluation/independent_metric_check.py` | `results/tables/edugpt_full_independent.json` |
| Figures 3 to 9 | `python figures/make_figures.py --source {reported,rerun}` | `figures/out/<source>/` |
| Text-vs-table consistency | `python tools/check_consistency.py` | `docs/CONSISTENCY_REPORT.md` |

## Configuration

`configs/squad2.yaml` holds every hyperparameter stated in the manuscript (w = 512, s = 128,
k_d = k_s = 20, kappa = 60, n = 20, K = 5, tau = 0.75, IVF4096/PQ64/n_probe = 128,
BM25 k1 = 1.2 and b = 0.75, GPT-3.5-turbo with temperature 0.1 and max_tokens 512,
seeds 42/123/456). The NLI model is the public checkpoint
`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`; the identifier written in earlier
manuscript versions (`microsoft/deberta-v3-large-mnli-fever-anli`) does not exist on the
Hugging Face Hub. Any value can be overridden, e.g. `--set verifier.tau=0.85`.

## Data

| Dataset | Licence | Source |
|---|---|---|
| SQuAD 2.0 | CC BY-SA 4.0 | https://rajpurkar.github.io/SQuAD-explorer/ |
| MS MARCO passage ranking | MS MARCO terms | https://microsoft.github.io/msmarco/ |
| QASPER v0.3 (protocol only, no results in V9) | CC BY 4.0 | https://allenai.org/data/qasper |

Raw data are not redistributed; `data/download.sh` fetches them and records SHA-256 sums.

## Layout

```
configs/            squad2.yaml, qasper.yaml, baselines/baselines.yaml
src/edugpt_rag/     chunking, retrieval, rrf, reranker, generator, prompts, verifier, metrics, pipeline, build
data/               download.sh, build_corpus_squad.py, build_corpus_qasper.py, processed/squad2/corpus_stats.json
training/           train_reranker_msmarco.py
experiments/        run_squad, run_ablation, run_baselines, run_qasper, sensitivity, latency_benchmark, contamination_tests
evaluation/         squad_eval, run_hallucination, significance, independent_metric_check, human_eval/
results/            reported/ (manuscript values), raw/, tables/, logs/
figures/            make_figures.py
tools/              check_consistency.py, export_tables.py
tests/              unit and smoke tests
docs/               KNOWN_ISSUES.md, BASELINES.md, CONSISTENCY_REPORT.md, CHANGELOG.md
```

Licence: CC0 1.0 (see `LICENSE`). Citation: `CITATION.cff`.
