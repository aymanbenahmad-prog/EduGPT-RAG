# Release checklist

1. `pytest -q tests` passes (CPU).
2. `python data/build_corpus_squad.py` reproduces `data/processed/squad2/corpus_stats.json`
   (20,239 paragraphs, 20,233 unique, 20,238 chunks; SHA-256 of the SQuAD files included).
3. `bash run_all.sh` completed on: GPU ______, date ______, wall time ______, OpenAI cost ______.
4. `python tools/export_tables.py` output attached to the release notes (re-run minus manuscript).
5. `results/raw/**/predictions.jsonl`, `results/tables/*.json`, `checkpoints/reranker_seed*`
   (or download links) and `indexes/squad2/faiss_seed*/build_stats.json` uploaded with the release.
6. Tag `v1.0.0`, archive on Zenodo, add the DOI to the manuscript's Code Availability statement.

Status at packaging (19 October 2026): steps 1 and 2 done; steps 3 to 6 pending.
