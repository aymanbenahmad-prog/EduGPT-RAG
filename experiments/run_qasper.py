"""QASPER evaluation on the official development split (future work; no results in manuscript V9).

Requires data/build_corpus_qasper.py. Answers all 1,005 development questions with the same
pipeline and scores them with the official QASPER definitions: answer F1 per answer type
(extractive, abstractive, yes/no, unanswerable; max over annotators) and evidence-free scoring.
Outputs results/raw/qasper_<run>/seed<k>/predictions.jsonl and results/tables/qasper_<run>.json.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edugpt_rag.build import build_system  # noqa: E402
from edugpt_rag.config import load_config  # noqa: E402
from edugpt_rag.metrics import qasper_answer_type, qasper_gold_string, qasper_token_f1  # noqa: E402
from edugpt_rag.pipeline import read_jsonl, write_jsonl  # noqa: E402


def score(rows):
    by_type = defaultdict(list)
    for r in rows:
        pred = r["answer"] if r["answer"] else "Unanswerable"
        best, best_t = -1.0, "none"
        for g in r["gold"]:
            f = qasper_token_f1(pred, qasper_gold_string(g))
            if f > best:
                best, best_t = f, qasper_answer_type(g)
        by_type[best_t].append(best)
        by_type["all"].append(best)
    return {f"F1_{k}": 100 * sum(v) / len(v) for k, v in by_type.items()} | {
        f"n_{k}": len(v) for k, v in by_type.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qasper.yaml")
    ap.add_argument("--run_name", default="edugpt_full")
    a = ap.parse_args()
    cfg = load_config(a.config)
    qs = read_jsonl(str(Path(cfg.corpus_dir) / "dev_questions.jsonl"))
    per_seed = {}
    for seed in cfg.seeds:
        system = build_system(cfg, seed)
        rows = []
        for q in qs:
            rec = system.answer(q["id"], q["question"], seed=seed)
            rec.update({"paper_id": q["paper_id"], "gold": q["answers"]})
            rows.append(rec)
        out = Path(cfg.output_dir) / f"qasper_{a.run_name}" / f"seed{seed}"
        write_jsonl(rows, str(out / "predictions.jsonl"))
        per_seed[f"seed{seed}"] = score(rows)
    keys = per_seed[next(iter(per_seed))].keys()
    summary = {k: {"mean": st.mean(v[k] for v in per_seed.values()),
                   "std": st.stdev(v[k] for v in per_seed.values()) if len(per_seed) > 1 else 0.0} for k in keys}
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path(f"results/tables/qasper_{a.run_name}.json").write_text(json.dumps({"per_seed": per_seed, "summary": summary},
                                                                             indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
