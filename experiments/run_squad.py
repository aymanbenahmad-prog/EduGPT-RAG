"""Runs EduGPT-RAG (or one of its ablated variants) on the SQuAD 2.0 development split.

Produces, per seed, results/raw/<run_name>/seed<k>/predictions.jsonl plus the resolved config.
Metrics are computed separately by evaluation/squad_eval.py so that any system's predictions
are scored by the same code.

Example:
  python experiments/run_squad.py --config configs/squad2.yaml --run_name edugpt_full
  python experiments/run_squad.py --config configs/squad2.yaml --run_name no_rerank --set use_rerank=false
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edugpt_rag.build import build_system  # noqa: E402
from edugpt_rag.config import load_config, save_config  # noqa: E402
from edugpt_rag.pipeline import write_jsonl  # noqa: E402


def parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        cur = out
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = yaml.safe_load(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/squad2.yaml")
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--set", nargs="*", help="dotted overrides, e.g. verifier.tau=0.85")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--questions", help="alternative question file (e.g. contamination probes)")
    a = ap.parse_args()
    cfg = load_config(a.config, parse_overrides(a.set))
    seeds = a.seeds or cfg.seeds
    qfile = a.questions or str(Path(cfg.corpus_dir) / "dev_questions.jsonl")
    questions = [json.loads(l) for l in open(qfile, encoding="utf-8")]
    if cfg.max_eval_examples:
        questions = questions[: cfg.max_eval_examples]
    for seed in seeds:
        random.seed(seed)
        out_dir = Path(cfg.output_dir) / a.run_name / f"seed{seed}"
        save_config(cfg, str(out_dir / "config.yaml"))
        system = build_system(cfg, seed)
        rows = []
        for q in questions:
            t0 = time.perf_counter()
            rec = system.answer(q["id"], q["question"], seed=seed)
            rec["latency_ms"] = 1000 * (time.perf_counter() - t0)
            rec.update({"gold_answers": q["answers"], "unanswerable": q["unanswerable"],
                        "gold_doc_id": q["gold_doc_id"]})
            rows.append(rec)
        write_jsonl(rows, str(out_dir / "predictions.jsonl"))
        print(f"[{a.run_name}] seed {seed}: {len(rows)} predictions -> {out_dir}")


if __name__ == "__main__":
    main()
