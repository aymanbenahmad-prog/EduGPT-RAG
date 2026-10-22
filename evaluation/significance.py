"""Paired two-tailed t-tests across seeds (Section 4.2) plus a paired bootstrap over questions.

The manuscript reports paired t-tests over the three seeds (df = 2). This script reproduces that
test from results/tables/<run>.json and, because three observations give little power, also
reports a paired bootstrap over development questions for EM and F1 (seed 42).

  python evaluation/significance.py --a edugpt_full --b self_rag
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edugpt_rag.metrics import squad_scores  # noqa: E402
from edugpt_rag.pipeline import read_jsonl  # noqa: E402

METRICS = ["EM", "F1", "Recall@5_context", "Faithfulness"]


def paired_t(a, b):
    from scipy.stats import ttest_rel
    t, p = ttest_rel(a, b)
    return {"t": float(t), "p": float(p), "df": len(a) - 1}


def paired_bootstrap(xa, xb, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    xa, xb = np.asarray(xa, float), np.asarray(xb, float)
    diff = xa - xb
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)])
    return {"mean_diff": float(diff.mean() * 100), "ci95": [float(np.percentile(boots, 2.5) * 100),
                                                          float(np.percentile(boots, 97.5) * 100)],
            "p_two_sided": float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--seed_dir", default="seed42")
    a = ap.parse_args()
    A = json.load(open(Path(a.tables, f"{a.a}.json")))["per_seed"]
    B = json.load(open(Path(a.tables, f"{a.b}.json")))["per_seed"]
    seeds = sorted(set(A) & set(B))
    out = {"seeds": seeds, "t_tests": {}}
    for m in METRICS:
        if all(m in A[s] and m in B[s] for s in seeds) and len(seeds) >= 2:
            out["t_tests"][m] = paired_t([A[s][m] for s in seeds], [B[s][m] for s in seeds])
    ra = {r["id"]: r for r in read_jsonl(str(Path(a.raw, a.a, a.seed_dir, "predictions.jsonl")))}
    rb = {r["id"]: r for r in read_jsonl(str(Path(a.raw, a.b, a.seed_dir, "predictions.jsonl")))}
    ids = sorted(set(ra) & set(rb))
    for m in ("em", "f1"):
        xa = [squad_scores(ra[i]["gold_answers"], ra[i].get("answer") or "")[m] for i in ids]
        xb = [squad_scores(rb[i]["gold_answers"], rb[i].get("answer") or "")[m] for i in ids]
        out[f"bootstrap_{m}"] = paired_bootstrap(xa, xb)
    Path(a.tables, f"significance_{a.a}_vs_{a.b}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
