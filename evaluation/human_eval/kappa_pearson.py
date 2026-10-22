"""Analyses completed human annotations (see rubric.md).

Input: annotations.csv with columns item_id, factual_accuracy, source_faithfulness, annotator
(one row per item and annotator) and key.csv from make_sheet.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def fleiss_kappa(table: np.ndarray) -> float:
    """table[i, j] = number of raters assigning category j to item i (equal raters per item)."""
    n = table.sum(axis=1)[0]
    p_j = table.sum(axis=0) / table.sum()
    P_i = ((table * (table - 1)).sum(axis=1)) / (n * (n - 1))
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    return float((P_bar - P_e) / (1 - P_e)) if P_e < 1 else 1.0


def main():
    from scipy.stats import pearsonr, spearmanr, wilcoxon
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--seed_dir", default="seed42")
    a = ap.parse_args()
    key = {r["item_id"]: r for r in csv.DictReader(open(a.key, encoding="utf-8"))}
    ratings = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(a.sheet, encoding="utf-8")):
        for scale in ("factual_accuracy", "source_faithfulness"):
            ratings[scale][r["item_id"]].append(int(r[scale]))
    out = {}
    for scale, items in ratings.items():
        ids = sorted(items, key=int)
        table = np.array([[items[i].count(c) for c in (1, 2, 3)] for i in ids])
        out[f"fleiss_kappa_{scale}"] = fleiss_kappa(table)
        by_sys = defaultdict(dict)
        for i in ids:
            by_sys[key[i]["system"]][key[i]["qid"]] = float(np.mean(items[i]))
        out[f"mean_{scale}"] = {s: float(np.mean(list(v.values()))) for s, v in by_sys.items()}
        systems = sorted(by_sys)
        if len(systems) == 2:
            q = sorted(set(by_sys[systems[0]]) & set(by_sys[systems[1]]))
            st = wilcoxon([by_sys[systems[0]][x] for x in q], [by_sys[systems[1]][x] for x in q])
            out[f"wilcoxon_{scale}"] = {"statistic": float(st.statistic), "p": float(st.pvalue), "n": len(q)}
        if scale == "source_faithfulness":
            from edugpt_rag.pipeline import read_jsonl
            for s, v in by_sys.items():
                f = Path(a.raw, s, a.seed_dir, "predictions.jsonl")
                if not f.exists():
                    continue
                nli = {r["id"]: r.get("faith") for r in read_jsonl(str(f))}
                q = [x for x in v if nli.get(x) is not None]
                if len(q) > 2:
                    hx, nx = [v[x] for x in q], [nli[x] for x in q]
                    out[f"corr_human_vs_nli_{s}"] = {"pearson": float(pearsonr(hx, nx)[0]),
                                                     "spearman": float(spearmanr(hx, nx)[0]), "n": len(q)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
