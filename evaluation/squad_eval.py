"""Scores prediction files of any system on SQuAD 2.0 (Tables 1 to 5, 9, 10).

Reads results/raw/<run>/seed*/predictions.jsonl and writes results/tables/<run>.json with the
per-seed values, mean and sample standard deviation of:
  EM, F1 (official SQuAD 2.0 normalisation), Recall@{1,3,5,10} and MRR over the fused candidate
  list (pre-re-ranking) and over the final context list (post-re-ranking), Faithfulness (Eq. 15,
  mean over non-abstained answers), Hallucination rate (all questions), Citation accuracy (answers
  with a cited passage), abstention rate / precision / recall / F1, regeneration rate and latency.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edugpt_rag.metrics import (abstention_metrics, aggregate_squad, reciprocal_rank,  # noqa: E402
                                recall_at_k, squad_scores)
from edugpt_rag.pipeline import read_jsonl  # noqa: E402


def score_rows(rows):
    per = [squad_scores(r["gold_answers"], r.get("answer") or "") for r in rows]
    m = aggregate_squad(per)
    out = {"EM": m["EM"], "F1": m["F1"], "n": m["n"]}
    ans = [r for r in rows if r.get("gold_doc_id")]   # every SQuAD 2.0 question has a source paragraph
    for key, field in [("fused", "fused_doc_ids"), ("context", "context_doc_ids")]:
        if all(field in r for r in ans) and ans:
            for k in (1, 3, 5, 10):
                out[f"Recall@{k}_{key}"] = 100.0 * sum(recall_at_k(r[field], r["gold_doc_id"], k) for r in ans) / len(ans)
            out[f"MRR_{key}"] = sum(reciprocal_rank(r[field], r["gold_doc_id"]) for r in ans) / len(ans)
    faith = [r["faith"] for r in rows if r.get("faith") is not None and not r.get("abstained")]
    if faith:
        out["Faithfulness"] = sum(faith) / len(faith)
    if all("hallucinated" in r for r in rows):
        out["HallucinationRate"] = 100.0 * sum(bool(r["hallucinated"]) for r in rows) / len(rows)
    cited = [r for r in rows if r.get("citation_correct") is not None]
    if cited:
        out["CitationAccuracy"] = 100.0 * sum(bool(r["citation_correct"]) for r in cited) / len(cited)
    if all("abstained" in r for r in rows):
        out.update({f"Abstention_{k}": v for k, v in abstention_metrics(rows).items()})
    if all("regenerated" in r for r in rows):
        out["RegenerationRate"] = 100.0 * sum(bool(r["regenerated"]) for r in rows) / len(rows)
    if all("latency_ms" in r for r in rows):
        out["LatencyMeanMs"] = sum(r["latency_ms"] for r in rows) / len(rows)
    return out


def summarise(per_seed):
    keys = sorted(set().union(*[s.keys() for s in per_seed.values()]))
    summ = {}
    for k in keys:
        vals = [s[k] for s in per_seed.values() if k in s and isinstance(s[k], (int, float))]
        if vals:
            summ[k] = {"mean": st.mean(vals), "std": st.stdev(vals) if len(vals) > 1 else 0.0, "values": vals}
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--raw_dir", default="results/raw")
    ap.add_argument("--out_dir", default="results/tables")
    ap.add_argument("--official", action="store_true", help="also run data/raw/squad2/evaluate-v2.0.py")
    ap.add_argument("--pred_name", default="predictions.jsonl", help="e.g. predictions_roberta.jsonl")
    ap.add_argument("--tag", default="", help="suffix of the output table name")
    a = ap.parse_args()
    per_seed = {}
    for d in sorted(Path(a.raw_dir, a.run).glob("seed*")):
        rows = read_jsonl(str(d / a.pred_name))
        per_seed[d.name] = score_rows(rows)
        if a.official and Path("data/raw/squad2/evaluate-v2.0.py").exists():
            preds = {r["id"]: r.get("answer") or "" for r in rows}
            (d / "official_preds.json").write_text(json.dumps(preds))
            res = subprocess.run([sys.executable, "data/raw/squad2/evaluate-v2.0.py", "data/raw/squad2/dev-v2.0.json",
                                  str(d / "official_preds.json")], capture_output=True, text=True, check=True)
            per_seed[d.name]["official"] = json.loads(res.stdout)
    if not per_seed:
        raise SystemExit(f"No predictions under {a.raw_dir}/{a.run}")
    out = {"run": a.run, "per_seed": per_seed, "summary": summarise(per_seed)}
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    Path(a.out_dir, f"{a.run}{a.tag}.json").write_text(json.dumps(out, indent=2))
    for k, v in out["summary"].items():
        print(f"{k:32s} {v['mean']:.4f} ± {v['std']:.4f}")


if __name__ == "__main__":
    main()
