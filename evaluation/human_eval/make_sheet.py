"""Creates a blinded, shuffled annotation sheet for two systems (see rubric.md)."""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from edugpt_rag.build import load_chunks  # noqa: E402
from edugpt_rag.pipeline import read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs=2, default=["edugpt_full", "self_rag"])
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--seed_dir", default="seed42")
    ap.add_argument("--corpus_dir", default="data/processed/squad2")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="evaluation/human_eval")
    a = ap.parse_args()
    chunks = load_chunks(a.corpus_dir)
    preds = [{r["id"]: r for r in read_jsonl(str(Path(a.raw, run, a.seed_dir, "predictions.jsonl")))} for run in a.runs]
    common = sorted(set(preds[0]) & set(preds[1]))
    rnd = random.Random(a.seed)
    qids = rnd.sample(common, a.n)
    items = []
    for run, p in zip(a.runs, preds):
        for q in qids:
            r = p[q]
            ctx = r.get("contexts") or [chunks[c]["text"] for c in r.get("context_ids", [])]
            items.append({"system": run, "qid": q, "question": r["question"], "answer": r.get("answer") or "[abstained]",
                          "passages": "\n\n".join(f"[{k}] {t}" for k, t in enumerate(ctx, 1))})
    rnd.shuffle(items)
    out = Path(a.out)
    with open(out / "annotation_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "question", "passages", "answer", "factual_accuracy", "source_faithfulness", "annotator"])
        for i, it in enumerate(items):
            w.writerow([i, it["question"], it["passages"], it["answer"], "", "", ""])
    with open(out / "key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "system", "qid"])
        for i, it in enumerate(items):
            w.writerow([i, it["system"], it["qid"]])
    print(f"wrote {len(items)} items")


if __name__ == "__main__":
    main()
