"""Builds a QASPER retrieval corpus on the official v0.3 split (future-work evaluation).

The manuscript (V9) does not report quantitative QASPER results. This script defines the
protocol recommended for that evaluation so that it can be run and reported later:
  * evaluation questions: all questions of the 281 official development papers (1,005 questions)
  * index: full text (title, abstract, paragraphs) of the 888 training + 281 development papers
  * only paper text is indexed; questions, answers and evidence annotations never enter the index
Every system evaluated on QASPER must use this same index.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edugpt_rag.chunking import chunk_corpus  # noqa: E402


def paper_text(p: dict) -> str:
    parts = [p.get("title", ""), p.get("abstract", "")]
    for sec in p.get("full_text", []):
        parts.extend(x for x in sec.get("paragraphs", []) if x)
    return "\n".join(x for x in parts if x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/qasper")
    ap.add_argument("--out", default="data/processed/qasper")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128)
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train = json.load(open(raw / "qasper-train-v0.3.json", encoding="utf-8"))
    dev = json.load(open(raw / "qasper-dev-v0.3.json", encoding="utf-8"))
    docs, questions = [], []
    for split, papers in [("train", train), ("dev", dev)]:
        for pid, p in papers.items():
            docs.append({"doc_id": pid, "split": split, "text": paper_text(p)})
            if split == "dev":
                for qa in p["qas"]:
                    questions.append({"id": qa["question_id"], "paper_id": pid, "question": qa["question"],
                                      "answers": [x["answer"] for x in qa["answers"]]})
    chunks = chunk_corpus(docs, a.window, a.overlap)
    stats = {"papers": {"train": len(train), "dev": len(dev)}, "dev_questions": len(questions),
             "chunks": len(chunks), "window_w": a.window, "overlap_s": a.overlap}
    for name, rows in [("docs", docs), ("chunks", chunks), ("dev_questions", questions)]:
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "corpus_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
