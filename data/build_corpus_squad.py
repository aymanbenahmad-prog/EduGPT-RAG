"""Builds the SQuAD 2.0 retrieval corpus exactly as described in Section 4.1.

Indexed text: the unique Wikipedia paragraphs of the official training split (442 articles)
and development split (35 articles). Questions, answers and answerability labels are never
written to the corpus files. Each paragraph is chunked with w = 512, s = 128 (Section 3.2).

Outputs (data/processed/squad2/):
  docs.jsonl           one line per paragraph   {doc_id, split, title, text}
  chunks.jsonl         one line per chunk       {chunk_id, doc_id, text, n_tokens, ...}
  dev_questions.jsonl  development questions    {id, question, answers, unanswerable, gold_doc_id}
  corpus_stats.json    counts used in the manuscript
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edugpt_rag.chunking import chunk_corpus  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/squad2")
    ap.add_argument("--out", default="data/processed/squad2")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128)
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    docs, seen, dev_q = [], {}, []
    stats = {"articles": {}, "paragraphs": {}, "questions": {}, "answerable": {}, "unanswerable": {}}
    for split, fname in [("train", "train-v2.0.json"), ("dev", "dev-v2.0.json")]:
        data = json.load(open(raw / fname, encoding="utf-8"))["data"]
        stats["articles"][split] = len(data)
        n_par = n_q = n_un = 0
        for ai, art in enumerate(data):
            for pi, par in enumerate(art["paragraphs"]):
                n_par += 1
                key = (art["title"], par["context"])
                if key not in seen:
                    doc_id = f"{split}-{ai:03d}-{pi:03d}"
                    seen[key] = doc_id
                    docs.append({"doc_id": doc_id, "split": split, "title": art["title"], "text": par["context"]})
                for qa in par["qas"]:
                    n_q += 1
                    n_un += int(qa["is_impossible"])
                    if split == "dev":
                        dev_q.append({"id": qa["id"], "question": qa["question"],
                                      "answers": [x["text"] for x in qa["answers"]],
                                      "unanswerable": bool(qa["is_impossible"]),
                                      "gold_doc_id": seen[key]})
        stats["paragraphs"][split] = n_par
        stats["questions"][split] = n_q
        stats["unanswerable"][split] = n_un
        stats["answerable"][split] = n_q - n_un

    chunks = chunk_corpus(docs, a.window, a.overlap)
    per_doc = Counter(c["doc_id"] for c in chunks)
    stats.update({
        "unique_paragraphs_indexed": len(docs),
        "chunks": len(chunks),
        "chunks_per_paragraph_mean": round(len(chunks) / len(docs), 4),
        "paragraphs_with_more_than_one_chunk": sum(1 for v in per_doc.values() if v > 1),
        "max_paragraph_tokens": max(len(d["text"].split()) for d in docs),
        "mean_paragraph_tokens": round(sum(len(d["text"].split()) for d in docs) / len(docs), 2),
        "window_w": a.window, "overlap_s": a.overlap, "token_unit": "whitespace-delimited tokens",
        "sha256": {f: sha256(raw / f) for f in ["train-v2.0.json", "dev-v2.0.json"]},
    })
    with open(out / "docs.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(out / "dev_questions.jsonl", "w", encoding="utf-8") as f:
        for q in dev_q:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    (out / "corpus_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
