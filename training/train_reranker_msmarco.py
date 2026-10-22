"""Fine-tunes the cross-encoder re-ranker on MS MARCO (Section 4.1, Implementation Details).

Base checkpoint : cross-encoder/ms-marco-MiniLM-L-12-v2
Data            : official triples.train.small.tsv (query, positive passage, BM25 hard negative)
Objective       : binary cross-entropy, Eq. 12 (one positive and one negative pair per triple)
Hyperparameters : 3 epochs, batch size 32, learning rate 2e-5, AdamW, linear warm-up over 10% of steps
Seeds           : 42, 123, 456 (one checkpoint per seed)

--max_triples limits the number of triples read; the value used is written to
training_args.json next to the checkpoint so that it can be reported exactly.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def read_triples(path: str, limit: int | None):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            q, pos, neg = line.rstrip("\n").split("\t")
            yield q, pos, neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", default="data/raw/msmarco/triples.train.small.tsv")
    ap.add_argument("--base", default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    ap.add_argument("--out", default="checkpoints/reranker_seed{seed}")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.10)
    ap.add_argument("--max_triples", type=int, default=None)
    ap.add_argument("--max_length", type=int, default=512)
    a = ap.parse_args()

    import torch
    from sentence_transformers import InputExample
    from sentence_transformers.cross_encoder import CrossEncoder
    from torch.utils.data import DataLoader

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    examples = []
    for q, pos, neg in read_triples(a.triples, a.max_triples):
        examples.append(InputExample(texts=[q, pos], label=1.0))
        examples.append(InputExample(texts=[q, neg], label=0.0))
    loader = DataLoader(examples, shuffle=True, batch_size=a.batch_size)
    steps = math.ceil(len(loader) * a.epochs)
    model = CrossEncoder(a.base, num_labels=1, max_length=a.max_length)
    out = a.out.format(seed=a.seed)
    model.fit(train_dataloader=loader, epochs=a.epochs, warmup_steps=int(a.warmup_ratio * steps),
              optimizer_params={"lr": a.lr}, output_path=out, show_progress_bar=True)
    model.save(out)
    Path(out, "training_args.json").write_text(json.dumps({**vars(a), "n_pairs": len(examples),
                                                            "total_steps": steps}, indent=2))


if __name__ == "__main__":
    main()
