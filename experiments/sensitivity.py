"""Sensitivity studies: chunking (Table 6), embedding model (Table 7), threshold tau (Tables 9, 10).

Each setting is a full run of experiments/run_squad.py with a different configuration, scored by
evaluation/squad_eval.py. Chunking settings rebuild the corpus into a separate directory.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

CHUNKS = [(256, 64), (256, 128), (512, 64), (512, 128), (1024, 256), (1024, 512)]
EMBEDDERS = {"minilm_l6": ("sentence-transformers/all-MiniLM-L6-v2", 384),
             "mpnet_base": ("microsoft/mpnet-base", 768),
             "all_mpnet_base_v2": ("sentence-transformers/all-mpnet-base-v2", 768),
             "e5_large_v2": ("intfloat/e5-large-v2", 1024),
             "bge_large_en_v15": ("BAAI/bge-large-en-v1.5", 1024)}
TAUS = [0.50, 0.65, 0.75, 0.85, 0.95]


def run(name, sets, cfg):
    subprocess.run([sys.executable, "experiments/run_squad.py", "--config", cfg, "--run_name", name, "--set", *sets],
                   check=True)
    subprocess.run([sys.executable, "evaluation/squad_eval.py", "--run", name], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/squad2.yaml")
    ap.add_argument("--study", choices=["chunk", "embedding", "tau"], required=True)
    a = ap.parse_args()
    if a.study == "chunk":
        for w, s in CHUNKS:
            out = f"data/processed/squad2_w{w}_s{s}"
            subprocess.run([sys.executable, "data/build_corpus_squad.py", "--out", out, "--window", str(w),
                            "--overlap", str(s)], check=True)
            run(f"chunk_w{w}_s{s}", [f"corpus_dir={out}", f"index_dir=indexes/squad2_w{w}_s{s}",
                                     f"chunking.window={w}", f"chunking.overlap={s}"], a.config)
    elif a.study == "embedding":
        # note: "microsoft/mpnet-base" is the un-tuned MPNet encoder (row "MPNet-base-v2" in Table 7 is ambiguous)
        for name, (model, dim) in EMBEDDERS.items():
            run(f"emb_{name}", [f"retrieval.sbert_model={model}", f"retrieval.embedding_dim={dim}",
                                f"index_dir=indexes/squad2_{name}"], a.config)
    else:
        for t in TAUS:
            run(f"tau_{t:.2f}", [f"verifier.tau={t}"], a.config)


if __name__ == "__main__":
    main()
