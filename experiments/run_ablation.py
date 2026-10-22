"""Same-backbone ablations (Table 2) and leave-one-out ablations (Table 4).

All variants share the frozen GPT-3.5-turbo generator. Each variant is a set of pipeline
switches; predictions are written by run_squad.py and scored by evaluation/squad_eval.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

TABLE2 = {  # cumulative, all with GPT-3.5-turbo
    "t2_bm25_only":            ["use_dense=false", "use_rerank=false", "use_verification=false"],
    "t2_dense_only":           ["use_sparse=false", "use_rerank=false", "use_verification=false"],
    "t2_hybrid_no_rerank":     ["use_rerank=false", "use_verification=false"],
    "t2_hybrid_rerank_no_ver": ["use_verification=false"],
    "edugpt_full":             [],
}
TABLE4 = {  # leave one component out of the full system
    "t4_minus_verification": ["use_verification=false"],
    "t4_minus_cot":          ["use_cot=false"],
    "t4_minus_rerank":       ["use_rerank=false"],
    "t4_dense_only":         ["use_sparse=false"],
    "t4_bm25_only":          ["use_dense=false"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/squad2.yaml")
    ap.add_argument("--table", choices=["2", "4", "all"], default="all")
    a = ap.parse_args()
    runs = {}
    if a.table in ("2", "all"):
        runs.update(TABLE2)
    if a.table in ("4", "all"):
        runs.update(TABLE4)
    for name, sets in runs.items():
        cmd = [sys.executable, "experiments/run_squad.py", "--config", a.config, "--run_name", name]
        if sets:
            cmd += ["--set", *sets]
        subprocess.run(cmd, check=True)
        subprocess.run([sys.executable, "evaluation/squad_eval.py", "--run", name], check=True)


if __name__ == "__main__":
    main()
