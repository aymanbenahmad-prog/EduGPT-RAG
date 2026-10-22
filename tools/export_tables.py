"""Turns results/tables/<run>.json into CSVs with the schema of results/reported/*.csv,
then prints the difference to the manuscript values (re-run minus reported).

Only runs that exist are exported; missing cells stay empty. Nothing is copied from the
reported values into the re-run CSVs.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

T = Path("results/tables")
OUT = T / "csv"
REP = Path("results/reported")

TABLE1 = {"Vanilla GPT-4": "vanilla_gpt4", "BM25+GPT-3.5": "t2_bm25_only", "DPR+BART": "dpr_bart",
          "ColBERT-QA": "colbert_qa", "REALM": "realm", "FiD": "fid", "Self-RAG": "self_rag", "Atlas": "atlas",
          "EduGPT-RAG": "edugpt_full"}
TABLE2 = {"GPT-3.5 + BM25 only": "t2_bm25_only", "GPT-3.5 + Dense (SBERT) only": "t2_dense_only",
          "GPT-3.5 + Hybrid, no re-rank": "t2_hybrid_no_rerank",
          "GPT-3.5 + Hybrid + Re-rank, no verify": "t2_hybrid_rerank_no_ver", "EduGPT-RAG (Full)": "edugpt_full"}
TABLE4 = {"Full EduGPT-RAG": "edugpt_full", "- Anti-hallucination verification layer": "t4_minus_verification",
          "- CoT prompting": "t4_minus_cot", "- Cross-encoder re-ranking": "t4_minus_rerank",
          "- Hybrid -> Dense only": "t4_dense_only", "- Hybrid -> BM25 only": "t4_bm25_only"}
TABLE3 = {"BM25 Only": ("t2_bm25_only", "fused"), "SBERT Dense Only": ("t2_dense_only", "fused"),
          "Hybrid (no re-rank)": ("t2_hybrid_no_rerank", "fused"), "Hybrid + Re-rank": ("edugpt_full", "context")}
TABLE5 = {"Vanilla GPT-4": "vanilla_gpt4", "BM25+GPT-3.5": "t2_bm25_only", "DPR+BART": "dpr_bart",
          "ColBERT-QA": "colbert_qa", "FiD": "fid", "Self-RAG": "self_rag", "EduGPT-RAG": "edugpt_full",
          "REALM": "realm", "Atlas": "atlas"}


def summ(run):
    f = T / f"{run}.json"
    return json.loads(f.read_text())["summary"] if f.exists() else None


def m(s, k, scale=1.0, std=False):
    if not s or k not in s:
        return ""
    return round(s[k]["std" if std else "mean"] * scale, 4)


def write(name, header, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    write("table1_overall.csv", ["method", "EM", "EM_std", "F1", "F1_std", "Recall@5", "Recall@5_std", "Faithfulness",
                                 "Faithfulness_std"],
          [[k, m(s, "EM"), m(s, "EM", std=True), m(s, "F1"), m(s, "F1", std=True), m(s, "Recall@5_context"),
            m(s, "Recall@5_context", std=True), m(s, "Faithfulness"), m(s, "Faithfulness", std=True)]
           for k, r in TABLE1.items() for s in [summ(r)]])
    write("table2_same_backbone.csv", ["configuration", "EM", "EM_std", "F1", "F1_std", "Recall@5", "Faithfulness",
                                       "Faithfulness_std"],
          [[k, m(s, "EM"), m(s, "EM", std=True), m(s, "F1"), m(s, "F1", std=True), m(s, "Recall@5_context"),
            m(s, "Faithfulness"), m(s, "Faithfulness", std=True)] for k, r in TABLE2.items() for s in [summ(r)]])
    write("table3_retrieval.csv", ["retrieval", "Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR"],
          [[k] + [m(s, f"Recall@{n}_{w}") for n in (1, 3, 5, 10)] + [m(s, f"MRR_{w}")]
           for k, (r, w) in TABLE3.items() for s in [summ(r)]])
    write("table4_ablation.csv", ["configuration", "EM", "F1", "Recall@5", "Faithfulness"],
          [[k, m(s, "EM"), m(s, "F1"), m(s, "Recall@5_context"), m(s, "Faithfulness")]
           for k, r in TABLE4.items() for s in [summ(r)]])
    write("table5_hallucination.csv", ["method", "HallucinationRate", "AvgFaithfulness", "Abstention", "CitationAccuracy"],
          [[k, m(s, "HallucinationRate"), m(s, "Faithfulness"), m(s, "Abstention_abstention_rate_unanswerable"),
            m(s, "CitationAccuracy")] for k, r in TABLE5.items() for s in [summ(r)]])
    write("table9_tau.csv", ["tau", "EM", "F1", "Faithfulness", "RegenRate", "LatencyMs"],
          [[t, m(s, "EM"), m(s, "F1"), m(s, "Faithfulness"), m(s, "RegenerationRate"), m(s, "LatencyMeanMs")]
           for t in ["0.50", "0.65", "0.75", "0.85", "0.95"] for s in [summ(f"tau_{t}")]])
    write("table10_abstention.csv", ["tau", "AbstentionRate", "Precision", "Recall", "F1"],
          [[t, m(s, "Abstention_abstention_rate_unanswerable"), m(s, "Abstention_precision"), m(s, "Abstention_recall"),
            m(s, "Abstention_f1")] for t in ["0.50", "0.65", "0.75", "0.85", "0.95"] for s in [summ(f"tau_{t}")]])
    # comparison with the manuscript
    for f in sorted(OUT.glob("*.csv")):
        rep = REP / f.name
        if not rep.exists():
            continue
        R = list(csv.DictReader(open(rep)))
        N = list(csv.DictReader(open(f)))
        key = list(R[0].keys())[0]
        print(f"== {f.name}")
        for r, n in zip(R, N):
            diffs = []
            for c in r:
                if c == key or c not in n or not r[c] or not n[c]:
                    continue
                diffs.append(f"{c}: {float(n[c]) - float(r[c]):+.2f}")
            print(f"  {r[key]:40s} " + (", ".join(diffs) if diffs else "(not re-run)"))


if __name__ == "__main__":
    main()
