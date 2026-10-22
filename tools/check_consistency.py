"""Checks every numeric claim in the manuscript text against the manuscript tables.

Inputs : results/reported/*.csv (values transcribed from manuscript V9) and
         data/processed/squad2/corpus_stats.json (built by data/build_corpus_squad.py).
Output : docs/CONSISTENCY_REPORT.md and a non-zero exit code if a check fails.

The checks only compare the manuscript with itself and with the dataset; they do not
validate the experimental values, which require re-running the experiments.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "results" / "reported"


def load(name, key):
    with open(REP / name, encoding="utf-8") as f:
        return {r[key]: r for r in csv.DictReader(f)}


def num(x):
    return float(x) if x not in ("", None) else None


checks = []


def check(cid, where, claim, computed, stated, tol=0.006, note="", kind="text"):
    """kind='text': a sentence of the manuscript restates a table value (mismatch = FAIL).
    kind='results': two reported results that should agree (mismatch = CONFLICT, needs a re-run)."""
    ok = computed is not None and stated is not None and abs(computed - stated) <= tol
    bad = "FAIL" if kind == "text" else "CONFLICT"
    checks.append({"id": cid, "where": where, "claim": claim, "computed": computed, "stated": stated,
                   "status": "PASS" if ok else bad, "note": note})


def flag(cid, where, claim, note):
    checks.append({"id": cid, "where": where, "claim": claim, "computed": None, "stated": None,
                   "status": "OPEN", "note": note})


def main():
    t1 = load("table1_overall.csv", "method")
    t2 = load("table2_same_backbone.csv", "configuration")
    t3 = load("table3_retrieval.csv", "retrieval")
    t4 = load("table4_ablation.csv", "configuration")
    t5 = load("table5_hallucination.csv", "method")
    t6 = list(csv.DictReader(open(REP / "table6_chunking.csv")))
    t8 = load("table8_latency.csv", "component")
    t9 = load("table9_tau.csv", "tau")
    t10 = load("table10_abstention.csv", "tau")
    g = lambda t, r, c: num(t[r][c])  # noqa: E731

    full2 = "EduGPT-RAG (Full)"
    hyb, hr = "GPT-3.5 + Hybrid, no re-rank", "GPT-3.5 + Hybrid + Re-rank, no verify"
    check("C1", "Contribution 2; Sec. 4.2; Sec. 4.4; Discussion", "Hybrid vs BM25-only: +4.00 F1",
          g(t2, hyb, "F1") - g(t2, "GPT-3.5 + BM25 only", "F1"), 4.00)
    check("C2", "Contribution 2; Sec. 4.4; Discussion; Conclusion", "Hybrid vs dense-only: +3.52 F1",
          g(t2, hyb, "F1") - g(t2, "GPT-3.5 + Dense (SBERT) only", "F1"), 3.52)
    check("C3", "Sec. 4.2; Sec. 4.4", "Re-ranking: +2.50 F1", g(t2, hr, "F1") - g(t2, hyb, "F1"), 2.50)
    check("C4", "Sec. 4.2; Sec. 4.4; Discussion", "Verification: +1.28 F1", g(t2, full2, "F1") - g(t2, hr, "F1"), 1.28)
    check("C5", "Contribution 3; Sec. 4.4; Discussion; Conclusion", "Verification: +0.12 Faithfulness",
          g(t2, full2, "Faithfulness") - g(t2, hr, "Faithfulness"), 0.12)
    check("C6", "Sec. 4.4 (component order, item 3)", "CoT prompting: +2.59 F1 (Table 4)",
          g(t4, "Full EduGPT-RAG", "F1") - g(t4, "- CoT prompting", "F1"), 2.59,
          note="V8 stated +1.86; corrected in V9 to the value implied by Table 4")
    check("C7", "Sec. 4.3", "Re-ranking: +5.36 Recall@5",
          g(t3, "Hybrid + Re-rank", "Recall@5") - g(t3, "Hybrid (no re-rank)", "Recall@5"), 5.36)
    for col, st in [("EM", 2.56), ("F1", 1.69), ("Faithfulness", 0.06)]:
        check(f"C8-{col}", "Sec. 4.2", f"vs Atlas: +{st} {col}", g(t1, "EduGPT-RAG", col) - g(t1, "Atlas", col), st)
    for col, st in [("EM", 4.13), ("F1", 2.45)]:
        check(f"C9-{col}", "Sec. 4.2", f"vs Self-RAG: +{st} {col}", g(t1, "EduGPT-RAG", col) - g(t1, "Self-RAG", col), st)
    check("C10", "Sec. 4.2", "BM25+GPT-3.5 vs Vanilla GPT-4: +4.68 EM",
          g(t1, "BM25+GPT-3.5", "EM") - g(t1, "Vanilla GPT-4", "EM"), 4.68)
    h_e, h_s = g(t5, "EduGPT-RAG", "HallucinationRate"), g(t5, "Self-RAG", "HallucinationRate")
    check("C11", "Contribution 3; Sec. 4.5; Conclusion; Limitation 9", "57.2% relative hallucination reduction",
          100 * (h_s - h_e) / h_s, 57.2, tol=0.06)
    r = {(x["w"], x["s"]): x for x in t6}
    check("C12", "Sec. 4.6", "s=128 vs s=64 at w=512: +0.91 F1",
          num(r[("512", "128")]["F1"]) - num(r[("512", "64")]["F1"]), 0.91)
    check("C13", "Sec. 4.8", "Retrieval (FAISS+BM25) 7 ms at 10K", g(t8, "FAISS Search", "10K") + g(t8, "BM25 Search", "10K"), 7)
    check("C14", "Sec. 4.8", "Retrieval (FAISS+BM25) 90 ms at 1M", g(t8, "FAISS Search", "1M") + g(t8, "BM25 Search", "1M"), 90)
    check("C15", "Sec. 4.10", "Edge stack 68 ms at 100K (8+6+18+1+35)",
          sum(g(t8, c, "100K") for c in ["SBERT Encoding", "FAISS Search", "BM25 Search", "RRF Fusion", "Cross-Encoder"]), 68)
    check("C16", "Discussion", "Retrieval stack 58 ms at 500K (FAISS + BM25 + RRF)",
          g(t8, "FAISS Search", "500K") + g(t8, "BM25 Search", "500K") + g(t8, "RRF Fusion", "500K"), 58)
    for col in ["10K", "50K", "100K", "500K", "1M"]:
        comp = sum(g(t8, c, col) for c in t8 if c != "Total")
        check(f"C17-{col}", "Table 8", f"Total row equals sum of components ({col})", comp, g(t8, "Total", col))
    check("C18", "Sec. 4.9", "Regeneration overhead 38 ms (506 - 468)", g(t9, "0.75", "LatencyMs") - g(t9, "0.50", "LatencyMs"), 38)
    for tau, row in t10.items():
        p, rc = num(row["Precision"]), num(row["Recall"])
        check(f"C19-{tau}", "Table 10", f"F1-Abstention = harmonic mean of P and R (tau={tau})",
              2 * p * rc / (p + rc), num(row["F1"]), tol=0.011)
        check(f"C20-{tau}", "Table 10", f"Abstention rate on unanswerable = 100 x Recall (tau={tau})",
              100 * rc, num(row["AbstentionRate"]), tol=1.0,
              note="Both columns are defined on the 5,945 unanswerable questions, so they should coincide",
              kind="results")
    check("C21", "Tables 5 and 10", "Abstention (Table 5) equals Table 10 at tau=0.75",
          g(t5, "EduGPT-RAG", "Abstention"), num(t10["0.75"]["AbstentionRate"]), kind="results")
    check("C22", "Tables 2 and 4", "'- Cross-encoder re-ranking' (Table 4) vs 'Hybrid, no re-rank' (Table 2) EM",
          g(t4, "- Cross-encoder re-ranking", "EM"), g(t2, hyb, "EM"),
          note="Table 4 keeps CoT and verification, so it should not be lower than Table 2", kind="results")
    check("C23", "Tables 2 and 4", "Table 4 '- Hybrid -> Dense only' differs from Table 2 dense-only (verification on vs off)",
          0.0 if g(t4, "- Hybrid -> Dense only", "EM") != g(t2, "GPT-3.5 + Dense (SBERT) only", "EM") else 1.0, 0.0,
          note="Identical values although Table 4 keeps re-ranking, CoT and verification", kind="results")
    for col in ["EM", "F1", "Recall@5", "Faithfulness"]:
        check(f"C24-{col}", "Abstract; Conclusion", f"Headline {col} equals Table 1",
              g(t1, "EduGPT-RAG", col), {"EM": 89.47, "F1": 94.12, "Recall@5": 96.83, "Faithfulness": 0.91}[col])
    check("C25", "Sec. 4.8", "Cost: 512 input + 256 output tokens at $0.002 per 1K = $0.0015", (512 + 256) * 0.002 / 1000, 0.0015,
          tol=5e-5)
    check("C26", "Sec. 4.10", "Cost: '512 tokens ... about $0.001' at $0.002 per 1K", 512 * 0.002 / 1000, 0.001, tol=5e-5,
          note="Second cost statement uses 512 tokens; consistent only to one significant figure")
    import statistics
    fx = [num(v["AvgFaithfulness"]) for v in t5.values()]
    hy = [num(v["HallucinationRate"]) for v in t5.values()]
    check("C36", "Sec. 4.5; Figure 6 caption", "Correlation between hallucination rate and faithfulness r = -0.998",
          statistics.correlation(fx, hy), -0.998, tol=0.0006, note="V8 stated r = -0.97; Table 5 gives r = -0.998")
    flag("C27", "Sec. 4.2, Table 1", "Recall@5 t-test vs Self-RAG",
         "Self-RAG Recall@5 has no per-seed standard deviation in Table 1; the per-seed values behind t(2)=11.54 are not archived")
    cs = ROOT / "data" / "processed" / "squad2" / "corpus_stats.json"
    if cs.exists():
        s = json.loads(cs.read_text())
        check("C28", "Sec. 4.1", "Indexed paragraphs 20,239 (19,035 + 1,204)",
              s["paragraphs"]["train"] + s["paragraphs"]["dev"], 20239)
        check("C29", "Sec. 4.1", "Unique paragraphs after de-duplication 20,233", s["unique_paragraphs_indexed"], 20233)
        check("C30", "Sec. 4.1", "Chunks 20,238", s["chunks"], 20238)
        check("C31", "Sec. 4.1", "Training split: 86,821 answerable / 43,498 unanswerable",
              s["answerable"]["train"] * 1e5 + s["unanswerable"]["train"], 86821 * 1e5 + 43498)
        check("C32", "Sec. 4.1", "Development split: 5,928 answerable / 5,945 unanswerable",
              s["answerable"]["dev"] * 1e5 + s["unanswerable"]["dev"], 5928 * 1e5 + 5945)
        n = s["chunks"]
        ivfpq_mb = (n * (64 + 8) + 4096 * 768 * 4 + 64 * 256 * 12 * 4) / 1e6
        flag("C33", "Table 7; Secs. 4.8 and 4.10", "Index size 2.4 GB",
             f"An IVF4096,PQ64 index over {n} 768-d vectors is about {ivfpq_mb:.0f} MB "
             "(codes + ids + coarse centroids + PQ codebooks); the 2.4 GB figure must be re-measured "
             "(experiments/latency_benchmark.py reports index size)")
        flag("C34", "Sec. 4.1 Implementation Details", "FAISS trained on 100K sampled embeddings",
             f"The SQuAD 2.0 corpus has {n} chunks, so at most {n} training vectors exist; V9 states 'up to 100K'")
    flag("C35", "Sec. 4.5", "Human evaluation (200 items, Fleiss kappa 0.74/0.71, r = 0.87)",
         "Annotation data not archived; protocol and analysis script in evaluation/human_eval/")

    lines = ["# Manuscript consistency report", "",
             "Generated by `tools/check_consistency.py`. PASS = text claim matches the manuscript tables or the "
             "dataset; FAIL = a sentence contradicts a table; CONFLICT = two reported results contradict each other and can only be "
             "resolved by re-running the experiments; OPEN = cannot be checked without the original experiment outputs.", "",
             "| ID | Location | Claim | Computed | Stated | Status | Note |", "|---|---|---|---|---|---|---|"]
    fmt = lambda v: "" if v is None else (f"{v:.4f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v))  # noqa
    for c in checks:
        lines.append(f"| {c['id']} | {c['where']} | {c['claim']} | {fmt(c['computed'])} | {fmt(c['stated'])} | "
                     f"{c['status']} | {c['note']} |")
    summary = {s: sum(1 for c in checks if c["status"] == s) for s in ("PASS", "FAIL", "CONFLICT", "OPEN")}
    lines += ["", f"Summary: {summary['PASS']} PASS, {summary['FAIL']} FAIL, {summary['CONFLICT']} CONFLICT, "
              f"{summary['OPEN']} OPEN."]
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "CONSISTENCY_REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-1:]))
    for c in checks:
        if c["status"] != "PASS":
            print(c["status"], c["id"], c["claim"], c["computed"], c["stated"])
    return 1 if summary["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
