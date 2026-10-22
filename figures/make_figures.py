"""Draws Figures 3 to 9 from table CSVs.

--source reported : values transcribed from manuscript V9 (results/reported) - lets reviewers
                    confirm that every figure matches its table.
--source rerun    : values produced by this package (results/tables/*.json, via tools/export_tables.py)
Figures are written to figures/out/<source>/fig<N>_*.png (600 dpi, Times New Roman, IEEE column width).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "savefig.dpi": 600, "savefig.bbox": "tight"})
W = 3.5  # inches, IEEE single column


def rows(d, name):
    return list(csv.DictReader(open(Path(d) / name, encoding="utf-8")))


def legend_below(ax, ncol):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=ncol, frameon=False)


def fig3(d, out):
    r = rows(d, "table1_overall.csv")
    names = [x["method"] for x in r]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(W, 2.4))
    for k, (col, sd) in enumerate([("EM", "EM_std"), ("F1", "F1_std")]):
        v = [float(i[col]) for i in r]
        e = [float(i[sd]) if i[sd] else 0 for i in r]
        ax.bar(x + (k - 0.5) * 0.38, v, 0.38, yerr=e, capsize=1.5, label=col)
    ax.set_xticks(x, names, rotation=35, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(70, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=2, frameon=False)
    fig.savefig(out / "fig3_em_f1.png")


def fig4(d, out):
    r = rows(d, "table3_retrieval.csv")
    ks = [1, 3, 5, 10]
    fig, ax = plt.subplots(figsize=(W, 2.2))
    for i in r:
        ax.plot(ks, [float(i[f"Recall@{k}"]) for k in ks], marker="o", ms=3, label=i["retrieval"])
    ax.set_xticks(ks)
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K (%)")
    legend_below(ax, 2)
    fig.savefig(out / "fig4_recall_at_k.png")


def fig5(d, out):
    t2 = {x["configuration"]: float(x["EM"]) for x in rows(d, "table2_same_backbone.csv")}
    steps = [("BM25 only", t2["GPT-3.5 + BM25 only"]), ("+ Hybrid", t2["GPT-3.5 + Hybrid, no re-rank"]),
             ("+ Re-rank", t2["GPT-3.5 + Hybrid + Re-rank, no verify"]), ("+ Verification", t2["EduGPT-RAG (Full)"])]
    fig, ax = plt.subplots(figsize=(W, 2.2))
    prev = steps[0][1]
    ax.bar(0, prev, color="0.6")
    ax.text(0, prev + 0.2, f"{prev:.2f}", ha="center", fontsize=6)
    for i, (n, v) in enumerate(steps[1:], start=1):
        ax.bar(i, v - prev, bottom=prev)
        ax.text(i, v + 0.2, f"+{v - prev:.2f}", ha="center", fontsize=6)
        prev = v
    ax.set_xticks(range(len(steps)), [s for s, _ in steps])
    ax.set_ylim(75, 92)
    ax.set_ylabel("EM (%)")
    fig.savefig(out / "fig5_waterfall_em.png")


def fig6(d, out):
    r = rows(d, "table5_hallucination.csv")
    fig, ax = plt.subplots(figsize=(W, 2.4))
    x = [float(i["AvgFaithfulness"]) for i in r]
    y = [float(i["HallucinationRate"]) for i in r]
    ax.scatter(x, y, s=12)
    for i, xi, yi in zip(r, x, y):
        ax.annotate(i["method"], (xi, yi), xytext=(3, 3), textcoords="offset points", fontsize=6)
    rr = np.corrcoef(x, y)[0, 1]
    ax.set_xlabel("Average faithfulness")
    ax.set_ylabel("Hallucination rate (%)")
    ax.set_title(f"Pearson r = {rr:.2f}", fontsize=7)
    fig.savefig(out / "fig6_halluc_vs_faith.png")
    return rr


def fig7(d, out):
    r = rows(d, "table6_chunking.csv")
    ws, ss = sorted({int(i["w"]) for i in r}), sorted({int(i["s"]) for i in r})
    M = np.full((len(ws), len(ss)), np.nan)
    for i in r:
        M[ws.index(int(i["w"])), ss.index(int(i["s"]))] = float(i["F1"])
    fig, ax = plt.subplots(figsize=(W, 2.2))
    im = ax.imshow(M, cmap="Blues", aspect="auto")
    for a_ in range(len(ws)):
        for b in range(len(ss)):
            if not np.isnan(M[a_, b]):
                ax.text(b, a_, f"{M[a_, b]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_xticks(range(len(ss)), ss)
    ax.set_yticks(range(len(ws)), ws)
    ax.set_xlabel("Overlap s (tokens)")
    ax.set_ylabel("Window w (tokens)")
    fig.colorbar(im, ax=ax, label="F1 (%)")
    fig.savefig(out / "fig7_chunk_heatmap.png")


def fig8(d, out):
    r = {x["component"]: x for x in rows(d, "table8_latency.csv")}
    cols = ["10K", "50K", "100K", "500K", "1M"]
    fig, ax = plt.subplots(figsize=(W, 2.4))
    bottom = np.zeros(len(cols))
    for c, v in r.items():
        if c == "Total":
            continue
        vals = np.array([float(v[k]) for k in cols])
        ax.bar(cols, vals, bottom=bottom, label=c.replace("Anti-hallucination verification layer", "Verification"))
        bottom += vals
    ax.set_xlabel("Corpus size (passages)")
    ax.set_ylabel("Latency (ms)")
    legend_below(ax, 3)
    fig.savefig(out / "fig8_latency.png")


def fig9(d, out):
    r = rows(d, "table9_tau.csv")
    fig, ax = plt.subplots(figsize=(W, 2.2))
    x = [float(i["LatencyMs"]) for i in r]
    y = [float(i["F1"]) for i in r]
    ax.plot(x, y, marker="o", ms=3)
    for i, xi, yi in zip(r, x, y):
        ax.annotate(f"τ={i['tau']}", (xi, yi), xytext=(3, -8), textcoords="offset points", fontsize=6)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("F1 (%)")
    fig.savefig(out / "fig9_pareto_tau.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["reported", "rerun"], default="reported")
    a = ap.parse_args()
    d = Path("results/reported" if a.source == "reported" else "results/tables/csv")
    out = Path("figures/out") / a.source
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig3, fig4, fig5, fig7, fig8, fig9):
        f(d, out)
    r = fig6(d, out)
    print(f"figures written to {out}; Figure 6 Pearson r = {r:.2f}")


if __name__ == "__main__":
    main()
