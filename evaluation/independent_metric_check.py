"""Independent validation of the faithfulness / hallucination / citation metrics (Reviewer 1, Comment 2).

The verification layer and the reported metrics share one NLI model. This script re-estimates
the metrics with scorers that do not share that model:
  (a) a second NLI model from a different family (default: FacebookAI/roberta-large-mnli);
  (b) an LLM judge (default: gpt-4o) that labels each answer sentence as SUPPORTED / NOT_SUPPORTED
      by the passages and checks the cited passage, on a random sample;
and reports agreement with the primary NLI scorer (Pearson and Spearman r with 95% bootstrap CIs,
Cohen's kappa on the binary hallucination label).

Outputs: results/tables/<run>_independent.json and the per-item judgements in results/raw.
No result of this check is reported in manuscript V9; it is provided so that it can be run.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edugpt_rag.build import load_chunks  # noqa: E402
from edugpt_rag.chunking import split_sentences  # noqa: E402
from edugpt_rag.config import load_config  # noqa: E402
from edugpt_rag.pipeline import read_jsonl, write_jsonl  # noqa: E402

JUDGE_PROMPT = """You are checking whether an answer is supported by source passages.
Passages:
{passages}

Answer sentences:
{sentences}

For each answer sentence, reply SUPPORTED if it is fully entailed by at least one passage, otherwise
NOT_SUPPORTED. Then state whether passage [{cited}] alone supports every sentence (CITATION_OK or
CITATION_WRONG). Reply as JSON: {{"sentences": ["SUPPORTED", ...], "citation": "CITATION_OK"}}"""


def bootstrap_ci(x, y, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    vals = [fn(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def cohen_kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def main():
    from scipy.stats import pearsonr, spearmanr
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/squad2.yaml")
    ap.add_argument("--run", default="edugpt_full")
    ap.add_argument("--seed_dir", default="seed42")
    ap.add_argument("--second_nli", default="FacebookAI/roberta-large-mnli")
    ap.add_argument("--judge", default="gpt-4o")
    ap.add_argument("--n_judge", type=int, default=500)
    a = ap.parse_args()
    cfg = load_config(a.config)
    # (a) second NLI model
    subprocess.run([sys.executable, "evaluation/run_hallucination.py", "--config", a.config, "--run", a.run,
                    "--nli_model", a.second_nli, "--suffix", "_second_nli"], check=True)
    d = Path(cfg.output_dir, a.run, a.seed_dir)
    prim = {r["id"]: r for r in read_jsonl(str(d / "predictions.jsonl"))}
    sec = {r["id"]: r for r in read_jsonl(str(d / "predictions_second_nli.jsonl"))}
    ids = [i for i in prim if prim[i].get("faith") is not None and sec[i].get("faith") is not None]
    f1 = [prim[i]["faith"] for i in ids]
    f2 = [sec[i]["faith"] for i in ids]
    h1 = [prim[i]["hallucinated"] for i in ids]
    h2 = [sec[i]["hallucinated"] for i in ids]
    report = {"n": len(ids), "second_nli": a.second_nli,
              "faith_primary_mean": float(np.mean(f1)), "faith_second_mean": float(np.mean(f2)),
              "halluc_primary": 100 * float(np.mean(h1)), "halluc_second": 100 * float(np.mean(h2)),
              "pearson": float(pearsonr(f1, f2)[0]), "pearson_ci95": bootstrap_ci(f1, f2, lambda x, y: pearsonr(x, y)[0]),
              "spearman": float(spearmanr(f1, f2)[0]),
              "spearman_ci95": bootstrap_ci(f1, f2, lambda x, y: spearmanr(x, y)[0]),
              "kappa_hallucination": cohen_kappa(h1, h2)}
    # (b) LLM judge on a random sample
    from edugpt_rag.generator import OpenAIGenerator
    judge = OpenAIGenerator(a.judge, temperature=0.0, max_tokens=512)
    chunks = load_chunks(cfg.corpus_dir)
    rnd = random.Random(0)
    sample = rnd.sample(ids, min(a.n_judge, len(ids)))
    judged = []
    for i in sample:
        r = prim[i]
        ctx = [chunks[c]["text"] for c in r["context_ids"]]
        sents = split_sentences(r["answer"])
        prompt = JUDGE_PROMPT.format(passages="\n".join(f"[{k}] {p}" for k, p in enumerate(ctx, 1)),
                                     sentences="\n".join(f"- {s}" for s in sents), cited=r.get("cited_passage") or 1)
        txt = judge._complete(prompt, seed=0)
        try:
            j = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
        except json.JSONDecodeError:
            j = {"sentences": [], "citation": None, "unparsed": txt}
        halluc = any(x != "SUPPORTED" for x in j.get("sentences", [])) if j.get("sentences") else None
        judged.append({"id": i, "judge": j, "judge_hallucinated": halluc, "nli_hallucinated": r["hallucinated"]})
    ok = [x for x in judged if x["judge_hallucinated"] is not None]
    report.update({"judge": a.judge, "n_judged": len(ok),
                   "halluc_judge": 100 * float(np.mean([x["judge_hallucinated"] for x in ok])) if ok else None,
                   "kappa_judge_vs_nli": cohen_kappa([x["judge_hallucinated"] for x in ok],
                                                     [x["nli_hallucinated"] for x in ok]) if ok else None})
    write_jsonl(judged, str(d / "llm_judge.jsonl"))
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path(f"results/tables/{a.run}_independent.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
