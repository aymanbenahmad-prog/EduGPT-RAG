"""Adds NLI-based faithfulness, hallucination and citation fields to any predictions file (Table 5).

For every non-abstained answer, each answer sentence is scored against each context passage
with the same NLI model used by the verification layer (configs/*.yaml: verifier.nli_model).
Systems without their own contexts (closed-book Vanilla GPT-4, REALM) are scored against the
final contexts of a reference run (default: edugpt_full, same seed); this choice is recorded
in the output so that it can be reported.

Use --nli_model to re-score with an independent NLI model (R1-C2, see
evaluation/independent_metric_check.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edugpt_rag.build import load_chunks  # noqa: E402
from edugpt_rag.config import load_config  # noqa: E402
from edugpt_rag.pipeline import read_jsonl, write_jsonl  # noqa: E402
from edugpt_rag.verifier import (NLIScorer, citation_correct, faithfulness, is_hallucinated,  # noqa: E402
                                 sentence_scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/squad2.yaml")
    ap.add_argument("--run", required=True)
    ap.add_argument("--reference_run", default="edugpt_full")
    ap.add_argument("--nli_model", default=None)
    ap.add_argument("--suffix", default="", help="output file suffix, e.g. _roberta")
    a = ap.parse_args()
    cfg = load_config(a.config)
    model = a.nli_model or cfg.verifier.nli_model
    nli = NLIScorer(model, cfg.verifier.max_length, cfg.device)
    chunks = load_chunks(cfg.corpus_dir)
    for d in sorted(Path(cfg.output_dir, a.run).glob("seed*")):
        rows = read_jsonl(str(d / "predictions.jsonl"))
        ref = {}
        ref_file = Path(cfg.output_dir, a.reference_run, d.name, "predictions.jsonl")
        if ref_file.exists():
            ref = {r["id"]: [chunks[c]["text"] for c in r["context_ids"]] for r in read_jsonl(str(ref_file))}
        for r in rows:
            ctx = r.get("contexts")
            if ctx is None and r.get("context_ids"):
                ctx = [chunks[c]["text"] for c in r["context_ids"]]
            source = "own"
            if not ctx:
                ctx, source = ref.get(r["id"], []), f"reference:{a.reference_run}"
            r["nli_context_source"] = source
            r["nli_model"] = model
            if r.get("abstained") or not r.get("answer") or not ctx:
                r.update({"faith": None, "hallucinated": False, "citation_correct": None, "sentence_scores": []})
                continue
            S = sentence_scores(r["answer"], ctx, nli)
            r.update({"faith": faithfulness(S), "hallucinated": is_hallucinated(S, cfg.verifier.hallucination_threshold),
                      "citation_correct": citation_correct(S, r.get("cited_passage"), cfg.verifier.citation_threshold),
                      "sentence_scores": S.tolist()})
        write_jsonl(rows, str(d / f"predictions{a.suffix}.jsonl"))
        print(f"scored {len(rows)} rows in {d} with {model}")


if __name__ == "__main__":
    main()
