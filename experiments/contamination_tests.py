"""Benchmark-contamination probes (Section 4.1, "Benchmark Contamination Analysis").

Three perturbed question sets are built from the SQuAD 2.0 development split with a fixed
random seed, then answered by the full system with run_squad.py-style inference:
  paraphrase  : 500 questions rewritten by an LLM (meaning preserving; prompt below)
  entity      : 300 questions whose PERSON / GPE / ORG entities are replaced by Wikidata
                entities of the same type that do not occur in the gold paragraph
  numeric     : 200 questions containing a number, replaced by a nearby wrong value

The manuscript does not state how paraphrases were produced or which Wikidata query was used;
the choices below are this package's and are recorded in each output file.

  python experiments/contamination_tests.py --probe numeric
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edugpt_rag.config import load_config  # noqa: E402
from edugpt_rag.pipeline import read_jsonl, write_jsonl  # noqa: E402

PARAPHRASE_PROMPT = ("Rewrite the question so that it asks for exactly the same information using different "
                     "wording. Keep all names, numbers and dates. Return only the rewritten question.\n\nQuestion: {q}")
NUM = re.compile(r"\b\d+(?:[.,]\d+)?\b")
WIKIDATA_TYPES = {"PERSON": "Q5", "GPE": "Q6256", "ORG": "Q43229"}


def perturb_numeric(q, rnd):
    m = list(NUM.finditer(q))
    if not m:
        return None
    x = rnd.choice(m)
    v = float(x.group().replace(",", ""))
    new = v + rnd.choice([-1, 1]) * max(1, round(abs(v) * rnd.uniform(0.1, 0.3)))
    s = str(int(new)) if v.is_integer() else f"{new:.1f}"
    return q[: x.start()] + s + q[x.end():]


def wikidata_labels(qtype: str, n: int = 200):
    import requests
    query = f"""SELECT ?l WHERE {{ ?e wdt:P31 wd:{qtype}; rdfs:label ?l . FILTER(lang(?l)='en') }} LIMIT {n}"""
    r = requests.get("https://query.wikidata.org/sparql", params={"query": query, "format": "json"},
                     headers={"User-Agent": "EduGPT-RAG-repro/1.0"}, timeout=60)
    return [b["l"]["value"] for b in r.json()["results"]["bindings"]]


def perturb_entity(q, context, nlp, pools, rnd):
    ents = [e for e in nlp(q).ents if e.label_ in pools]
    if not ents:
        return None
    e = rnd.choice(ents)
    cands = [c for c in pools[e.label_] if c.lower() not in context.lower() and c != e.text]
    return q[: e.start_char] + rnd.choice(cands) + q[e.end_char:] if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/squad2.yaml")
    ap.add_argument("--probe", choices=["paraphrase", "entity", "numeric"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cfg = load_config(a.config)
    rnd = random.Random(a.seed)
    qs = read_jsonl(str(Path(cfg.corpus_dir) / "dev_questions.jsonl"))
    docs = {d["doc_id"]: d["text"] for d in read_jsonl(str(Path(cfg.corpus_dir) / "docs.jsonl"))}
    n = {"paraphrase": 500, "entity": 300, "numeric": 200}[a.probe]
    pool = [q for q in qs if not q["unanswerable"]]
    rnd.shuffle(pool)
    out = []
    if a.probe == "paraphrase":
        from edugpt_rag.generator import OpenAIGenerator
        gen = OpenAIGenerator(cfg.generator.model, 0.0, 128)
        for q in pool[:n]:
            out.append({**q, "original_question": q["question"],
                        "question": gen._complete(PARAPHRASE_PROMPT.format(q=q["question"]), a.seed).strip()})
    elif a.probe == "numeric":
        for q in pool:
            p = perturb_numeric(q["question"], rnd)
            if p:
                out.append({**q, "original_question": q["question"], "question": p})
            if len(out) == n:
                break
    else:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        pools = {k: wikidata_labels(v) for k, v in WIKIDATA_TYPES.items()}
        for q in pool:
            p = perturb_entity(q["question"], docs[q["gold_doc_id"]], nlp, pools, rnd)
            if p:
                out.append({**q, "original_question": q["question"], "question": p})
            if len(out) == n:
                break
    dst = Path(cfg.corpus_dir) / f"contamination_{a.probe}.jsonl"
    write_jsonl(out, str(dst))
    print(f"wrote {len(out)} perturbed questions to {dst}; answer them with:\n"
          f"  python experiments/run_squad.py --run_name contam_{a.probe} --questions {dst}\n"
          f"  python evaluation/squad_eval.py --run contam_{a.probe}")


if __name__ == "__main__":
    main()
