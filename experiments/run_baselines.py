"""Baselines of Table 1, run on the same SQuAD 2.0 development questions and the same corpus.

Every baseline writes results/raw/<name>/seed<k>/predictions.jsonl with the fields used by
evaluation/squad_eval.py and evaluation/run_hallucination.py:
  id, answer, abstained, gold_answers, unanswerable, gold_doc_id, contexts (texts given to the
  generator, empty for closed-book), context_doc_ids (when retrieval is from this corpus).

Status of each adapter (see docs/BASELINES.md for details):
  vanilla_gpt4 .... implemented here (OpenAI API, question only)
  bm25_gpt35 ...... identical to the Table 2 run "t2_bm25_only" (run_ablation.py)
  dpr_bart ........ implemented here (facebook/dpr-*-single-nq-base, facebook/bart-large)
  colbert_qa ...... implemented here (colbert-ai, colbert-ir/colbertv2.0, PLAID) + GPT-3.5 CoT reader
  realm ........... implemented here (google/realm-orqa-nq-openqa, its own retrieval corpus)
  self_rag ........ implemented here (selfrag/selfrag_llama2-7b, short-form inference, threshold 0.5)
  fid, atlas ...... export/import adapters around the official repositories (no HF checkpoint)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edugpt_rag.build import load_chunks  # noqa: E402
from edugpt_rag.config import load_config  # noqa: E402
from edugpt_rag.generator import OpenAIGenerator, parse_output  # noqa: E402
from edugpt_rag.pipeline import read_jsonl, write_jsonl  # noqa: E402
from edugpt_rag.prompts import ABSTAIN_STRING  # noqa: E402


def questions(cfg):
    qs = read_jsonl(str(Path(cfg.corpus_dir) / "dev_questions.jsonl"))
    return qs[: cfg.max_eval_examples] if cfg.max_eval_examples else qs


def base_row(q):
    return {"id": q["id"], "question": q["question"], "gold_answers": q["answers"],
            "unanswerable": q["unanswerable"], "gold_doc_id": q["gold_doc_id"]}


def is_abstain(text):
    t = (text or "").strip().lower()
    return t == "" or ABSTAIN_STRING.lower().rstrip(".") in t or t in {"unanswerable", "no answer"}


# ---------------------------------------------------------------- vanilla GPT-4 (closed book)
def run_vanilla_gpt4(cfg, seed, model="gpt-4"):
    gen = OpenAIGenerator(model, cfg.generator.temperature, cfg.generator.max_tokens)
    rows = []
    for q in questions(cfg):
        out = gen.generate(q["question"], [], mode="closed_book", seed=seed)
        r = base_row(q)
        ans = out["raw"].strip()
        r.update({"answer": "" if is_abstain(ans) else ans, "abstained": is_abstain(ans), "contexts": [],
                  "cited_passage": None})
        rows.append(r)
    return rows


# ---------------------------------------------------------------- DPR + BART
def run_dpr_bart(cfg, seed, top_k=5):
    import faiss
    import numpy as np
    import torch
    from transformers import (BartForConditionalGeneration, BartTokenizer, DPRContextEncoder,
                              DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer)
    torch.manual_seed(seed)
    dev = cfg.device
    chunks = load_chunks(cfg.corpus_dir)
    ids = list(chunks)
    ctok = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    cenc = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(dev).eval()
    qtok = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    qenc = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(dev).eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(ids), 64):
            b = [chunks[j]["text"] for j in ids[i:i + 64]]
            enc = ctok(b, truncation=True, max_length=256, padding=True, return_tensors="pt").to(dev)
            embs.append(cenc(**enc).pooler_output.cpu().numpy())
    E = np.concatenate(embs).astype("float32")
    index = faiss.IndexFlatIP(E.shape[1])
    index.add(E)
    btok = BartTokenizer.from_pretrained("facebook/bart-large")
    bart = BartForConditionalGeneration.from_pretrained("facebook/bart-large").to(dev).eval()
    rows = []
    for q in questions(cfg):
        with torch.no_grad():
            qv = qenc(**qtok(q["question"], return_tensors="pt").to(dev)).pooler_output.cpu().numpy()
        _, I = index.search(qv.astype("float32"), top_k)
        ctx_ids = [ids[i] for i in I[0]]
        ctx = [chunks[c]["text"] for c in ctx_ids]
        src = f"question: {q['question']} context: " + " ".join(ctx)
        enc = btok(src, truncation=True, max_length=1024, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = bart.generate(**enc, max_new_tokens=32, num_beams=4)
        ans = btok.decode(out[0], skip_special_tokens=True).strip()
        r = base_row(q)
        r.update({"answer": "" if is_abstain(ans) else ans, "abstained": is_abstain(ans), "contexts": ctx,
                  "context_ids": ctx_ids, "context_doc_ids": [chunks[c]["doc_id"] for c in ctx_ids],
                  "cited_passage": None})
        rows.append(r)
    return rows


# ---------------------------------------------------------------- ColBERTv2 (PLAID) + GPT-3.5 reader
def run_colbert_qa(cfg, seed, top_k=5):
    from colbert import Indexer, Searcher
    from colbert.infra import ColBERTConfig, Run, RunConfig
    chunks = load_chunks(cfg.corpus_dir)
    ids = list(chunks)
    coll = Path(cfg.index_dir) / "colbert" / "collection.tsv"
    coll.parent.mkdir(parents=True, exist_ok=True)
    with open(coll, "w", encoding="utf-8") as f:
        for n, i in enumerate(ids):
            f.write(f"{n}\t{chunks[i]['text'].replace(chr(9), ' ').replace(chr(10), ' ')}\n")
    with Run().context(RunConfig(nranks=1, experiment="edugpt_colbert")):
        Indexer(checkpoint="colbert-ir/colbertv2.0", config=ColBERTConfig(nbits=2, doc_maxlen=300)).index(
            name="squad2.plaid", collection=str(coll), overwrite="reuse")
        searcher = Searcher(index="squad2.plaid", collection=str(coll))
    gen = OpenAIGenerator(cfg.generator.model, cfg.generator.temperature, cfg.generator.max_tokens)
    rows = []
    for q in questions(cfg):
        pids, _, _ = searcher.search(q["question"], k=top_k)
        ctx_ids = [ids[p] for p in pids]
        ctx = [chunks[c]["text"] for c in ctx_ids]
        out = gen.generate(q["question"], ctx, mode="cot", seed=seed)
        r = base_row(q)
        r.update({"answer": out["answer"], "abstained": out["abstained"], "contexts": ctx, "context_ids": ctx_ids,
                  "context_doc_ids": [chunks[c]["doc_id"] for c in ctx_ids], "cited_passage": out["cited_passage"]})
        rows.append(r)
    return rows


# ---------------------------------------------------------------- REALM (open-QA checkpoint)
def run_realm(cfg, seed):
    import torch
    from transformers import RealmForOpenQA, RealmRetriever, RealmTokenizer  # available in transformers 4.36
    torch.manual_seed(seed)
    name = "google/realm-orqa-nq-openqa"
    tok = RealmTokenizer.from_pretrained(name)
    retriever = RealmRetriever.from_pretrained(name)
    model = RealmForOpenQA.from_pretrained(name, retriever=retriever).to(cfg.device).eval()
    rows = []
    for q in questions(cfg):
        enc = tok([q["question"]], return_tensors="pt").to(cfg.device)
        with torch.no_grad():
            reader_output, predicted_ids = model(**enc, return_dict=False)
        ans = tok.decode(predicted_ids).strip()
        r = base_row(q)
        r.update({"answer": "" if is_abstain(ans) else ans, "abstained": is_abstain(ans),
                  "contexts": [], "cited_passage": None,
                  "note": "REALM retrieves from its own Wikipedia block records; contexts are not exposed"})
        rows.append(r)
    return rows


# ---------------------------------------------------------------- Self-RAG (short-form, threshold 0.5)
def run_self_rag(cfg, seed, top_k=5, threshold=0.5):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from edugpt_rag.build import build_or_load_indexes
    torch.manual_seed(seed)
    name = "selfrag/selfrag_llama2-7b"
    tok = AutoTokenizer.from_pretrained(name)
    lm = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(cfg.device).eval()
    chunks = load_chunks(cfg.corpus_dir)
    retr = build_or_load_indexes(cfg, chunks, seed)
    tid = {t: tok.convert_tokens_to_ids(t) for t in
           ["[Retrieval]", "[No Retrieval]", "[Relevant]", "[Irrelevant]", "[Fully supported]",
            "[Partially supported]", "[No support / Contradictory]", "[Utility:5]", "[Utility:4]",
            "[Utility:3]", "[Utility:2]", "[Utility:1]"]}

    def gen(prompt, max_new=64):
        enc = tok(prompt, return_tensors="pt").to(cfg.device)
        with torch.no_grad():
            out = lm.generate(**enc, max_new_tokens=max_new, do_sample=False, output_scores=True,
                              return_dict_in_generate=True)
        text = tok.decode(out.sequences[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)
        probs = [torch.softmax(s[0].float(), -1).cpu().numpy() for s in out.scores]
        return text, probs

    def clean(t):
        for k in tid:
            t = t.replace(k, "")
        return t.replace("</s>", "").replace("<paragraph>", "").replace("</paragraph>", "").strip()

    rows = []
    for q in questions(cfg):
        prompt = f"### Instruction:\n{q['question']}\n\n### Response:\n"
        text, probs = gen(prompt, max_new=1)
        p_ret = probs[0][tid["[Retrieval]"]] / (probs[0][tid["[Retrieval]"]] + probs[0][tid["[No Retrieval]"]])
        r = base_row(q)
        if p_ret <= threshold:
            text, _ = gen(prompt + "[No Retrieval]")
            ans, ctx, ctx_ids = clean(text), [], []
        else:
            ctx_ids = retr.retrieve(q["question"])[:top_k]
            best, best_s, ctx = "", -1e9, []
            for cid in ctx_ids:
                p = chunks[cid]["text"]
                text, probs = gen(prompt + f"[Retrieval]<paragraph>{p}</paragraph>")
                flat = np.max(np.stack(probs), axis=0)
                rel = flat[tid["[Relevant]"]] / (flat[tid["[Relevant]"]] + flat[tid["[Irrelevant]"]] + 1e-9)
                sup = (flat[tid["[Fully supported]"]] + 0.5 * flat[tid["[Partially supported]"]]) / (
                    flat[tid["[Fully supported]"]] + flat[tid["[Partially supported]"]] +
                    flat[tid["[No support / Contradictory]"]] + 1e-9)
                score = rel + sup
                if score > best_s:
                    best, best_s, ctx = clean(text), score, [p]
            ans = best
        r.update({"answer": "" if is_abstain(ans) else ans, "abstained": is_abstain(ans), "contexts": ctx,
                  "context_ids": ctx_ids, "context_doc_ids": [chunks[c]["doc_id"] for c in ctx_ids],
                  "cited_passage": 1 if ctx else None, "p_retrieval": float(p_ret)})
        rows.append(r)
    return rows


# ---------------------------------------------------------------- FiD / Atlas adapters
def export_for_reader(cfg, seed, out_path, top_k=100):
    """Writes questions with hybrid-retrieved passages in the FiD/Atlas JSON format."""
    chunks = load_chunks(cfg.corpus_dir)
    from edugpt_rag.build import build_or_load_indexes
    retr = build_or_load_indexes(cfg, chunks, seed)
    cfg.retrieval.n_fused = top_k
    data = []
    for q in questions(cfg):
        cids = retr.retrieve(q["question"])
        data.append({"id": q["id"], "question": q["question"], "answers": q["answers"] or [""],
                     "ctxs": [{"id": c, "title": "", "text": chunks[c]["text"]} for c in cids]})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(data))
    return out_path


def import_reader_outputs(cfg, pred_file, top_k=5):
    """Reads FiD (`id\tanswer`) or Atlas (jsonl with id, generation) predictions."""
    chunks = load_chunks(cfg.corpus_dir)
    preds = {}
    for line in open(pred_file, encoding="utf-8"):
        if line.lstrip().startswith("{"):
            d = json.loads(line)
            preds[str(d.get("id") or d.get("query_id"))] = d.get("generation") or d.get("answer", "")
        elif "\t" in line:
            k, v = line.rstrip("\n").split("\t", 1)
            preds[k] = v
    exported = {d["id"]: d for d in json.load(open(Path(pred_file).with_name("reader_input.json")))}
    rows = []
    for q in questions(cfg):
        ans = preds.get(q["id"], "")
        cids = [c["id"] for c in exported[q["id"]]["ctxs"][:top_k]]
        r = base_row(q)
        r.update({"answer": "" if is_abstain(ans) else ans, "abstained": is_abstain(ans),
                  "contexts": [chunks[c]["text"] for c in cids], "context_ids": cids,
                  "context_doc_ids": [chunks[c]["doc_id"] for c in cids], "cited_passage": None})
        rows.append(r)
    return rows


RUNNERS = {"vanilla_gpt4": run_vanilla_gpt4, "dpr_bart": run_dpr_bart, "colbert_qa": run_colbert_qa,
           "realm": run_realm, "self_rag": run_self_rag}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/squad2.yaml")
    ap.add_argument("--baseline", required=True, choices=list(RUNNERS) + ["bm25_gpt35", "fid", "atlas"])
    ap.add_argument("--stage", choices=["export", "import"], help="for fid / atlas only")
    ap.add_argument("--pred_file", help="reader predictions to import (fid / atlas)")
    ap.add_argument("--seeds", nargs="*", type=int)
    a = ap.parse_args()
    cfg = load_config(a.config)
    seeds = a.seeds or cfg.seeds
    if a.baseline == "bm25_gpt35":
        subprocess.run([sys.executable, "experiments/run_ablation.py", "--table", "2"], check=True)
        return
    for seed in seeds:
        out = Path(cfg.output_dir) / a.baseline / f"seed{seed}"
        if a.baseline in ("fid", "atlas"):
            if a.stage == "export":
                p = export_for_reader(cfg, seed, str(out / "reader_input.json"))
                print(f"wrote {p}; run the official {a.baseline} reader on it (docs/BASELINES.md)")
                continue
            rows = import_reader_outputs(cfg, a.pred_file or str(out / "reader_predictions.txt"))
        else:
            rows = RUNNERS[a.baseline](cfg, seed)
        write_jsonl(rows, str(out / "predictions.jsonl"))
        print(f"[{a.baseline}] seed {seed}: {len(rows)} predictions -> {out}")


if __name__ == "__main__":
    main()
