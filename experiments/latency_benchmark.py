"""Per-component latency by corpus size (Table 8) and a concurrent-load probe (Section 4.8).

Corpus sizes above the SQuAD 2.0 corpus are obtained by adding distractor chunks drawn from
the SQuAD 2.0 training paragraphs and, beyond that, from a user-supplied passage file
(--extra_passages, one passage per line, e.g. a Wikipedia dump). Each component is timed
separately over --n_queries development questions (median and mean, milliseconds).
Index size on disk is reported for every corpus size.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edugpt_rag.build import load_chunks  # noqa: E402
from edugpt_rag.config import load_config  # noqa: E402
from edugpt_rag.pipeline import read_jsonl  # noqa: E402


def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return out, 1000 * (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/squad2.yaml")
    ap.add_argument("--sizes", nargs="*", type=int, default=[10_000, 50_000, 100_000, 500_000, 1_000_000])
    ap.add_argument("--extra_passages", default=None)
    ap.add_argument("--n_queries", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=10)
    a = ap.parse_args()
    cfg = load_config(a.config)
    from edugpt_rag.generator import OpenAIGenerator
    from edugpt_rag.reranker import CrossEncoderReranker
    from edugpt_rag.retrieval import DenseIndex, HybridRetriever, PythonBM25, PyseriniBM25
    from edugpt_rag.rrf import rrf_fuse
    from edugpt_rag.verifier import NLIScorer, sentence_scores
    base = load_chunks(cfg.corpus_dir)
    texts = [c["text"] for c in base.values()]
    if a.extra_passages:
        texts += [l.strip() for l in open(a.extra_passages, encoding="utf-8") if l.strip()]
    qs = read_jsonl(str(Path(cfg.corpus_dir) / "dev_questions.jsonl"))[: a.n_queries]
    r = cfg.retrieval
    rer = CrossEncoderReranker(cfg.reranker.checkpoint or cfg.reranker.base_model, device=cfg.device)
    gen = OpenAIGenerator(cfg.generator.model, cfg.generator.temperature, cfg.generator.max_tokens, cache_dir=None)
    nli = NLIScorer(cfg.verifier.nli_model, device=cfg.device)
    results = {}
    for size in a.sizes:
        if size > len(texts):
            print(f"skip {size}: only {len(texts)} passages available")
            continue
        ids = [f"p{i}" for i in range(size)]
        sub = texts[:size]
        dense = DenseIndex(r.sbert_model, min(r.faiss_nlist, size // 39), r.faiss_pq_m, r.faiss_nprobe,
                           r.faiss_train_sample, device=cfg.device)
        dense.build(ids, sub)
        idx_dir = Path(cfg.index_dir) / f"latency_{size}"
        dense.save(str(idx_dir / "faiss"))
        sparse = PyseriniBM25(str(idx_dir / "bm25")) if r.bm25_backend == "pyserini" else PythonBM25()
        sparse.index(ids, sub)
        id2text = dict(zip(ids, sub))
        comp = {k: [] for k in ["sbert", "faiss", "bm25", "rrf", "cross_encoder", "generation", "verification"]}
        for q in qs:
            qv, t = timed(dense.encode, [q["question"]]); comp["sbert"].append(t)
            (sc, ix), t = timed(dense.index.search, qv, r.k_dense); comp["faiss"].append(t)
            d_ids = [ids[i] for i in ix[0] if i >= 0]
            s_hits, t = timed(sparse.search, q["question"], r.k_sparse); comp["bm25"].append(t)
            fused, t = timed(rrf_fuse, [d_ids, [h for h, _ in s_hits]], r.rrf_kappa, r.n_fused); comp["rrf"].append(t)
            cand = [p for p, _ in fused]
            ctx_ids, t = timed(rer.rerank, q["question"], cand, id2text, cfg.reranker.top_k); comp["cross_encoder"].append(t)
            ctx = [id2text[c] for c in ctx_ids]
            out, t = timed(gen.generate, q["question"], ctx, "cot", 42); comp["generation"].append(t)
            _, t = timed(sentence_scores, out["answer"] or "-", ctx, nli); comp["verification"].append(t)
        size_mb = sum(f.stat().st_size for f in (idx_dir / "faiss").glob("*")) / 1e6
        results[size] = {k: {"median": st.median(v), "mean": st.mean(v)} for k, v in comp.items()}
        results[size]["total_mean"] = sum(results[size][k]["mean"] for k in comp)
        results[size]["faiss_index_mb"] = size_mb
        print(size, json.dumps(results[size]))
        if size == 100_000 or size == max(s for s in a.sizes if s <= len(texts)):
            retr = HybridRetriever(dense, sparse, cfg)

            def end_to_end(q):
                t0 = time.perf_counter()
                c = retr.retrieve(q["question"])
                c = rer.rerank(q["question"], c, id2text, cfg.reranker.top_k)
                ctx = [id2text[x] for x in c]
                o = gen.generate(q["question"], ctx, "cot", 42)
                sentence_scores(o["answer"] or "-", ctx, nli)
                return 1000 * (time.perf_counter() - t0)

            single = [end_to_end(q) for q in qs[:50]]
            with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
                conc = list(ex.map(end_to_end, qs[:50]))
            results[f"concurrency_{size}"] = {"single_mean_ms": st.mean(single), "concurrent_mean_ms": st.mean(conc),
                                              "workers": a.concurrency}
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/tables/latency.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
