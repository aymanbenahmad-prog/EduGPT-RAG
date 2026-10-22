"""Helpers that assemble the full system from a configuration and a built corpus."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .config import Config
from .retrieval import DenseIndex, HybridRetriever, PyseriniBM25, PythonBM25


def load_chunks(corpus_dir: str) -> Dict[str, Dict]:
    chunks = {}
    with open(Path(corpus_dir) / "chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunks[c["chunk_id"]] = c
    return chunks


def build_or_load_indexes(cfg: Config, chunks: Dict[str, Dict], seed: int) -> HybridRetriever:
    r = cfg.retrieval
    ids = list(chunks)
    texts = [chunks[i]["text"] for i in ids]
    dense_dir = Path(cfg.index_dir) / f"faiss_seed{seed}"
    dense = DenseIndex(r.sbert_model, r.faiss_nlist, r.faiss_pq_m, r.faiss_nprobe, r.faiss_train_sample,
                       device=cfg.device, seed=seed)
    if (dense_dir / "ivfpq.index").exists():
        dense.load(str(dense_dir))
    else:
        stats = dense.build(ids, texts)
        dense.save(str(dense_dir))
        (dense_dir / "build_stats.json").write_text(json.dumps(stats, indent=2))
    if r.bm25_backend == "pyserini":
        sparse = PyseriniBM25(str(Path(cfg.index_dir) / "bm25"), r.bm25_k1, r.bm25_b)
        if not (Path(cfg.index_dir) / "bm25" / "lucene").exists():
            sparse.index(ids, texts)
    else:
        sparse = PythonBM25(r.bm25_k1, r.bm25_b)
        sparse.index(ids, texts)
    return HybridRetriever(dense, sparse, cfg)


def build_system(cfg: Config, seed: int):
    from .generator import OpenAIGenerator
    from .pipeline import EduGPTRAG
    from .reranker import CrossEncoderReranker
    from .verifier import NLIScorer
    chunks = load_chunks(cfg.corpus_dir)
    retriever = build_or_load_indexes(cfg, chunks, seed)
    ckpt = cfg.reranker.checkpoint or cfg.reranker.base_model
    if cfg.reranker.checkpoint and "{seed}" in cfg.reranker.checkpoint:
        ckpt = cfg.reranker.checkpoint.format(seed=seed)
    reranker = CrossEncoderReranker(ckpt, cfg.reranker.max_length, cfg.device) if cfg.use_rerank else None
    gen = OpenAIGenerator(cfg.generator.model, cfg.generator.temperature, cfg.generator.max_tokens)
    nli = NLIScorer(cfg.verifier.nli_model, cfg.verifier.max_length, cfg.device)
    return EduGPTRAG(cfg, retriever, reranker, gen, nli, chunks)
