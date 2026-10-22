"""EduGPT-RAG inference pipeline (Algorithm 1)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .verifier import citation_correct, faithfulness, is_hallucinated, sentence_scores


class EduGPTRAG:
    """Composes retriever, re-ranker, generator and verifier.

    Any component can be swapped by a stub with the same interface, which is how the unit
    tests run without GPUs or API access.
    """

    def __init__(self, cfg, retriever, reranker, generator, nli, chunks: Dict[str, Dict]):
        self.cfg, self.retriever, self.reranker = cfg, retriever, reranker
        self.generator, self.nli, self.chunks = generator, nli, chunks
        self.id2text = {k: v["text"] for k, v in chunks.items()}

    def answer(self, qid: str, question: str, seed: int = 42) -> Dict:
        c = self.cfg
        cand = self.retriever.retrieve(question, use_dense=c.use_dense, use_sparse=c.use_sparse)  # lines 1-4
        if c.use_rerank:                                                                               # line 5
            ctx_ids = self.reranker.rerank(question, cand, self.id2text, top_k=c.reranker.top_k)
        else:
            ctx_ids = cand[: c.reranker.top_k]
        ctx = [self.id2text[i] for i in ctx_ids]
        first = self.generator.generate(question, ctx, mode="cot" if c.use_cot else "direct", seed=seed)  # 6
        S = sentence_scores(first["answer"], ctx, self.nli) if first["answer"] else None
        f = faithfulness(S) if S is not None else 0.0                                                   # 7
        final, regenerated = first, False
        if c.use_verification and f < c.verifier.tau:                                                  # 8-10
            final = self.generator.generate(question, ctx, mode="constrained", seed=seed)
            regenerated = True
            S = sentence_scores(final["answer"], ctx, self.nli) if final["answer"] else None
        rec = {
            "id": qid, "question": question,
            "fused_candidates": cand,
            "fused_doc_ids": [self.chunks[i]["doc_id"] for i in cand],
            "context_ids": ctx_ids,
            "context_doc_ids": [self.chunks[i]["doc_id"] for i in ctx_ids],
            "first_answer": first["answer"], "first_faith": f, "regenerated": regenerated,
            "answer": final["answer"], "abstained": final["abstained"] or not final["answer"],
            "cited_passage": final.get("cited_passage"),
            "raw": final["raw"],
        }
        if S is not None:
            rec["faith"] = faithfulness(S)
            rec["hallucinated"] = is_hallucinated(S, c.verifier.hallucination_threshold)
            rec["citation_correct"] = citation_correct(S, rec["cited_passage"], c.verifier.citation_threshold)
            rec["sentence_scores"] = S.tolist()
        else:
            rec.update({"faith": None, "hallucinated": False, "citation_correct": None, "sentence_scores": []})
        return rec


def write_jsonl(rows: List[Dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
