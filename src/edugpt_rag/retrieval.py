"""Sparse (BM25, Eq. 4) and dense (SBERT + FAISS IVF-PQ, Eqs. 5 to 7) retrieval."""
from __future__ import annotations

import json
import logging
import math
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .rrf import rrf_fuse

log = logging.getLogger(__name__)


def bm25_tokenize(text: str) -> List[str]:
    """Lower-casing and whitespace normalisation, as stated in Section 4.1."""
    return text.lower().split()


class PythonBM25:
    """Reference BM25 implementation of Eq. 4 (used for tests and small corpora)."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b

    def index(self, ids: Sequence[str], texts: Sequence[str]) -> None:
        self.ids = list(ids)
        self.tf = [Counter(bm25_tokenize(t)) for t in texts]
        self.len = np.array([sum(c.values()) for c in self.tf], dtype=float)
        self.avg = float(self.len.mean()) if len(self.len) else 0.0
        df: Dict[str, int] = defaultdict(int)
        for c in self.tf:
            for t in c:
                df[t] += 1
        n = len(self.tf)
        # Lucene-style non-negative IDF, as used by Pyserini
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.post: Dict[str, List[int]] = defaultdict(list)
        for i, c in enumerate(self.tf):
            for t in c:
                self.post[t].append(i)

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        scores: Dict[int, float] = defaultdict(float)
        for t in set(bm25_tokenize(query)):
            if t not in self.idf:
                continue
            for i in self.post[t]:
                f = self.tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                scores[i] += self.idf[t] * f * (self.k1 + 1) / denom
        top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [(self.ids[i], s) for i, s in top]


class PyseriniBM25:
    """BM25 through a Pyserini/Lucene index (k1 = 1.2, b = 0.75)."""

    def __init__(self, index_dir: str, k1: float = 1.2, b: float = 0.75):
        self.index_dir, self.k1, self.b = index_dir, k1, b

    def index(self, ids: Sequence[str], texts: Sequence[str]) -> None:
        coll = Path(self.index_dir) / "collection"
        coll.mkdir(parents=True, exist_ok=True)
        with open(coll / "docs.jsonl", "w", encoding="utf-8") as f:
            for i, t in zip(ids, texts):
                f.write(json.dumps({"id": i, "contents": " ".join(bm25_tokenize(t))}) + "\n")
        subprocess.run(["python", "-m", "pyserini.index.lucene", "--collection", "JsonCollection",
                        "--input", str(coll), "--index", str(Path(self.index_dir) / "lucene"),
                        "--generator", "DefaultLuceneDocumentGenerator", "--threads", "8",
                        "--storeRaw"], check=True)
        self._open()

    def _open(self):
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher(str(Path(self.index_dir) / "lucene"))
        self.searcher.set_bm25(self.k1, self.b)

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        if not hasattr(self, "searcher"):
            self._open()
        hits = self.searcher.search(" ".join(bm25_tokenize(query)), k=k)
        return [(h.docid, float(h.score)) for h in hits]


class DenseIndex:
    """SBERT encoder + FAISS IVF-PQ with inner product on L2-normalised vectors (cosine, Eq. 6)."""

    def __init__(self, model_name: str, nlist: int, pq_m: int, nprobe: int,
                 train_sample: int, device: str = "cuda", seed: int = 42):
        self.model_name, self.nlist, self.pq_m = model_name, nlist, pq_m
        self.nprobe, self.train_sample, self.device, self.seed = nprobe, train_sample, device, seed
        self._encoder = None

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model_name, device=self.device)
            self._encoder.max_seq_length = 512
        return self._encoder

    def encode(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        # SentenceTransformer applies mean pooling over non-padding tokens for all-mpnet-base-v2
        return np.asarray(self.encoder.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                                              show_progress_bar=False), dtype="float32")

    def build(self, ids: Sequence[str], texts: Sequence[str]) -> Dict:
        import faiss
        self.ids = list(ids)
        emb = self.encode(texts)
        d = emb.shape[1]
        n = emb.shape[0]
        rng = np.random.default_rng(self.seed)
        sample = emb[rng.choice(n, size=min(self.train_sample, n), replace=False)]
        if sample.shape[0] < 39 * self.nlist:
            log.warning("FAISS: %d training vectors for nlist=%d (fewer than the 39*nlist recommended by FAISS).",
                        sample.shape[0], self.nlist)
        quantizer = faiss.IndexFlatIP(d)
        self.index = faiss.IndexIVFPQ(quantizer, d, self.nlist, self.pq_m, 8, faiss.METRIC_INNER_PRODUCT)
        self.index.cp.seed = self.seed            # coarse k-means seed
        self.index.pq.cp.seed = self.seed         # product-quantiser k-means seed
        self.index.train(sample)
        self.index.add(emb)
        self.index.nprobe = self.nprobe
        return {"n_vectors": int(n), "dim": int(d), "n_train": int(sample.shape[0]),
                "nlist": self.nlist, "pq_m": self.pq_m, "nprobe": self.nprobe}

    def save(self, path: str) -> None:
        import faiss
        Path(path).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(Path(path) / "ivfpq.index"))
        with open(Path(path) / "ids.json", "w") as f:
            json.dump(self.ids, f)

    def load(self, path: str) -> None:
        import faiss
        self.index = faiss.read_index(str(Path(path) / "ivfpq.index"))
        self.index.nprobe = self.nprobe
        with open(Path(path) / "ids.json") as f:
            self.ids = json.load(f)

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        q = self.encode([query])
        scores, idx = self.index.search(q, k)
        return [(self.ids[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]


class HybridRetriever:
    def __init__(self, dense, sparse, cfg):
        self.dense, self.sparse, self.cfg = dense, sparse, cfg

    def retrieve(self, query: str, use_dense: bool = True, use_sparse: bool = True) -> List[str]:
        r = self.cfg.retrieval
        rankings = []
        if use_dense:
            rankings.append([pid for pid, _ in self.dense.search(query, r.k_dense)])
        if use_sparse:
            rankings.append([pid for pid, _ in self.sparse.search(query, r.k_sparse)])
        if not rankings:
            raise ValueError("At least one retriever must be enabled")
        if len(rankings) == 1:
            return rankings[0][: r.n_fused]
        return [pid for pid, _ in rrf_fuse(rankings, kappa=r.rrf_kappa, top_n=r.n_fused)]
