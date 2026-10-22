"""Configuration loading. All defaults equal the values stated in the manuscript."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class ChunkingConfig:
    window: int = 512          # w, whitespace-delimited tokens (Section 3.2)
    overlap: int = 128         # s, minimum overlap between consecutive chunks (Eq. 3)


@dataclass
class RetrievalConfig:
    sbert_model: str = "sentence-transformers/all-mpnet-base-v2"   # d = 768
    embedding_dim: int = 768
    k_dense: int = 20          # k_d
    k_sparse: int = 20         # k_s
    rrf_kappa: int = 60        # kappa (Eq. 8)
    n_fused: int = 20          # n candidates forwarded to the re-ranker (Eq. 9)
    faiss_nlist: int = 4096    # IVF4096
    faiss_pq_m: int = 64       # PQ64
    faiss_nprobe: int = 128
    faiss_train_sample: int = 100_000
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    bm25_backend: str = "pyserini"   # "pyserini" or "python" (pure-Python fallback)


@dataclass
class RerankerConfig:
    base_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    checkpoint: Optional[str] = None   # path of the MS MARCO fine-tuned checkpoint
    top_k: int = 5                     # K (Eq. 11)
    max_length: int = 512


@dataclass
class GeneratorConfig:
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    max_tokens: int = 512


@dataclass
class VerifierConfig:
    nli_model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    tau: float = 0.75                  # faithfulness threshold (Eq. 16)
    hallucination_threshold: float = 0.50
    citation_threshold: float = 0.75
    max_length: int = 512


@dataclass
class Config:
    dataset: str = "squad2"
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    # pipeline switches used by the ablations (Tables 2 and 4)
    use_dense: bool = True
    use_sparse: bool = True
    use_rerank: bool = True
    use_cot: bool = True
    use_verification: bool = True
    corpus_dir: str = "data/processed/squad2"
    index_dir: str = "indexes/squad2"
    output_dir: str = "results/raw"
    max_eval_examples: Optional[int] = None   # None = full development split
    device: str = "cuda"


def _merge(dc, values: dict):
    for k, v in values.items():
        if not hasattr(dc, k):
            raise KeyError(f"Unknown configuration key: {k}")
        cur = getattr(dc, k)
        if dataclasses.is_dataclass(cur) and isinstance(v, dict):
            _merge(cur, v)
        else:
            setattr(dc, k, v)
    return dc


def load_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> Config:
    cfg = Config()
    if path:
        with open(path, "r", encoding="utf-8") as f:
            _merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        _merge(cfg, overrides)
    return cfg


def to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)


def save_config(cfg: Config, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False)
