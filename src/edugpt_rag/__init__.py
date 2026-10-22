"""EduGPT-RAG: hybrid retrieval, cross-encoder re-ranking, CoT generation and NLI verification.

The package implements the pipeline described in the manuscript (Sections 3.1 to 3.9,
Algorithm 1). Heavy dependencies (torch, transformers, sentence-transformers, faiss,
pyserini, openai) are imported lazily so that the pure-Python parts (chunking, RRF,
metrics, verification logic) can be tested without them.
"""

__version__ = "1.0.0"
