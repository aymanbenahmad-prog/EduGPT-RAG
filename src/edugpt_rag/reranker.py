"""Cross-encoder re-ranker (Eqs. 10 to 12)."""
from __future__ import annotations

from typing import Dict, List, Sequence


class CrossEncoderReranker:
    def __init__(self, model_path: str, max_length: int = 512, device: str = "cuda"):
        from sentence_transformers import CrossEncoder
        # num_labels=1 with a sigmoid activation gives s_CE(q, p) in (0, 1), Eq. 10
        self.model = CrossEncoder(model_path, num_labels=1, max_length=max_length, device=device)

    def score(self, query: str, passages: Sequence[str]) -> List[float]:
        import torch
        return [float(s) for s in self.model.predict([(query, p) for p in passages],
                                                     activation_fct=torch.nn.Sigmoid())]

    def rerank(self, query: str, cand_ids: Sequence[str], id2text: Dict[str, str], top_k: int = 5) -> List[str]:
        scores = self.score(query, [id2text[c] for c in cand_ids])
        order = sorted(range(len(cand_ids)), key=lambda i: (-scores[i], i))
        return [cand_ids[i] for i in order[:top_k]]
