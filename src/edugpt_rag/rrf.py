"""Reciprocal Rank Fusion (Eqs. 8 and 9). Ranks are 1-based."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple


def rrf_fuse(rankings: Sequence[Sequence[str]], kappa: int = 60, top_n: int = 20) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, pid in enumerate(ranking, start=1):
            scores[pid] += 1.0 / (kappa + rank)
    # deterministic tie-break on passage id
    fused = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return fused[:top_n]
