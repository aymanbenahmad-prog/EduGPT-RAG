"""NLI-based faithfulness verification (Eqs. 15 and 16) and the hallucination / citation metrics.

Entailment probabilities are obtained by a softmax over the three NLI logits; the entailment
index is read from the model configuration (id2label) and never hard-coded.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .chunking import split_sentences


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class NLIScorer:
    """Returns P(entailment | premise, hypothesis) for a batch of pairs."""

    def __init__(self, model_name: str, max_length: int = 512, device: str = "cuda", batch_size: int = 16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        if "entailment" not in labels:
            raise ValueError(f"Model {model_name} has no 'entailment' label: {labels}")
        self.ent = labels["entailment"]
        self.device, self.max_length, self.batch_size = device, max_length, batch_size

    def __call__(self, pairs: Sequence[tuple]) -> np.ndarray:
        out = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            enc = self.tok([p for p, _ in batch], [h for _, h in batch], truncation="only_first",
                           max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                logits = self.model(**enc).logits.float().cpu().numpy()
            out.append(softmax(logits)[:, self.ent])
        return np.concatenate(out) if out else np.zeros(0)


def sentence_scores(answer: str, contexts: Sequence[str], nli: Callable) -> np.ndarray:
    """Matrix S[s, c] = P(entail | context c, answer sentence s)."""
    sents = split_sentences(answer)
    if not sents or not contexts:
        return np.zeros((0, len(contexts)))
    pairs = [(c, s) for s in sents for c in contexts]
    return np.asarray(nli(pairs)).reshape(len(sents), len(contexts))


def faithfulness(S: np.ndarray) -> float:
    """Eq. 15: mean over answer sentences of the max entailment over the K contexts."""
    return float(S.max(axis=1).mean()) if S.size else 0.0


def is_hallucinated(S: np.ndarray, threshold: float = 0.50) -> bool:
    """Section 4.5: at least one sentence whose max entailment over the contexts is < 0.50."""
    return bool(S.size and (S.max(axis=1) < threshold).any())


def citation_correct(S: np.ndarray, cited: Optional[int], threshold: float = 0.75) -> Optional[bool]:
    """Every answer sentence must be entailed (>= 0.75) by the cited passage (1-based index)."""
    if cited is None or not S.size or not 1 <= cited <= S.shape[1]:
        return None if cited is None else False
    return bool((S[:, cited - 1] >= threshold).all())


def verify(answer: str, contexts: Sequence[str], nli: Callable, tau: float = 0.75) -> Dict:
    S = sentence_scores(answer, contexts, nli)
    f = faithfulness(S)
    return {"faith": f, "regenerate": f < tau, "scores": S.tolist()}
