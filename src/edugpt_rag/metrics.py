"""Evaluation metrics.

SQuAD 2.0 EM/F1 follow the official evaluate-v2.0.py normalisation exactly. QASPER token F1
follows the official qasper_evaluator.py. Retrieval Recall@K and MRR use the gold paragraph:
a hit is any retrieved chunk whose doc_id equals the gold paragraph id.
"""
from __future__ import annotations

import collections
import re
import string
from typing import Dict, Iterable, List, Optional, Sequence


def normalize_answer(s: str) -> str:
    def remove_articles(t):
        return re.sub(re.compile(r"\b(a|an|the)\b", re.UNICODE), " ", t)

    def white_space_fix(t):
        return " ".join(t.split())

    def remove_punc(t):
        exclude = set(string.punctuation)
        return "".join(ch for ch in t if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def get_tokens(s: str) -> List[str]:
    return normalize_answer(s).split() if s else []


def compute_exact(a_gold: str, a_pred: str) -> int:
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold: str, a_pred: str) -> float:
    gold_toks, pred_toks = get_tokens(a_gold), get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return float(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def squad_scores(gold_answers: Sequence[str], pred: str) -> Dict[str, float]:
    """gold_answers empty means unanswerable; the gold is then the empty string (official script)."""
    golds = [a for a in gold_answers if normalize_answer(a)] or [""]
    return {"em": max(compute_exact(g, pred) for g in golds),
            "f1": max(compute_f1(g, pred) for g in golds)}


def aggregate_squad(rows: Iterable[Dict]) -> Dict[str, float]:
    rows = list(rows)
    n = len(rows)
    return {"EM": 100.0 * sum(r["em"] for r in rows) / n, "F1": 100.0 * sum(r["f1"] for r in rows) / n, "n": n}


def recall_at_k(retrieved_doc_ids: Sequence[str], gold_doc_id: str, k: int) -> int:
    return int(gold_doc_id in list(retrieved_doc_ids)[:k])


def reciprocal_rank(retrieved_doc_ids: Sequence[str], gold_doc_id: str) -> float:
    for r, d in enumerate(retrieved_doc_ids, start=1):
        if d == gold_doc_id:
            return 1.0 / r
    return 0.0


def abstention_metrics(rows: Iterable[Dict]) -> Dict[str, float]:
    """Precision over all abstentions (answerable and unanswerable); recall over unanswerable questions."""
    rows = list(rows)
    abst = [r for r in rows if r["abstained"]]
    unans = [r for r in rows if r["unanswerable"]]
    tp = sum(1 for r in abst if r["unanswerable"])
    p = tp / len(abst) if abst else 0.0
    rec = tp / len(unans) if unans else 0.0
    f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
    return {"abstention_rate_unanswerable": 100.0 * rec, "precision": p, "recall": rec, "f1": f1,
            "n_abstained": len(abst), "n_unanswerable": len(unans)}


# ---- QASPER (official qasper_evaluator.py logic) ----
def qasper_token_f1(pred: str, gold: str) -> float:
    return compute_f1(gold, pred)


def qasper_answer_type(ans: Dict) -> str:
    if ans.get("unanswerable"):
        return "none"
    if ans.get("extractive_spans"):
        return "extractive"
    if ans.get("free_form_answer"):
        return "abstractive"
    if ans.get("yes_no") is not None:
        return "boolean"
    return "none"


def qasper_gold_string(ans: Dict) -> str:
    t = qasper_answer_type(ans)
    if t == "extractive":
        return ", ".join(ans["extractive_spans"])
    if t == "abstractive":
        return ans["free_form_answer"]
    if t == "boolean":
        return "Yes" if ans["yes_no"] else "No"
    return "Unanswerable"
