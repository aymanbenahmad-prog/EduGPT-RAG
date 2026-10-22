"""Hierarchical, sentence-aligned chunking (Section 3.2, Eqs. 2 and 3).

Tokens are whitespace-delimited. Each chunk p_{i,j} holds whole sentences with
|p_{i,j}| <= w tokens. Consecutive chunks overlap by at least s tokens
(|p_{i,j} ∩ p_{i,j+1}| >= s) whenever the preceding chunk is long enough to allow it.
A document of at most w tokens therefore yields exactly one chunk. A single sentence
longer than w tokens is split into w-token pieces.
"""
from __future__ import annotations

import re
from typing import Dict, List

_FALLBACK_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str, use_nltk: bool = True) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if use_nltk:
        try:
            from nltk.tokenize import sent_tokenize
            return [s for s in sent_tokenize(text) if s.strip()]
        except LookupError:
            pass
    return [s for s in _FALLBACK_SPLIT.split(text) if s.strip()]


def _pieces(sentences: List[str], window: int) -> List[List[str]]:
    """Token lists per sentence, splitting any sentence longer than the window."""
    out = []
    for s in sentences:
        toks = s.split()
        for i in range(0, len(toks), window):
            out.append(toks[i:i + window])
    return out


def chunk_document(doc_id: str, text: str, window: int = 512, overlap: int = 128,
                   use_nltk: bool = True) -> List[Dict]:
    if not 0 <= overlap < window:
        raise ValueError("overlap s must satisfy 0 <= s < w")
    sents = _pieces(split_sentences(text, use_nltk), window)
    chunks: List[Dict] = []
    i = 0
    n = len(sents)
    while i < n:
        j, size = i, 0
        while j < n and size + len(sents[j]) <= window:
            size += len(sents[j])
            j += 1
        toks = [t for s in sents[i:j] for t in s]
        chunks.append({"chunk_id": f"{doc_id}::{len(chunks)}", "doc_id": doc_id,
                       "text": " ".join(toks), "n_tokens": len(toks),
                       "first_sentence": i, "last_sentence": j - 1})
        if j >= n:
            break
        # next start: the latest sentence k > i whose suffix k..j-1 still holds >= s tokens
        k, tail = j, 0
        while k - 1 > i and tail < overlap:
            k -= 1
            tail += len(sents[k])
        i = k if tail >= overlap else max(i + 1, k)
    return chunks


def chunk_corpus(docs: List[Dict], window: int = 512, overlap: int = 128,
                 use_nltk: bool = True) -> List[Dict]:
    out: List[Dict] = []
    for d in docs:
        out.extend(chunk_document(d["doc_id"], d["text"], window, overlap, use_nltk))
    return out
