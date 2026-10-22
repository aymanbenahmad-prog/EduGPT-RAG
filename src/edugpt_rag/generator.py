"""Frozen GPT-3.5-turbo generator accessed through the OpenAI API (Eqs. 13 and 14)."""
from __future__ import annotations

import json
import os
import re
import time
from hashlib import sha256
from pathlib import Path
from typing import Dict, Optional, Sequence

from .prompts import (ABSTAIN_STRING, CLOSED_BOOK_PROMPT, CONSTRAINED_PROMPT, COT_PROMPT,
                      DIRECT_PROMPT, format_passages)

_ANSWER = re.compile(r"Answer:\s*(.*?)(?:\s*Confidence:|$)", re.S | re.I)
_PASSAGE = re.compile(r"Most relevant passage:\s*\[?(\d+)\]?", re.I)
_CONF = re.compile(r"Confidence:\s*([01](?:\.\d+)?)", re.I)


def parse_output(text: str) -> Dict:
    """Extract answer, cited passage index (1-based) and confidence from the CoT format."""
    t = text.strip()
    m = _ANSWER.search(t)
    answer = (m.group(1) if m else t).strip().strip('"').strip()
    p = _PASSAGE.search(t)
    c = _CONF.search(t)
    abstained = ABSTAIN_STRING.lower().rstrip(".") in t.lower() or answer.lower() in {"", "unanswerable"}
    return {"answer": "" if abstained else answer, "cited_passage": int(p.group(1)) if p else None,
            "confidence": float(c.group(1)) if c else None, "abstained": abstained, "raw": text}


class OpenAIGenerator:
    """Chat-completion wrapper with an on-disk cache so that re-runs do not re-query the API."""

    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.1, max_tokens: int = 512,
                 cache_dir: Optional[str] = "results/cache/openai"):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.cache = Path(cache_dir) if cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)

    def _complete(self, prompt: str, seed: int) -> str:
        key = sha256(json.dumps([self.model, self.temperature, self.max_tokens, seed, prompt]).encode()).hexdigest()
        if self.cache and (self.cache / f"{key}.json").exists():
            return json.loads((self.cache / f"{key}.json").read_text())["text"]
        for attempt in range(6):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature, max_tokens=self.max_tokens, seed=seed)
                text = r.choices[0].message.content or ""
                break
            except Exception:  # rate limits and transient errors
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        if self.cache:
            (self.cache / f"{key}.json").write_text(json.dumps({"prompt": prompt, "text": text}))
        return text

    def generate(self, question: str, passages: Sequence[str], mode: str = "cot", seed: int = 42) -> Dict:
        if mode == "cot":
            prompt = COT_PROMPT.format(passages=format_passages(passages), question=question)
        elif mode == "direct":
            prompt = DIRECT_PROMPT.format(passages=format_passages(passages), question=question)
        elif mode == "constrained":
            prompt = CONSTRAINED_PROMPT.format(passages=format_passages(passages), question=question)
        elif mode == "closed_book":
            prompt = CLOSED_BOOK_PROMPT.format(question=question)
        else:
            raise ValueError(mode)
        out = parse_output(self._complete(prompt, seed))
        if mode == "constrained":
            out["answer"] = "" if out["abstained"] else out["raw"].strip()
        return out
