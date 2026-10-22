"""End-to-end run of Algorithm 1 with stub components (no GPU, no API)."""
from edugpt_rag.config import load_config
from edugpt_rag.generator import parse_output
from edugpt_rag.pipeline import EduGPTRAG
from edugpt_rag.retrieval import HybridRetriever, PythonBM25


class StubDense:
    def __init__(self, ids):
        self.ids = ids

    def search(self, q, k):
        return [(i, 1.0) for i in self.ids[:k]]


class StubReranker:
    def rerank(self, q, cand, id2text, top_k=5):
        return sorted(cand, key=lambda c: -sum(w in id2text[c].lower() for w in q.lower().split()))[:top_k]


class StubGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, question, passages, mode="cot", seed=42):
        self.calls.append(mode)
        if mode == "constrained":
            return parse_output("I cannot answer this question using the information available.")
        return parse_output("Most relevant passage: [1] Evidence: x Reasoning: y Answer: Denver Broncos Confidence: 0.9")


def nli_entails_if_word_present(pairs):
    return [0.95 if h.split()[0].lower() in p.lower() else 0.05 for p, h in pairs]


def make(tau):
    cfg = load_config(overrides={"verifier": {"tau": tau}})
    chunks = {f"c{i}": {"chunk_id": f"c{i}", "doc_id": f"d{i}", "text": t} for i, t in enumerate(
        ["The Denver Broncos won Super Bowl 50.", "Paris is in France.", "Water boils at 100 C."])}
    bm25 = PythonBM25()
    bm25.index(list(chunks), [c["text"] for c in chunks.values()])
    retr = HybridRetriever(StubDense(list(chunks)), bm25, cfg)
    gen = StubGenerator()
    return EduGPTRAG(cfg, retr, StubReranker(), gen, nli_entails_if_word_present, chunks), gen


def test_accepts_faithful_answer():
    sys_, gen = make(0.75)
    r = sys_.answer("q1", "Who won Super Bowl 50?")
    assert r["answer"] == "Denver Broncos" and not r["regenerated"] and gen.calls == ["cot"]
    assert r["context_doc_ids"][0] == "d0" and r["citation_correct"] is True and not r["hallucinated"]


def test_regenerates_once_and_abstains():
    sys_, gen = make(0.99)                       # threshold above the stub score forces regeneration
    r = sys_.answer("q2", "Who won Super Bowl 50?")
    assert r["regenerated"] and r["abstained"] and r["answer"] == "" and gen.calls == ["cot", "constrained"]


def test_parse_output():
    o = parse_output("Most relevant passage: [3] Evidence: e Reasoning: r Answer: 1945 Confidence: 0.80")
    assert o["answer"] == "1945" and o["cited_passage"] == 3 and o["confidence"] == 0.8 and not o["abstained"]
