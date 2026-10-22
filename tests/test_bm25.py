import math

from edugpt_rag.retrieval import PythonBM25


def test_bm25_eq4_single_term():
    b = PythonBM25(1.2, 0.75)
    b.index(["a", "b"], ["cat dog", "dog dog bird"])
    (top, s), = b.search("cat", 1)
    n, df, f, L, avg = 2, 1, 1, 2, 2.5
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    assert top == "a" and abs(s - idf * f * 2.2 / (f + 1.2 * (1 - 0.75 + 0.75 * L / avg))) < 1e-9
