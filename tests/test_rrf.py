from edugpt_rag.rrf import rrf_fuse


def test_rrf_formula():
    fused = dict(rrf_fuse([["a", "b"], ["b", "c"]], kappa=60, top_n=10))
    assert abs(fused["b"] - (1 / 62 + 1 / 61)) < 1e-12
    assert abs(fused["a"] - 1 / 61) < 1e-12
    assert list(dict(rrf_fuse([["a", "b"], ["b", "c"]], 60, 10)))[0] == "b"


def test_top_n():
    assert len(rrf_fuse([list("abcdefghij"), list("klmnopqrst")], 60, 5)) == 5
