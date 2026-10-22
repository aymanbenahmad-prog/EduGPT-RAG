import numpy as np

from edugpt_rag.verifier import citation_correct, faithfulness, is_hallucinated, softmax


def test_softmax_rows_sum_to_one():
    p = softmax(np.array([[1.0, 2.0, 3.0], [0, 0, 0]]))
    assert np.allclose(p.sum(axis=1), 1)


def test_eq15_and_hallucination_rule():
    S = np.array([[0.9, 0.2], [0.3, 0.45]])     # 2 sentences x 2 passages
    assert abs(faithfulness(S) - (0.9 + 0.45) / 2) < 1e-12
    assert is_hallucinated(S, 0.5)              # sentence 2 max 0.45 < 0.50
    assert not is_hallucinated(np.array([[0.6], [0.7]]), 0.5)


def test_citation():
    S = np.array([[0.8, 0.1], [0.9, 0.2]])
    assert citation_correct(S, 1, 0.75) is True
    assert citation_correct(S, 2, 0.75) is False
    assert citation_correct(S, None, 0.75) is None
