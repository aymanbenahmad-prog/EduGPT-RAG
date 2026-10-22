from edugpt_rag.chunking import chunk_document


def doc(n_sent, words=50):
    return " ".join(" ".join(f"w{i}_{j}" for j in range(words - 1)) + f" end{i}." for i in range(n_sent))


def test_short_document_is_one_chunk():
    c = chunk_document("d", doc(5), 512, 128, use_nltk=False)
    assert len(c) == 1 and c[0]["n_tokens"] == 250


def test_window_and_overlap_bounds():
    c = chunk_document("d", doc(40), 512, 128, use_nltk=False)
    assert len(c) > 1
    for a, b in zip(c, c[1:]):
        assert a["n_tokens"] <= 512 and b["n_tokens"] <= 512
        shared = a["last_sentence"] - b["first_sentence"] + 1
        assert shared * 50 >= 128                      # Eq. 3: overlap >= s
        assert b["first_sentence"] > a["first_sentence"]  # always advances
    assert c[-1]["last_sentence"] == 39                # covers the whole document


def test_long_sentence_is_split():
    c = chunk_document("d", " ".join(["x"] * 1200) + ".", 512, 128, use_nltk=False)
    assert all(x["n_tokens"] <= 512 for x in c) and sum(x["n_tokens"] for x in c) >= 1200


def test_invalid_overlap():
    import pytest
    with pytest.raises(ValueError):
        chunk_document("d", "a b c.", 128, 128)
