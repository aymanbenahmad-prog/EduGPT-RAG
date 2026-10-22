from edugpt_rag.metrics import abstention_metrics, compute_f1, normalize_answer, recall_at_k, squad_scores


def test_normalisation_matches_official():
    assert normalize_answer("The  Quick, brown fox!") == "quick brown fox"


def test_unanswerable_scoring():
    assert squad_scores([], "") == {"em": 1, "f1": 1.0}
    assert squad_scores([], "Paris") == {"em": 0, "f1": 0.0}


def test_max_over_golds():
    s = squad_scores(["Denver Broncos", "Broncos"], "the Broncos")
    assert s["em"] == 1 and s["f1"] == 1.0
    assert abs(compute_f1("w x y z", "w x") - 2 * (1.0 * 0.5) / 1.5) < 1e-9


def test_recall_and_abstention():
    assert recall_at_k(["x", "g", "y"], "g", 1) == 0 and recall_at_k(["x", "g"], "g", 2) == 1
    rows = [{"abstained": True, "unanswerable": True}, {"abstained": True, "unanswerable": False},
            {"abstained": False, "unanswerable": True}, {"abstained": False, "unanswerable": False}]
    m = abstention_metrics(rows)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["abstention_rate_unanswerable"] == 50.0
