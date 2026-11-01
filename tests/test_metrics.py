import json

import pytest

from face_intel.metrics import evaluate, iou, latency_summary, load_ground_truth


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


def test_evaluate_counts_and_scores():
    truth = [(0, 0, 10, 10), (50, 50, 10, 10)]
    pred = [(0, 0, 10, 10), (100, 100, 10, 10)]       # 1 TP, 1 FP, 1 FN
    m = evaluate(pred, truth)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["precision"] == m["recall"] == m["f1"] == 0.5


def test_evaluate_no_double_matching():
    m = evaluate([(0, 0, 10, 10), (0, 0, 10, 10)], [(0, 0, 10, 10)])
    assert (m["tp"], m["fp"]) == (1, 1)


def test_ground_truth_loading(tmp_path):
    ok = tmp_path / "a.json"
    ok.write_text(json.dumps({"boxes": [[1, 2, 3, 4]]}))
    assert load_ground_truth(ok) == [(1.0, 2.0, 3.0, 4.0)]
    for bad in ("not json", json.dumps([[1, 2, 3]]), json.dumps([[1, 2, 0, 4]])):
        f = tmp_path / "b.json"
        f.write_text(bad)
        with pytest.raises(ValueError):
            load_ground_truth(f)
    with pytest.raises(ValueError):
        load_ground_truth(tmp_path / "missing.json")


def test_latency_summary():
    s = latency_summary([10, 20, 30])
    assert s["mean"] == 20 and s["median"] == 20 and s["fps"] == 50
    with pytest.raises(ValueError):
        latency_summary([])
