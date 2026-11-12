import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import telebench as tb


def ch(n, true, pred, i="x"):
    return {"id": i, "n": n, "true": true, "pred": pred}


def test_event_counts_hand_computed():
    e = tb.event_counts([(10, 20), (50, 60)], [(15, 30), (100, 110)])
    assert e == {"true": 2, "found": 1, "pred": 2, "good_pred": 1}


def test_event_f1_hand_computed():
    s = tb.score([ch(200, [(15, 30), (100, 110)], [(10, 20), (50, 60)])])
    assert s["event_recall"] == 0.5 and s["event_precision"] == 0.5 and s["event_f1"] == pytest.approx(0.5)


def test_point_adjust_inflates_a_one_point_hit():
    c = ch(1000, [(400, 600)], [(500, 500)])
    s = tb.score([c])
    assert s["pointwise_f1"] < 0.02
    assert s["point_adjusted_f1"] == pytest.approx(1.0)      # the known inflation
    assert s["event_f1"] == pytest.approx(1.0)


def test_no_prediction_scores_zero():
    s = tb.score([ch(100, [(10, 20)], [])])
    assert s["event_f1"] == 0 and s["pointwise_f1"] == 0 and s["point_adjusted_f1"] == 0


def test_position_prior_beats_random_when_anomalies_are_position_biased():
    rng = np.random.default_rng(1)
    chans = []
    for i in range(40):
        n = 1000; c = int(rng.normal(700, 20)); chans.append(ch(n, [(c - 30, c + 30)], [(0, 60)], str(i)))
    r = tb.compare(chans, reps=100)
    assert r["position"]["event_recall"][0] > 0.9
    assert r["random"]["event_recall"][0] < 0.2


def test_negative_control_uniform_anomalies_no_position_advantage():
    rng = np.random.default_rng(2)
    chans = []
    for i in range(60):
        n = 1000; c = int(rng.uniform(60, 940)); chans.append(ch(n, [(c - 30, c + 30)], [(0, 60)], str(i)))
    r = tb.compare(chans, reps=100)
    assert abs(r["position"]["event_recall"][0] - r["random"]["event_recall"][0]) < 0.06


def test_baseline_matches_budget():
    chans = [ch(1000, [(100, 200)], [(10, 40), (500, 560)], "a"), ch(1000, [(300, 400)], [(0, 9)], "b")]
    out = tb.baseline(chans, "random", np.random.default_rng(0))
    for c in out:
        assert [b - a for a, b in c["base"]] == [b - a for a, b in c["pred"]]
        assert all(0 <= a and b < c["n"] for a, b in c["base"])
