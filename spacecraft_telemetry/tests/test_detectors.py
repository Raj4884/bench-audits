import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import detectors as d


def _series(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, 1500), rng.normal(0, 1, n)


def test_level_score_finds_a_level_shift():
    tr, te = _series(); te[1200:1260] += 8
    seg = d.topk_segments(d.level_score(tr, te), [60])[0]
    assert seg[0] <= 1260 and seg[1] >= 1200


def test_change_score_finds_a_spike():
    tr, te = _series(1); te[700] += 25
    seg = d.topk_segments(d.change_score(tr, te), [40])[0]
    assert seg[0] <= 700 <= seg[1]


def test_negative_control_pure_noise_has_no_stable_peak():
    hits = 0
    for s in range(40):
        tr, te = _series(s); te2 = te
        seg = d.topk_segments(d.level_score(tr, te2), [60])[0]
        hits += seg[0] <= 1000 <= seg[1]
    assert hits < 12          # a fixed target position is not found much more than by chance (~3%)


def test_topk_matches_budget_and_bounds():
    s = np.random.default_rng(0).normal(size=500)
    segs = d.topk_segments(s, [50, 30, 80])
    assert sorted(b - a + 1 for a, b in segs) == [30, 50, 80]
    assert all(0 <= a and b < 500 for a, b in segs)
