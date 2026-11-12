"""Simple telemetry-reading baselines. Nothing here is tuned on the test labels.

Each scorer maps a test series x (1-D, the telemetry column) to a per-point anomaly score.
topk_segments() then turns a score into segments with the SAME count and lengths as the
detector being compared, so the comparison asks only: does it point at the right places?
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

Seg = Tuple[int, int]


def smooth(a: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return a
    k = np.ones(w) / w
    return np.convolve(a, k, mode="same")


def level_score(train: np.ndarray, test: np.ndarray, w: int = 30) -> np.ndarray:
    """Distance from the training mean in training standard deviations, lightly smoothed."""
    sd = train.std() or 1.0
    return smooth(np.abs(test - train.mean()) / sd, w)


def change_score(train: np.ndarray, test: np.ndarray, w: int = 30) -> np.ndarray:
    """Naive 'same as last step' forecast error, scaled by its size on the training data."""
    r_tr = np.abs(np.diff(train)); r_te = np.abs(np.diff(test, prepend=test[0]))
    sd = r_tr.std() or 1.0
    return smooth((r_te - r_tr.mean()) / sd, w)


def topk_segments(score: np.ndarray, lengths: Sequence[int]) -> List[Seg]:
    """Place one segment per requested length, centred on the highest remaining score."""
    s = score.astype("float64").copy()
    n = len(s); out: List[Seg] = []
    for ln in lengths:
        if not np.isfinite(s).any():
            break
        i = int(np.nanargmax(s))
        a = min(max(i - ln // 2, 0), max(n - ln, 0)); b = a + ln - 1
        out.append((a, b))
        s[max(a - ln // 2, 0): min(b + ln // 2, n - 1) + 1] = -np.inf
    return sorted(out)
