"""Honest scoring for NASA SMAP/MSL telemetry anomaly detection.

Compares a detector's predicted segments with labels under three metrics, and against
budget-matched baselines that use no telemetry at all (random and position-prior).
Segments are inclusive [start, end] index pairs.
"""
from __future__ import annotations

import ast
import csv
from typing import Dict, List, Sequence, Tuple

import numpy as np

Seg = Tuple[int, int]


def parse_segs(text) -> List[Seg]:
    v = ast.literal_eval(text) if isinstance(text, str) else text
    return [(int(a), int(b)) for a, b in v]


def overlaps(a: Seg, b: Seg) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def to_mask(segs: Sequence[Seg], n: int) -> np.ndarray:
    m = np.zeros(n, bool)
    for a, b in segs:
        m[max(0, a): min(n - 1, b) + 1] = True
    return m


def event_counts(pred: Sequence[Seg], true: Sequence[Seg]) -> Dict[str, int]:
    """Segment-level counts. A true segment is found if any prediction overlaps it."""
    found = sum(any(overlaps(p, t) for p in pred) for t in true)
    good = sum(any(overlaps(p, t) for t in true) for p in pred)
    return {"true": len(true), "found": found, "pred": len(pred), "good_pred": good}


def point_adjust_mask(pred: np.ndarray, true_segs: Sequence[Seg]) -> np.ndarray:
    """The common 'point adjustment': one hit inside a true segment credits the whole segment."""
    out = pred.copy()
    for a, b in true_segs:
        if pred[a: b + 1].any():
            out[a: b + 1] = True
    return out


def f1(tp: float, fp: float, fn: float) -> float:
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 0.0


def score(channels: List[dict], pred_key: str = "pred") -> Dict[str, float]:
    """channels: dicts with 'n' (length), 'true' and pred_key (segment lists). Pooled over channels."""
    ev = {"true": 0, "found": 0, "pred": 0, "good_pred": 0}
    pw = np.zeros(3)   # tp, fp, fn point-wise
    pa = np.zeros(3)   # point-adjusted
    for c in channels:
        e = event_counts(c[pred_key], c["true"])
        for k in ev:
            ev[k] += e[k]
        t = to_mask(c["true"], c["n"]); p = to_mask(c[pred_key], c["n"])
        pw += [(p & t).sum(), (p & ~t).sum(), (~p & t).sum()]
        q = point_adjust_mask(p, c["true"])
        pa += [(q & t).sum(), (q & ~t).sum(), (~q & t).sum()]
    rec = ev["found"] / ev["true"] if ev["true"] else 0.0
    prec = ev["good_pred"] / ev["pred"] if ev["pred"] else 0.0
    return {"event_recall": rec, "event_precision": prec,
            "event_f1": 2 * rec * prec / (rec + prec) if rec + prec else 0.0,
            "pointwise_f1": f1(*pw), "point_adjusted_f1": f1(*pa)}


def load(labels_csv: str, results_csv: str) -> List[dict]:
    """Join the label file with the original telemanom results (predicted segments = tp + fp)."""
    lab = {r["chan_id"]: r for r in csv.DictReader(open(labels_csv))}
    out = []
    for r in csv.DictReader(open(results_csv)):
        L = lab[r["chan_id"]]
        pred = sorted(set(parse_segs(r["tp_sequences"]) + parse_segs(r["fp_sequences"])))
        out.append({"id": r["chan_id"], "n": int(L["num_values"]), "true": parse_segs(L["anomaly_sequences"]),
                    "pred": pred})
    return out


def baseline(channels: List[dict], mode: str, rng: np.random.Generator) -> List[dict]:
    """Same number and lengths of predicted segments per channel as the real detector.

    mode 'random'   : positions uniform over the series.
    mode 'position' : positions drawn from where anomalies sit in OTHER channels (leave-one-out).
    No telemetry is used.
    """
    centres = {c["id"]: [((a + b) / 2) / c["n"] for a, b in c["true"]] for c in channels}
    out = []
    for c in channels:
        pool = [x for k, v in centres.items() if k != c["id"] for x in v]
        segs = []
        for a, b in c["pred"]:
            ln = b - a + 1
            p = rng.uniform(0, 1) if mode == "random" else rng.choice(pool)
            s = int(round(p * c["n"] - ln / 2))
            s = min(max(s, 0), max(c["n"] - ln, 0))
            segs.append((s, s + ln - 1))
        out.append({**c, "base": segs})
    return out


def compare(channels: List[dict], reps: int = 300, seed: int = 0) -> Dict[str, dict]:
    rng = np.random.default_rng(seed)
    res = {"detector": score(channels, "pred")}
    for mode in ("random", "position"):
        runs = [score(baseline(channels, mode, rng), "base") for _ in range(reps)]
        res[mode] = {k: (float(np.mean([r[k] for r in runs])),
                         float(np.percentile([r[k] for r in runs], 5)),
                         float(np.percentile([r[k] for r in runs], 95))) for k in runs[0]}
    return res
