"""Does NAB's published detector ranking survive a different metric (event-F1, all 58 series)?
Reads only files in a NAB clone. Event-wise F1, each detector at its own published 'standard' threshold."""
import bisect, csv, json, sys, itertools
from datetime import datetime
import numpy as np
from scipy.stats import kendalltau

NAB = sys.argv[1]
def ts(s): return datetime.strptime(s.split(".")[0], "%Y-%m-%d %H:%M:%S")
thr = json.load(open(f"{NAB}/config/thresholds.json"))
raw = {k: json.load(open(f"{NAB}/labels/raw/{k}_labels_v1.0.json")) for k in ("AL", "CB", "SA")}
comb = json.load(open(f"{NAB}/labels/combined_windows.json"))
series = sorted(comb)
dets = sorted(thr)

def windows_from_points(times, pts):
    """NAB-style: window width = 10% of series length split over the labels, centred on each label."""
    n = len(times); k = len(pts)
    if k == 0: return []
    w = int(0.10 * n / k); out = []
    for p in pts:
        i = bisect.bisect_left(times, ts(p)); i = min(i, n - 1)
        out.append((max(0, i - w // 2), min(n - 1, i + w // 2)))
    return out

def windows_from_pairs(times, pairs):
    n = len(times); out = []
    for a, b in pairs:
        out.append((min(bisect.bisect_left(times, ts(a)), n - 1), min(bisect.bisect_right(times, ts(b)) - 1, n - 1)))
    return out

def runs(mask):
    out = []; start = None
    for i, m in enumerate(mask):
        if m and start is None: start = i
        if not m and start is not None: out.append((start, i - 1)); start = None
    if start is not None: out.append((start, len(mask) - 1))
    return out

cache = {}
def load(det, s):
    cat, fn = s.split("/")
    rows = list(csv.DictReader(open(f"{NAB}/results/{det}/{cat}/{det}_{fn}")))
    t = [ts(r["timestamp"]) for r in rows]; sc = np.array([float(r["anomaly_score"]) for r in rows])
    return t, sc

truths = {"combined": {}}
times_of = {}
for s in series:
    t, _ = load("numenta", s); times_of[s] = t
    pass
    truths["combined"][s] = windows_from_pairs(t, comb[s])

def f1_for(det, truth):
    th = thr[det]["standard"]["threshold"]
    found = nt = good = npred = 0
    for s in series:
        t, sc = load(det, s)
        pr = runs(sc >= th); win = truths[truth][s]
        nt += len(win); npred += len(pr)
        found += sum(any(p[0] <= w[1] and w[0] <= p[1] for p in pr) for w in win)
        good += sum(any(p[0] <= w[1] and w[0] <= p[1] for w in win) for p in pr)
    r = found / nt if nt else 0; p = good / npred if npred else 0
    return 2 * p * r / (p + r) if p + r else 0.0

res = {d: {k: f1_for(d, k) for k in truths} for d in dets}

pub = json.load(open(f"{NAB}/results/final_results.json"))
D = [d for d in dets if d in pub]
print("series used:", len(series), "| detectors compared:", len(D), "(bayesChangePt has no published score)")
tau, p = kendalltau([res[d]["combined"] for d in D], [pub[d]["standard"] for d in D])
print("Kendall tau, event-F1 vs NAB published standard score: %.2f (p=%.3f)" % (tau, p))
keep = [d for d in D if d not in ("null", "random", "expose")]
print("same, excluding null/random/expose: %.2f" % kendalltau([res[d]["combined"] for d in keep], [pub[d]["standard"] for d in keep])[0])
for d in sorted(D, key=lambda d: -pub[d]["standard"]):
    print("%-18s NAB %5.1f  | event-F1 %.3f" % (d, pub[d]["standard"], res[d]["combined"]))
