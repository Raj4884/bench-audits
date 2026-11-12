"""Run in Colab:
    !git clone -q https://github.com/khundman/telemanom
    !unzip -oq telebench.zip
    %cd telebench
    %run run_colab.py ../telemanom
Needs the telemetry arrays (data/train/*.npy, data/test/*.npy) inside the telemanom folder.
The script tries the original download first and prints Kaggle instructions if that fails.
"""
import glob, os, sys, subprocess
import numpy as np
import telebench as tb
import detectors as dt

root = sys.argv[1]
if not os.path.isdir(f"{root}/data/test"):
    print("downloading telemetry ...")
    subprocess.run(f"cd {root} && wget -q https://s3-us-west-2.amazonaws.com/telemanom/data.zip -O data.zip "
                   f"&& unzip -oq data.zip && rm -f data.zip", shell=True)
if not os.path.isdir(f"{root}/data/test"):
    sys.exit("No data/test folder. Get it from Kaggle: pip install kaggle; kaggle datasets download -d "
             "patrickfleith/nasa-anomaly-detection-dataset-smap-msl; unzip it so that "
             f"{root}/data/train and {root}/data/test exist, then re-run.")

res = sorted(glob.glob(f"{root}/results/*.csv"))[0]
ch = tb.load(f"{root}/labeled_anomalies.csv", res)
bad = 0
for c in ch:
    tr = np.load(f"{root}/data/train/{c['id']}.npy")[:, 0]
    te = np.load(f"{root}/data/test/{c['id']}.npy")[:, 0]
    if len(te) != c["n"]:
        bad += 1
    lens = [b - a + 1 for a, b in c["pred"]]
    c["lvl"] = dt.topk_segments(dt.level_score(tr, te), lens)
    c["chg"] = dt.topk_segments(dt.change_score(tr, te), lens)
print(f"{len(ch)} channels; length mismatches between labels and data: {bad}")

rng = np.random.default_rng(0)
keys = ["event_recall", "event_precision", "event_f1", "pointwise_f1", "point_adjusted_f1"]
cmp_ = tb.compare(ch, reps=300)
rows = {"LSTM (published)": tb.score(ch, "pred"), "level z-score": tb.score(ch, "lvl"),
        "naive-forecast change": tb.score(ch, "chg"),
        "random (mean)": {k: cmp_["random"][k][0] for k in keys},
        "position prior (mean)": {k: cmp_["position"][k][0] for k in keys}}
print("%-24s" % "" + "".join("%18s" % k for k in keys))
for name, r in rows.items():
    print("%-24s" % name + "".join("%18.3f" % r[k] for k in keys))

# paired bootstrap over channels: LSTM event-F1 minus each telemetry baseline
for key, label in (("lvl", "level z-score"), ("chg", "naive-forecast change")):
    d = []
    for _ in range(1000):
        idx = rng.integers(0, len(ch), len(ch)); sub = [ch[i] for i in idx]
        d.append(tb.score(sub, "pred")["event_f1"] - tb.score(sub, key)["event_f1"])
    print(f"LSTM minus {label}, event-F1: mean {np.mean(d):+.3f}, 95% CI [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}]")
