"""Usage: python run_pilot.py path/to/telemanom   (a clone of github.com/khundman/telemanom)"""
import sys, glob
import telebench as tb

root = sys.argv[1]
res = sorted(glob.glob(f"{root}/results/*.csv"))[0]
ch = tb.load(f"{root}/labeled_anomalies.csv", res)
r = tb.compare(ch, reps=500)
print("channels", len(ch), "| true segments", sum(len(c["true"]) for c in ch),
      "| predicted segments", sum(len(c["pred"]) for c in ch))
print("%-18s %7s | %-22s | %-22s" % ("metric", "LSTM", "random mean [5-95%]", "position-prior"))
for k in ["event_recall", "event_precision", "event_f1", "pointwise_f1", "point_adjusted_f1"]:
    d, a, b = r["detector"][k], r["random"][k], r["position"][k]
    print("%-18s %7.3f | %.3f [%.3f-%.3f]    | %.3f [%.3f-%.3f]" % (k, d, *a, *b))
