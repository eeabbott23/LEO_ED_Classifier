#!/usr/bin/env python3
"""Reproduce the manuscript's Tables 4-6 from a predictions JSONL + gold CSV.

  python score_predictions.py outputs/predictions_EVAL_v1.0_*.jsonl \\
      police_random_10pct_updated_07_08_26.csv --label-col "LEO-Transport?"

Read-only. Runs on Minerva (needs the gold file); published so every reported
number is regenerable from the archived artifacts.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import argparse, glob, json, sys
import numpy as np
import pandas as pd

MODE2CODE = {"no_police": 0, "only_police": 1, "police_and_ems": 2}
NAMES = {0: "No police", 1: "Police transport", 2: "Co-transport"}

def load_csv(p):
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try: return pd.read_csv(p, encoding=enc)
        except UnicodeDecodeError: continue
    raise ValueError(f"cannot decode {p}")

def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cats = sorted(set(a) | set(b)); ix = {c: i for i, c in enumerate(cats)}
    O = np.zeros((len(cats),)*2)
    for i, j in zip(a, b): O[ix[i], ix[j]] += 1
    po = np.trace(O)/len(a); pe = ((O.sum(1)/len(a))*(O.sum(0)/len(a))).sum()
    return (po-pe)/(1-pe) if (1-pe) > 1e-12 else float("nan")

def boot(a, b, fn, B=5000, seed=12345):
    a, b = np.asarray(a), np.asarray(b); rng = np.random.default_rng(seed); v = []
    for _ in range(B):
        i = rng.integers(0, len(a), len(a)); r = fn(a[i], b[i])
        if r == r: v.append(r)
    return np.percentile(v, 2.5), np.percentile(v, 97.5)

def cp_ci(k, n):
    try:
        from scipy.stats import beta
        return ((0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)),
                (1.0 if k == n else beta.ppf(0.975, k + 1, n - k)))
    except ImportError:
        z = 1.959963985; p = k/n
        d = 1 + z*z/n; c = p + z*z/(2*n)
        h = z*((p*(1-p)/n + z*z/(4*n*n))**0.5)
        return max(0.0, (c-h)/d), min(1.0, (c+h)/d)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions"); ap.add_argument("gold")
    ap.add_argument("--label-col", default="LEO-Transport?")
    args = ap.parse_args()
    pfile = sorted(glob.glob(args.predictions))[-1] if "*" in args.predictions else args.predictions

    preds = pd.DataFrame(json.loads(l) for l in open(pfile) if l.strip())
    if "EncounterKey" in preds.columns:
        preds = preds.rename(columns={"EncounterKey": "encounter_key"})
    preds["encounter_key"] = pd.to_numeric(preds["encounter_key"], errors="coerce").astype(int)
    gold = load_csv(args.gold)
    gold["EncounterKey"] = pd.to_numeric(gold["EncounterKey"], errors="coerce").astype("Int64")
    m = preds.merge(gold[["EncounterKey", args.label_col]].rename(
        columns={"EncounterKey": "encounter_key", args.label_col: "gold"}),
        on="encounter_key", how="inner")
    m["gold"] = pd.to_numeric(m["gold"], errors="coerce")
    m = m.dropna(subset=["gold"]); m["gold"] = m["gold"].astype(int)
    m["pred"] = m["transport_mode"].map(MODE2CODE)
    n = len(m)
    print(f"scored {n} charts ({os.path.basename(pfile)})\n")

    g, p = m["gold"].to_numpy(), m["pred"].to_numpy()
    gb, pb = (g >= 1).astype(int), (p >= 1).astype(int)

    print("=== TABLE 4: agreement ===")
    for tag, x, y in [("3-category", g, p), ("binary any-police", gb, pb)]:
        ag = int((x == y).sum()); lo, hi = cp_ci(ag, n)
        k = kappa(x, y); klo, khi = boot(x, y, kappa)
        print(f"  {tag:<18} {ag}/{n} = {100*ag/n:.1f}% (95% CI {100*lo:.1f}-{100*hi:.1f})"
              f"  κ = {k:.3f} (95% CI {klo:.3f}-{khi:.3f})")
    for c in (0, 1, 2):
        tp = int(((p == c) & (g == c)).sum())
        npred, ngold = int((p == c).sum()), int((g == c).sum())
        plo, phi = cp_ci(tp, max(npred, 1)); rlo, rhi = cp_ci(tp, max(ngold, 1))
        print(f"  {NAMES[c]:<18} PPV {tp}/{npred} = {tp/max(npred,1):.2f} ({plo:.2f}-{phi:.2f})"
              f" | recall {tp}/{ngold} = {tp/max(ngold,1):.2f} ({rlo:.2f}-{rhi:.2f})")
    flag_tp = int((g >= 1).sum())
    flo, fhi = cp_ci(flag_tp, n)
    print(f"  source-flag PPV (this file's charts): {flag_tp}/{n} = {100*flag_tp/n:.1f}%"
          f" ({100*flo:.1f}-{100*fhi:.1f})")

    print("\n=== TABLE 5: confusion (rows = adjudicator, cols = classifier) ===")
    print(pd.crosstab(m["gold"].map(NAMES), m["pred"].map(NAMES), margins=True).to_string())

    if "confidence" in m.columns:
        print("\n=== TABLE 6: agreement by self-rated confidence ===")
        for tier in ("high", "medium", "low"):
            s = m[m["confidence"] == tier]
            if not len(s): continue
            ag = int((s["gold"] == s["pred"]).sum())
            lo, hi = cp_ci(ag, len(s))
            print(f"  {tier:<7} {ag}/{len(s)} = {100*ag/len(s):.1f}% "
                  f"(95% CI {100*lo:.1f}-{100*hi:.1f})")

if __name__ == "__main__":
    sys.exit(main())
