# ============ KEYWORD BASELINE — transparent, LLM-free ============
# Scores the same 259 held-out EVAL charts with two pre-specified keyword rule
# sets and compares against the adjudicated labels. Read-only, zero API cost.
#   B1 naive co-occurrence: police terms + EMS terms anywhere in the notes.
#   B2 arrival-anchored: same terms, but only inside arrival-language windows
#      ("BIB…", "brought in by…", "arrived via/with…", "presented via/with…").
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import glob, json, re
import numpy as np
import pandas as pd

NOTES_FILE = "full_cohort_notes.csv"  # local notes extract (never committed)
NOTE_TYPE_EXCLUDE = {"Discharge Instructions", "Attestation"}
GOLD_FILE, GOLD_COL = "police_random_10pct_updated_07_08_26.csv", "LEO-Transport?"

def load_csv(p):
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try: return pd.read_csv(p, encoding=enc)
        except UnicodeDecodeError: continue
    raise ValueError(f"cannot decode {p}")

# ---- eval keys + gold ----
eval_fp = sorted(glob.glob("outputs/predictions_EVAL_v1.0_*.jsonl"))[-1]
run1 = pd.DataFrame(json.loads(l) for l in open(eval_fp) if l.strip())
if "EncounterKey" in run1.columns: run1 = run1.rename(columns={"EncounterKey": "encounter_key"})
eval_keys = set(pd.to_numeric(run1["encounter_key"], errors="coerce").astype(int))
gold = load_csv(GOLD_FILE)
gold["EncounterKey"] = pd.to_numeric(gold["EncounterKey"], errors="coerce").astype("Int64")
gmap = {int(k): int(v) for k, v in
        zip(gold["EncounterKey"], pd.to_numeric(gold[GOLD_COL], errors="coerce"))
        if pd.notna(k) and pd.notna(v)}

# ---- notes text per encounter (same exclusions as the classifier) ----
ndf = load_csv(NOTES_FILE)
ndf = ndf[~ndf["NOTE_TYPE"].isin(NOTE_TYPE_EXCLUDE)]
ndf["NOTE_TEXT"] = ndf["NOTE_TEXT"].fillna("").astype(str)
text = {int(k): " ".join(g["NOTE_TEXT"]).lower()
        for k, g in ndf.groupby("EncounterKey") if int(k) in eval_keys}
print(f"eval charts with notes: {len(text)} / {len(eval_keys)}")

# ---- pre-specified term sets ----
POLICE = re.compile(r"\b(nypd|police|law enforcement|officers?|sgt|sergeant)\b")
EMS    = re.compile(r"\b(ems|emts?|paramedics?|ambulance|fdny|biba|bibems)\b")
ARRIVAL_WINDOW = re.compile(
    r"(?:\bbib\w*\b|brought in by|brought to (?:the )?(?:ed|er|hospital) by"
    r"|arriv\w+ (?:via|by|with|in)|presented (?:via|by|with)"
    r"|transported (?:via|by|with))[^.\n]{0,80}")

def b1(t):  # naive co-occurrence anywhere
    p, e = bool(POLICE.search(t)), bool(EMS.search(t))
    return 2 if (p and e) else (1 if p else 0)

def b2(t):  # arrival-anchored
    win = " ".join(mm.group(0) for mm in ARRIVAL_WINDOW.finditer(t))
    p, e = bool(POLICE.search(win)), bool(EMS.search(win))
    return 2 if (p and e) else (1 if p else 0)

rows = [{"encounter_key": k, "gold": gmap[k], "b1": b1(t), "b2": b2(t)}
        for k, t in text.items() if k in gmap]
df = pd.DataFrame(rows)
print(f"scored with gold labels: {len(df)}")

# ---- metrics ----
def kappa(x, y):
    x, y = np.asarray(x), np.asarray(y)
    cats = sorted(set(x) | set(y)); ix = {c: i for i, c in enumerate(cats)}
    O = np.zeros((len(cats),)*2)
    for i, j in zip(x, y): O[ix[i], ix[j]] += 1
    po = np.trace(O)/len(x); pe = ((O.sum(1)/len(x))*(O.sum(0)/len(x))).sum()
    return (po-pe)/(1-pe) if (1-pe) > 1e-12 else float("nan")
def bci(x, y, B=5000):
    x, y = np.asarray(x), np.asarray(y); rng = np.random.default_rng(7); v = []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x)); r = kappa(x[i], y[i])
        if r == r: v.append(r)
    return np.percentile(v, 2.5), np.percentile(v, 97.5)

NAMES = {0: "0 no_police", 1: "1 police", 2: "2 co-transport"}
for tag, col in [("B1 naive co-occurrence", "b1"), ("B2 arrival-anchored", "b2")]:
    g, p = df["gold"].to_numpy(), df[col].to_numpy()
    ag = float((g == p).mean()); k3 = kappa(g, p); lo, hi = bci(g, p)
    gb, pb = (g >= 1).astype(int), (p >= 1).astype(int)
    bag = float((gb == pb).mean()); kb = kappa(gb, pb); blo, bhi = bci(gb, pb)
    print(f"\n=== {tag} ===")
    print(f"  3-cat : {100*ag:.1f}%  κ={k3:.3f} [{lo:.3f},{hi:.3f}]")
    print(f"  binary: {100*bag:.1f}%  κ={kb:.3f} [{blo:.3f},{bhi:.3f}]")
    print(pd.crosstab(df['gold'].map(NAMES), df[col].map(NAMES),
                      rownames=['gold'], colnames=[tag], margins=True).to_string())

print("\nclassifier reference (same 259): 3-cat 78.0% κ=0.653 | binary 91.9% κ=0.734")
