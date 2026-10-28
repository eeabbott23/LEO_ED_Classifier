# ==========================================================================
# Validation-sample description and rule-level error audit (read-only).
# Run on the secure platform from the repo root, after the locked evaluation
# run. Scores nothing; writes only aggregate CSVs to outputs/.
#   Part A  decision rule recorded for each held-out disagreement, by
#           direction (adjudicator code -> classifier code), and the
#           adjudicated codes of charts where the R3 presence branch fired
#   Part B  characteristics of the development set, evaluation set, full
#           validation sample, and the flagged parent cohort
#   Part C  unique patients and development/evaluation patient overlap
# Cells < 11 print as "<11" unless SUPPRESS_SMALL=0 is set in the environment.
# Share only the printed aggregates and the two CSVs, never row-level data.
# ==========================================================================
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import glob, json, re
from pathlib import Path
import pandas as pd

SUPPRESS_SMALL = os.environ.get("SUPPRESS_SMALL", "1") != "0"
DATA_DIR = Path(os.environ.get("DATA_DIR", "."))
OUT = Path(os.environ.get("OUTPUT_DIR", "./outputs")); OUT.mkdir(exist_ok=True)
GOLD_FILE = os.environ.get("GOLD_STANDARD_FILE", "police_random_10pct_updated_07_08_26.csv")
GOLD_COL = os.environ.get("GOLD_LABEL_COL", "LEO-Transport?")
# Full flagged-cohort file (one row per encounter: demographics, site, year).
# Set FLAT_FILE to its path if the search below does not find it.
FLAT_FILE = os.environ.get("FLAT_FILE", "")
if not FLAT_FILE:
    _hits = sorted({h for pat in ("fullcohortpolice*", "**/fullcohortpolice*",
                                  "../fullcohortpolice*", "../**/fullcohortpolice*")
                    for h in glob.glob(str(DATA_DIR / pat), recursive=True)
                    if not h.endswith((".ipynb", ".py"))})
    if not _hits:
        _any = sorted(glob.glob(str(DATA_DIR / "**/*cohort*"), recursive=True)
                      + glob.glob(str(DATA_DIR / "../**/*cohort*"), recursive=True))
        raise FileNotFoundError(
            "full-cohort flat file not found. Set FLAT_FILE = '<path>' and rerun.\n"
            "files with 'cohort' in the name near here:\n  " + "\n  ".join(_any[:30]))
    FLAT_FILE = _hits[0]
    print("flat file candidates:", _hits, "-> using", FLAT_FILE)
KEY = "EncounterKey"
NAME = {0: "0_no_police", 1: "1_police", 2: "2_co_transport"}
PRED_CODE = {"no_police": 0, "only_police": 1, "police_and_ems": 2}


def n(x):
    x = int(x)
    return "<11" if (SUPPRESS_SMALL and 0 < x < 11) else x


def load(p):
    p = Path(p)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p, dtype=str)
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(p, dtype=str, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("all", b"", 0, 1, str(p))


def normkey(df):
    if KEY in df.columns:
        return df
    for c in df.columns:
        cl = str(c).lower().replace("_", "").replace(" ", "")
        if cl in ("encounterkey", "encounterid", "csn"):
            return df.rename(columns={c: KEY})
    raise KeyError(f"no EncounterKey-like column in {list(df.columns)[:30]}")


def jsonl(fp):
    return normkey(pd.DataFrame(json.loads(l) for l in open(fp) if l.strip()))


def intkey(df):
    df[KEY] = pd.to_numeric(df[KEY], errors="coerce").astype("Int64")
    return df.dropna(subset=[KEY])


def find(df, *pats):
    """First column whose lowercased name matches any regex in pats."""
    for p in pats:
        for c in df.columns:
            if re.search(p, str(c).lower()):
                return c
    return None


# ---- load ----------------------------------------------------------------
gold = intkey(normkey(load(DATA_DIR / GOLD_FILE)))
gold["gold"] = pd.to_numeric(gold[GOLD_COL], errors="coerce").astype("Int64")
dc = intkey(normkey(load(DATA_DIR / "kappa_overlap_list.csv")))
dc_keys = set(dc[KEY].astype(int))
ev = intkey(jsonl(sorted(glob.glob(str(OUT / "predictions_EVAL_v1.0_*.jsonl")))[-1]))
eval_keys = set(ev[KEY].astype(int))
dev_keys = None
for fp in sorted(glob.glob(str(OUT / "predictions_v1.0_*.jsonl"))):
    k = set(intkey(jsonl(fp))[KEY].astype(int))
    if len(k) == 50 and not (k & dc_keys) and not (k & eval_keys):
        dev_keys = k
assert dev_keys, "could not identify the definitive 50-chart dev run"
flat = intkey(normkey(load(FLAT_FILE)))
print(f"gold rows={len(gold)}  eval={len(eval_keys)}  dev={len(dev_keys)}  "
      f"flat={FLAT_FILE} rows={len(flat)} unique enc={flat[KEY].nunique()}")
print("GOLD COLUMNS:", list(gold.columns))
print("FLAT COLUMNS:", list(flat.columns))

# ==========================================================================
# PART A — definition-discrepancy audit on the 259 held-out charts
# ==========================================================================
m = ev.merge(gold[[KEY, "gold"]], on=KEY, how="inner")
m["pred"] = m["transport_mode"].map(PRED_CODE).astype("Int64")
m["err"] = m["pred"] != m["gold"]
m["pair"] = m["gold"].map(NAME) + " -> " + m["pred"].map(NAME)
print("\n" + "=" * 72)
print(f"PART A  held-out n={len(m)}  disagreements={int(m.err.sum())}")
print("=" * 72)
print("\n[A1] rule_fired x adjudicator code (all held-out charts)")
print(pd.crosstab(m["rule_fired"], m["gold"].map(NAME), margins=True).to_string())
print("\n[A2] rule_fired x error type (disagreements only)")
print(pd.crosstab(m.loc[m.err, "rule_fired"], m.loc[m.err, "pair"], margins=True).to_string())

pres = m[m["rule_fired"] == "police_present_at_arrival_with_ems"]
print(f"\n[A3] presence branch (R3 'police physically present ... alongside EMS') "
      f"fired on {len(pres)} charts; adjudicator code:")
print(pres["gold"].map(NAME).value_counts().to_string())

summary_a = pd.crosstab(m.loc[m.err, "rule_fired"], m.loc[m.err, "pair"])
summary_a.to_csv(OUT / "rule_by_disagreement.csv")

# ==========================================================================
# PART B — sample characteristics
# ==========================================================================
print("\n" + "=" * 72)
print("PART B  validation-sample characteristics")
print("=" * 72)
analyzable = gold[gold[KEY].isin(dev_keys | eval_keys)][[KEY, "gold"]]
assert len(analyzable) == 309, f"expected 309 analyzable, got {len(analyzable)}"
d = analyzable.merge(flat.drop_duplicates(KEY), on=KEY, how="left")
print(f"analyzable charts matched to flat file: {d[find(d, r'site') or KEY].notna().sum()}/309")
d["Set"] = d[KEY].map(lambda k: "Development" if k in dev_keys else "Evaluation")

COL = {
    "age": find(flat, r"^age$", r"age_?at", r"ageyears", r"\bage\b"),
    "sex": find(flat, r"^sex$", r"gender", r"sex"),
    "race": find(flat, r"^race", r"race", r"sparcsrollup"),
    "ethnicity": find(flat, r"ethnic"),
    "insurance": None,  # payer class unspecified for most encounters; not reported
    "language": find(flat, r"language"),
    "site": find(flat, r"ed_?site", r"site", r"department"),
    "year": find(flat, r"visit_?year", r"arrival_?year", r"year"),
    "acuity": find(flat, r"acuity", r"esi"),
}
print("detected columns:", COL, "\n  -> if any is wrong/None, set COL[...] by hand and rerun")

if COL["year"] is None:
    dt = find(gold, r"arrival_?datetime", r"arrival")
    d["_year"] = pd.to_datetime(d[KEY].map(gold.set_index(KEY)[dt]), errors="coerce").dt.year
    COL["year"] = "_year"
for frame in (d, flat):
    y = pd.to_numeric(frame[COL["year"]], errors="coerce")
    frame["Period"] = pd.cut(y, [2011, 2015, 2019, 2024],
                             labels=["2012-2015", "2016-2019", "2020-2024"])

groups = {"Development (n=50)": d[d.Set == "Development"],
          "Evaluation (n=259)": d[d.Set == "Evaluation"],
          "Validation sample (n=309)": d,
          f"Full flagged cohort (n={flat[KEY].nunique()})": flat.drop_duplicates(KEY)}
rows = []
if COL["age"]:
    r = {"Characteristic": "Age, median (IQR), y"}
    for g, df in groups.items():
        a = pd.to_numeric(df[COL["age"]], errors="coerce")
        r[g] = f"{a.median():.0f} ({a.quantile(.25):.0f}-{a.quantile(.75):.0f})"
    rows.append(r)
for lab, c in [("Sex", COL["sex"]), ("Race", COL["race"]), ("Ethnicity", COL["ethnicity"]),
               ("Insurance", COL["insurance"]), ("Preferred language", COL["language"]),
               ("ED site", COL["site"]), ("Acuity (ESI)", COL["acuity"]), ("Encounter period", "Period")]:
    if not c:
        continue
    rows.append({"Characteristic": lab})
    levels = groups["Validation sample (n=309)"][c].astype(object).fillna("Missing").astype(str).value_counts().index
    for lv in levels:
        r = {"Characteristic": f"  {lv}"}
        for g, df in groups.items():
            s = df[c].astype(object).fillna("Missing").astype(str)
            k = int((s == lv).sum())
            r[g] = f"{n(k)} ({100 * k / len(df):.1f}%)" if not (SUPPRESS_SMALL and 0 < k < 11) else "<11"
        rows.append(r)
tab = pd.DataFrame(rows).fillna("")
pd.set_option("display.width", 250, "display.max_colwidth", 40)
print(tab.to_string(index=False))
tab.to_csv(OUT / "validation_sample_characteristics.csv", index=False)
# year range per set (no counts)
for g, df in list(groups.items())[:3]:
    y = pd.to_numeric(df[COL["year"]], errors="coerce")
    print(f"  {g}: arrival years {int(y.min())}-{int(y.max())}")

# ==========================================================================
# PART C — unique patients and dev/eval patient overlap
# ==========================================================================
print("\n" + "=" * 72)
print("PART C  unique patients")
print("=" * 72)
mrn_col = find(gold, r"primarymrn", r"mrn", r"patientkey", r"patient_?id")
src = gold
if mrn_col is None:
    mrn_col, src = find(flat, r"primarymrn", r"mrn", r"patientkey", r"patient_?id"), flat
assert mrn_col, "no patient identifier found; set mrn_col by hand"
pid = src.drop_duplicates(KEY).set_index(KEY)[mrn_col]
d["pid"] = d[KEY].map(pid)
print(f"patient id column: {mrn_col}  (missing for {int(d.pid.isna().sum())} of 309)")
dev_p = set(d.loc[d.Set == "Development", "pid"].dropna())
ev_p = set(d.loc[d.Set == "Evaluation", "pid"].dropna())
both = dev_p & ev_p
per = d.groupby("pid").size()
print(f"unique patients: all 309 -> {d.pid.nunique()} | dev -> {len(dev_p)} | eval -> {len(ev_p)}")
print(f"patients with >1 encounter in the 309: {n((per > 1).sum())} "
      f"(max encounters/patient {int(per.max())})")
print(f"patients appearing in BOTH dev and eval: {len(both)}")
print(f"  eval encounters belonging to those patients: "
      f"{int(d[(d.Set == 'Evaluation') & d.pid.isin(both)].shape[0])}")
if both:
    ov = m[m[KEY].isin(d.loc[(d.Set == "Evaluation") & d.pid.isin(both), KEY])]
    rest = m[~m[KEY].isin(ov[KEY])]
    for lab, s in (("overlap-patient eval charts", ov), ("eval excluding them", rest)):
        print(f"  {lab}: n={len(s)}  3-cat agreement {(~s.err).mean():.3f}  "
              f"binary agreement {((s.pred > 0) == (s.gold > 0)).mean():.3f}")
fp_all = flat.drop_duplicates(KEY)
fmrn = find(fp_all, r"primarymrn", r"mrn", r"patientkey", r"patient_?id")
if fmrn:
    print(f"full flagged cohort: {fp_all[KEY].nunique()} encounters, "
          f"{fp_all[fmrn].nunique()} unique patients")
print("\nDONE. Share only this printed output and the two CSVs in outputs/.")
