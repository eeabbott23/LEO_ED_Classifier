# ============ STABILITY CHECK — repeated inference ============
# Re-scores a seeded random 50-chart subsample of the 259 held-out EVAL charts
# with the IDENTICAL locked v1.0 prompt/schema/deployment, then compares run 2
# against the archived EVAL run 1. Checkpointed & resumable (fixed filename).
# COST: ~$4 nominal (50 charts). Writes: predictions_STABILITY_v1.0.jsonl,
# metadata_STABILITY_v1.0.json, and prints the run1-vs-run2 comparison.
import glob, hashlib, json, os, time, random, copy
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

NOTES_FILE = "full_cohort_notes.csv"  # local notes extract (never committed)
NOTE_TYPE_EXCLUDE = {"Discharge Instructions", "Attestation"}
ENDPOINT_FALLBACK = ""  # set AZURE_OPENAI_ENDPOINT in .env
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
STAB_FILE = OUT / "predictions_STABILITY_v1.0.jsonl"   # fixed name -> resume works
META_FILE = OUT / "metadata_STABILITY_v1.0.json"
SUBSAMPLE_N, SUBSAMPLE_SEED = 50, 20260714
COST_CEILING, INP, OUTP, MAXTOK, MAXRETRY = 25.0, 10.0, 30.0, 4096, 5
LOCKED_PROMPT_SHA = "e20bb0a1b06ac15e295356ad788fd0a93ca3d99ee1011b160006a1dc205b18dd"

def load_csv(p):
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try: return pd.read_csv(p, encoding=enc)
        except UnicodeDecodeError: continue
    raise ValueError(f"cannot decode {p}")

# ---- locked prompt, verified by hash ----
import importlib, classifier_lib; importlib.reload(classifier_lib)
from classifier_lib import ChartClassification, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, PROMPT_VERSION, format_notes
_sha = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
assert PROMPT_VERSION == "v1.0" and _sha == LOCKED_PROMPT_SHA, \
    f"classifier_lib is NOT the locked v1.0! sha={_sha}"
print(f"prompt: {PROMPT_VERSION} (locked, sha {_sha[:12]}…)")

def inline_refs(s, d):
    if isinstance(s, dict):
        if "$ref" in s:
            t = copy.deepcopy(d[s["$ref"].split("/")[-1]]); t.update({k: v for k, v in s.items() if k != "$ref"}); return inline_refs(t, d)
        return {k: inline_refs(v, d) for k, v in s.items()}
    if isinstance(s, list): return [inline_refs(x, d) for x in s]
    return s
def enforce(s):
    if isinstance(s, dict):
        if s.get("type") == "object" and "properties" in s:
            s["additionalProperties"] = False; s["required"] = list(s["properties"])
        for v in s.values():
            if isinstance(v, (dict, list)): enforce(v)
    elif isinstance(s, list):
        for it in s:
            if isinstance(it, (dict, list)): enforce(it)
    return s
_raw = ChartClassification.model_json_schema(); _defs = _raw.pop("$defs", {})
SCHEMA = enforce(inline_refs(_raw, _defs)); assert "$ref" not in json.dumps(SCHEMA)
SCHEMA_SHA = hashlib.sha256(json.dumps(SCHEMA, sort_keys=True).encode()).hexdigest()

# ---- run-1 (archived EVAL) predictions + the seeded 50-chart subsample ----
eval_fp = sorted(glob.glob(str(OUT / "predictions_EVAL_v1.0_*.jsonl")))[-1]
run1 = pd.DataFrame(json.loads(l) for l in open(eval_fp) if l.strip())
if "EncounterKey" in run1.columns: run1 = run1.rename(columns={"EncounterKey": "encounter_key"})
run1["encounter_key"] = pd.to_numeric(run1["encounter_key"], errors="coerce").astype(int)
eval_keys = sorted(run1["encounter_key"].unique())
sub_keys = sorted(random.Random(SUBSAMPLE_SEED).sample(eval_keys, SUBSAMPLE_N))
print(f"run 1: {eval_fp} ({len(run1)} charts); subsample {SUBSAMPLE_N} (seed {SUBSAMPLE_SEED})")

# ---- notes (same stitch as the locked runner) ----
ndf = load_csv(NOTES_FILE)
ndf = ndf[~ndf["NOTE_TYPE"].isin(NOTE_TYPE_EXCLUDE)]
ndf = ndf.reset_index().rename(columns={"index": "_o"})
ndf["NOTE_TEXT"] = ndf["NOTE_TEXT"].fillna("").astype(str)
ndf = (ndf.sort_values(["EncounterKey", "NOTE_ID", "_o"])
         .groupby(["EncounterKey", "NOTE_ID"], as_index=False, sort=False)
         .agg(NOTE_TEXT=("NOTE_TEXT", "\n".join), NOTE_TYPE=("NOTE_TYPE", "first"), _o=("_o", "min"))
         .sort_values(["EncounterKey", "NOTE_ID"]).reset_index(drop=True))
enc_notes = {int(k): [{"text": r["NOTE_TEXT"], "role": r.get("NOTE_TYPE")} for _, r in g.iterrows() if r["NOTE_TEXT"].strip()]
             for k, g in ndf.groupby("EncounterKey", sort=False)}
missing = [k for k in sub_keys if k not in enc_notes]
assert not missing, f"subsample keys without notes: {missing}"

# ---- resume ----
done = set()
if STAB_FILE.exists():
    with open(STAB_FILE) as f:
        for line in f:
            try: done.add(int(json.loads(line)["encounter_key"]))
            except Exception: pass
target = [k for k in sub_keys if k not in done]
print(f"already scored this run: {len(done)} | remaining: {len(target)}")

# ---- Azure client (reuses an existing one in the kernel) ----
try:
    client; DEPLOYMENT
    print("reusing existing Azure client")
except NameError:
    from openai import AzureOpenAI
    from getpass import getpass
    ep = os.environ.get("AZURE_OPENAI_ENDPOINT") or ENDPOINT_FALLBACK
    if not ep.lower().startswith("http"): ep = "https://" + ep
    key = os.environ.get("AZURE_OPENAI_API_KEY") or getpass("API key: ")
    client = AzureOpenAI(api_key=key, api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"), azure_endpoint=ep)
    DEPLOYMENT = os.environ.get("AZURE_OPENAI_COMPLETION_DEPLOYMENT", "gpt-5-2025-08-07")

from openai import RateLimitError, APIConnectionError, APITimeoutError
def classify(notes):
    um = USER_PROMPT_TEMPLATE.format(formatted_notes=format_notes(notes))
    for a in range(MAXRETRY):
        try:
            r = client.chat.completions.create(model=DEPLOYMENT,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": um}],
                response_format={"type": "json_schema", "json_schema": {"name": "ChartClassification", "strict": True, "schema": SCHEMA}},
                max_completion_tokens=MAXTOK)
            return ChartClassification.model_validate_json(r.choices[0].message.content), r.usage.prompt_tokens, r.usage.completion_tokens
        except (RateLimitError, APIConnectionError, APITimeoutError):
            time.sleep((2 ** a) + random.random())
    raise RuntimeError("retries exhausted")

run_started = datetime.now(timezone.utc).isoformat()
errs = []; cost = 0.0; n_ok = 0; t0 = time.time()
with open(STAB_FILE, "a") as fout:
    for i, k in enumerate(target, 1):
        est = (len(SYSTEM_PROMPT) // 4 + len(format_notes(enc_notes[k])) // 4) * INP / 1e6 + MAXTOK * OUTP / 1e6
        if cost + est > COST_CEILING:
            print(f"ABORT: nominal cost guard at {i}/{len(target)}"); break
        try:
            p, a, b = classify(enc_notes[k]); cost += a * INP / 1e6 + b * OUTP / 1e6; n_ok += 1
            fout.write(json.dumps({"encounter_key": int(k), "transport_mode": p.transport_mode.value,
                                   "rule_fired": p.rule_fired.value, "evidence_quote": p.evidence_quote,
                                   "confidence": p.confidence, "custody_context": p.custody_context.value,
                                   "custody_evidence_quote": p.custody_evidence_quote,
                                   "prompt_tokens": a, "completion_tokens": b}) + "\n")
            fout.flush()
        except Exception as e:
            errs.append({"encounter_key": int(k), "error": repr(e)})
            print(f"[{i}] ERROR {k}: {e}")
        if i % 10 == 0 or i == len(target):
            print(f"[{i}/{len(target)}] ok={n_ok} err={len(errs)} ${cost:.2f} nominal")

with open(META_FILE, "w") as f:
    json.dump({"purpose": "repeated-inference stability check",
               "prompt_version": PROMPT_VERSION,
               "system_prompt_sha256": _sha, "schema_sha256": SCHEMA_SHA,
               "deployment": DEPLOYMENT,
               "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
               "run1_file": os.path.basename(eval_fp),
               "subsample_n": SUBSAMPLE_N, "subsample_seed": SUBSAMPLE_SEED,
               "run_started_at": run_started,
               "run_ended_at": datetime.now(timezone.utc).isoformat(),
               "n_scored_this_pass": n_ok, "n_errors": len(errs)}, f, indent=2)
print(f"metadata -> {META_FILE}")

# ================= COMPARISON: run 1 vs run 2 =================
run2 = pd.DataFrame(json.loads(l) for l in open(STAB_FILE) if l.strip())
run2["encounter_key"] = pd.to_numeric(run2["encounter_key"], errors="coerce").astype(int)
run2 = run2.drop_duplicates("encounter_key", keep="last")
cmp = run1.merge(run2, on="encounter_key", suffixes=("_1", "_2"))
n = len(cmp)
print(f"\n{'='*66}\nSTABILITY: run 1 vs run 2 on {n} charts\n{'='*66}")

def kappa(x, y):
    x, y = np.asarray(x), np.asarray(y)
    cats = sorted(set(x) | set(y)); ix = {c: i for i, c in enumerate(cats)}
    O = np.zeros((len(cats),)*2)
    for i2, j2 in zip(x, y): O[ix[i2], ix[j2]] += 1
    po = np.trace(O)/len(x); pe = ((O.sum(1)/len(x))*(O.sum(0)/len(x))).sum()
    return (po-pe)/(1-pe) if (1-pe) > 1e-12 else float("nan")
def wilson(k, nn):
    z = 1.959963985; p2 = k/nn
    d = 1 + z*z/nn; c = p2 + z*z/(2*nn)
    h = z*((p2*(1-p2)/nn + z*z/(4*nn*nn))**0.5)
    return max(0.0, (c-h)/d), min(1.0, (c+h)/d)

MODE2CODE = {"no_police": 0, "only_police": 1, "police_and_ems": 2}
c1 = cmp["transport_mode_1"].map(MODE2CODE); c2 = cmp["transport_mode_2"].map(MODE2CODE)
same = int((c1 == c2).sum()); lo, hi = wilson(same, n)
b1, b2 = (c1 >= 1).astype(int), (c2 >= 1).astype(int)
bsame = int((b1 == b2).sum()); blo, bhi = wilson(bsame, n)
print(f"transport_mode identical : {same}/{n} = {100*same/n:.1f}%  (95% CI {100*lo:.1f}-{100*hi:.1f})  κ={kappa(c1,c2):.3f}")
print(f"binary identical         : {bsame}/{n} = {100*bsame/n:.1f}%  (95% CI {100*blo:.1f}-{100*bhi:.1f})  κ={kappa(b1,b2):.3f}")
print(f"rule_fired identical     : {int((cmp['rule_fired_1']==cmp['rule_fired_2']).sum())}/{n}")
print(f"confidence identical     : {int((cmp['confidence_1']==cmp['confidence_2']).sum())}/{n}")
flips = cmp[c1 != c2]
if len(flips):
    print(f"\ncategory flips ({len(flips)}) — key, run1->run2, confidences, rules:")
    for _, r in flips.iterrows():
        print(f"  {r['encounter_key']}: {r['transport_mode_1']}->{r['transport_mode_2']} "
              f"(conf {r['confidence_1']}->{r['confidence_2']}; "
              f"rule {r['rule_fired_1']}->{r['rule_fired_2']})")
    onetwo = int(((c1[c1!=c2] >= 1) & (c2[c1!=c2] >= 1)).sum())
    print(f"  flips within the police-only<->co-transport boundary: {onetwo}/{len(flips)}")
else:
    print("\nno category flips — perfectly stable on this subsample")
print(f"\nDONE: {n_ok} scored this pass, {len(errs)} errors, ${cost:.2f} nominal")
