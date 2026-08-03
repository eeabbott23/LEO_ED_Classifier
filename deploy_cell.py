# ============ DEPLOYMENT RUN (locked v1.0, full cohort) — checkpointed & resumable ============
# Scores every flagged encounter with notes (~3,097). Appends each result to disk
# immediately; safe to interrupt and re-run — already-scored charts are skipped.
import glob, json, os, time, random, copy
from pathlib import Path
import pandas as pd

NOTES_FILE = "full_cohort_notes.csv"  # local notes extract (never committed)
NOTE_TYPE_EXCLUDE = {"Discharge Instructions", "Attestation"}
ENDPOINT_FALLBACK = ""  # set AZURE_OPENAI_ENDPOINT in .env
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
DEPLOY_FILE = OUT / "predictions_DEPLOY_v1.0.jsonl"   # fixed name -> resume works
COST_CEILING, INP, OUTP, MAXTOK, MAXRETRY = 500.0, 10.0, 30.0, 4096, 5

def load_csv(p):
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try: return pd.read_csv(p, encoding=enc)
        except UnicodeDecodeError: continue
    raise ValueError(f"cannot decode {p}")

import importlib, classifier_lib; importlib.reload(classifier_lib)
from classifier_lib import ChartClassification, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, PROMPT_VERSION, format_notes
assert PROMPT_VERSION == "v1.0" and "co-transport" not in SYSTEM_PROMPT, "classifier_lib is NOT locked v1.0!"
print("prompt:", PROMPT_VERSION, "(locked)")

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

ndf = load_csv(NOTES_FILE)
ndf = ndf[~ndf["NOTE_TYPE"].isin(NOTE_TYPE_EXCLUDE)]
ndf = ndf.reset_index().rename(columns={"index": "_o"})
ndf["NOTE_TEXT"] = ndf["NOTE_TEXT"].fillna("").astype(str)
ndf = (ndf.sort_values(["EncounterKey", "NOTE_ID", "_o"])
         .groupby(["EncounterKey", "NOTE_ID"], as_index=False, sort=False)
         .agg(NOTE_TEXT=("NOTE_TEXT", "\n".join), NOTE_TYPE=("NOTE_TYPE", "first"), _o=("_o", "min"))
         .sort_values(["EncounterKey", "NOTE_ID"]).reset_index(drop=True))
enc_notes = {k: [{"text": r["NOTE_TEXT"], "role": r.get("NOTE_TYPE")} for _, r in g.iterrows() if r["NOTE_TEXT"].strip()]
             for k, g in ndf.groupby("EncounterKey", sort=False)}
all_keys = sorted(enc_notes)
print(f"deployment universe: {len(all_keys)} encounters with notes")

# ---- resume: skip anything already written ----
done = set()
if DEPLOY_FILE.exists():
    with open(DEPLOY_FILE) as f:
        for line in f:
            try: done.add(json.loads(line)["encounter_key"])
            except Exception: pass
target = [k for k in all_keys if int(k) not in done and k not in done]
print(f"already scored: {len(done)} | remaining: {len(target)}")

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

errs = []; cost = 0.0; n_ok = 0; t0 = time.time()
with open(DEPLOY_FILE, "a") as fout:
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
            fout.flush()  # checkpoint every chart
        except Exception as e:
            errs.append({"encounter_key": int(k), "error": repr(e)})
            print(f"[{i}] ERROR {k}: {e}")
        if i % 25 == 0 or i == len(target):
            el = time.time() - t0
            rate = el / max(i, 1)
            rem = rate * (len(target) - i) / 60
            print(f"[{i}/{len(target)}] ok={n_ok} err={len(errs)} ${cost:.2f} | ~{rem:.0f} min left")

if errs:
    with open(OUT / "errors_DEPLOY_v1.0.json", "w") as f: json.dump(errs, f, indent=2)
print(f"\nDONE this pass: {n_ok} scored, {len(errs)} errors, ${cost:.2f} nominal")
print(f"total on disk: {sum(1 for _ in open(DEPLOY_FILE))} / {len(all_keys)} -> {DEPLOY_FILE}")
