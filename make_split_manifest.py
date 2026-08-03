# ============ PUBLIC SPLIT MANIFEST (salted-hash identifiers) ============
# Builds the publishable dev/eval split manifest: every analyzable encounter's
# salted-SHA-256 identifier, split assignment, double-coded status, and
# pre-lock-exposure flag. The salt stays on Minerva (gitignored file) so the
# hashes are one-way for readers but verifiable by anyone with data access +
# the salt. Run on Minerva. Writes: outputs/split_manifest_public.csv (+ salt
# file .split_salt on first run — NEVER commit or share the salt).
import glob, hashlib, json, os, secrets
from pathlib import Path
import pandas as pd

SALT_FILE = Path(".split_salt")
if SALT_FILE.exists():
    SALT = SALT_FILE.read_text().strip()
    print("using existing salt")
else:
    SALT = secrets.token_hex(32)
    SALT_FILE.write_text(SALT)
    print("NEW salt generated -> .split_salt  (gitignored; keep on Minerva only)")

def hkey(k):
    return hashlib.sha256((SALT + str(int(k))).encode()).hexdigest()[:16]

def keys_of(jsonl):
    out = set()
    for l in open(jsonl):
        if l.strip():
            r = json.loads(l)
            out.add(int(r.get("encounter_key", r.get("EncounterKey"))))
    return out

# definitive dev run = v1.0 dev predictions disjoint from the double-coded 62
dc = None
for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
    try:
        dc = pd.read_csv("kappa_overlap_list.csv", encoding=enc); break
    except UnicodeDecodeError:
        continue
assert dc is not None, "could not read kappa_overlap_list.csv"
dc_keys = set(pd.to_numeric(dc["EncounterKey"], errors="coerce").dropna().astype(int))

eval_keys = keys_of(sorted(glob.glob("outputs/predictions_EVAL_v1.0_*.jsonl"))[-1])
dev_keys = None
for fp in sorted(glob.glob("outputs/predictions_v1.0_*.jsonl")):
    k = keys_of(fp)
    if not (k & dc_keys) and not (k & eval_keys):
        dev_keys = k; dev_src = os.path.basename(fp)
assert dev_keys and len(dev_keys) == 50, "could not identify the definitive 50-chart dev run"
print(f"dev 50 from {dev_src}; eval {len(eval_keys)}; total {len(dev_keys | eval_keys)}")

exposed = set()
if os.path.exists("outputs/split_exposure_manifest.csv"):
    man = pd.read_csv("outputs/split_exposure_manifest.csv")
    exposed = set(man.loc[man.exposed_before_lock_in_eval, "EncounterKey"].astype(int))

rows = [{"encounter_hash": hkey(k),
         "split": "dev" if k in dev_keys else "eval",
         "double_coded": k in dc_keys,
         "scored_prelock_in_provisional_draw": k in exposed}
        for k in sorted(dev_keys | eval_keys)]
pub = pd.DataFrame(rows).sort_values(["split", "encounter_hash"]).reset_index(drop=True)
out = Path("outputs/split_manifest_public.csv")
pub.to_csv(out, index=False)
print(f"written -> {out}  ({len(pub)} rows)")
print(pub.groupby(["split", "double_coded", "scored_prelock_in_provisional_draw"]).size().to_string())
print("\nManifest sha256:", hashlib.sha256(open(out, "rb").read()).hexdigest())
print("Copy split_manifest_public.csv into the repo; NEVER copy .split_salt off Minerva.")
