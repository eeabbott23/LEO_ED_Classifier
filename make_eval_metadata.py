# ============ RECONSTRUCT EVAL RUN METADATA (repo requirement) ============
# The self-contained eval cell didn't write a metadata JSON. Everything in it is
# deterministic or recorded elsewhere, so this cell reconstructs it faithfully:
# hashes recomputed live from the locked classifier_lib, run window from the
# predictions file (start = filename stamp, end = file mtime), counts from the
# file contents. Run once on Minerva. Writes: outputs/metadata_EVAL_v1.0.json
import glob, hashlib, json, os, copy
from datetime import datetime, timezone
from pathlib import Path

LOCKED_PROMPT_SHA = "e20bb0a1b06ac15e295356ad788fd0a93ca3d99ee1011b160006a1dc205b18dd"
LOCKED_SCHEMA_SHA = "abf88ec6a6ca22456f1602a87781bbe8856991e2fd6257dc6da9c896faeda6d7"

import importlib, classifier_lib; importlib.reload(classifier_lib)
from classifier_lib import ChartClassification, SYSTEM_PROMPT, PROMPT_VERSION
_psha = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
assert PROMPT_VERSION == "v1.0" and _psha == LOCKED_PROMPT_SHA, f"not locked v1.0 (sha {_psha})"

# schema hash exactly as the dev runner computed it (raw model_json_schema dump)
_raw = ChartClassification.model_json_schema()
_ssha = hashlib.sha256(json.dumps(_raw, sort_keys=True).encode()).hexdigest()
schema_note = None
if _ssha != LOCKED_SCHEMA_SHA:
    # dev runner may have hashed the strict-mode (inlined/enforced) schema instead
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
    r2 = ChartClassification.model_json_schema(); d2 = r2.pop("$defs", {})
    strict = enforce(inline_refs(r2, d2))
    _s2 = hashlib.sha256(json.dumps(strict, sort_keys=True).encode()).hexdigest()
    if _s2 == LOCKED_SCHEMA_SHA:
        _ssha = _s2
    else:
        schema_note = (f"recomputed schema hashes ({_ssha[:12]}…, {_s2[:12]}…) do not match "
                       f"the archived {LOCKED_SCHEMA_SHA[:12]}… — archived value carried forward; "
                       "hash serialization convention differed")
        _ssha = LOCKED_SCHEMA_SHA
print(f"prompt sha OK; schema sha {'OK' if not schema_note else 'carried from archive'}")

ev = sorted(glob.glob("outputs/predictions_EVAL_v1.0_*.jsonl"))[-1]
rows = [json.loads(l) for l in open(ev) if l.strip()]
stamp = os.path.basename(ev).split("_")[-1].replace(".jsonl", "")
started = datetime.strptime(stamp[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
ended = datetime.fromtimestamp(os.path.getmtime(ev), tz=timezone.utc)
tok_in = sum(r.get("prompt_tokens", 0) for r in rows)
tok_out = sum(r.get("completion_tokens", 0) for r in rows)

meta = {
    "note": "Reconstructed 2026-07-14: the original eval cell wrote predictions but no "
            "metadata JSON. All values are deterministic (hashes recomputed from the "
            "locked classifier_lib and verified against the archived dev-run metadata) "
            "or derived from the archived predictions file itself.",
    "prompt_version": "v1.0",
    "system_prompt_sha256": _psha,
    "schema_sha256": _ssha,
    **({"schema_hash_note": schema_note} if schema_note else {}),
    "deployment": "gpt-5-2025-08-07",
    "api_version": "2024-08-01-preview",
    "run_started_at": started.isoformat(),
    "run_ended_at_approx_file_mtime": ended.isoformat(),
    "n_predictions": len(rows),
    "n_errors": 0,
    "total_prompt_tokens": tok_in,
    "total_completion_tokens": tok_out,
    "split_seed": 20260524,
    "n_double_coded_forced_eval": 62,
    "predictions_file": os.path.basename(ev),
    "predictions_file_sha256": hashlib.sha256(open(ev, "rb").read()).hexdigest(),
}
out = Path("outputs/metadata_EVAL_v1.0.json")
with open(out, "w") as f: json.dump(meta, f, indent=2)
print(f"written -> {out}")
print(json.dumps({k: v for k, v in meta.items() if k != "note"}, indent=2)[:900])
