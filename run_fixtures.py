#!/usr/bin/env python3
"""End-to-end pipeline demo on synthetic fixtures (no PHI, ~$0.50 nominal).

Runs the locked v1.0 classifier over fixtures/synthetic_notes.csv and checks
each result against fixtures/expected_outputs.json. Requires Azure OpenAI
credentials (env vars or interactive prompt) but NO patient data — this is the
public verification path for readers without data access.

  python run_fixtures.py
"""
import copy, hashlib, json, os, random, time
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
LOCKED_PROMPT_SHA = "e20bb0a1b06ac15e295356ad788fd0a93ca3d99ee1011b160006a1dc205b18dd"
MAXTOK, MAXRETRY = 4096, 5

import classifier_lib
from classifier_lib import ChartClassification, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, PROMPT_VERSION, format_notes
_sha = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
assert PROMPT_VERSION == "v1.0" and _sha == LOCKED_PROMPT_SHA, f"not locked v1.0 (sha {_sha})"
print(f"prompt v1.0 verified (sha {_sha[:12]}…)")

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
SCHEMA = enforce(inline_refs(_raw, _defs))

ndf = pd.read_csv(HERE / "fixtures" / "synthetic_notes.csv")
expected = json.loads((HERE / "fixtures" / "expected_outputs.json").read_text())
enc_notes = {int(k): [{"text": r["NOTE_TEXT"], "role": r["NOTE_TYPE"]} for _, r in g.iterrows()]
             for k, g in ndf.groupby("EncounterKey", sort=True)}

from openai import AzureOpenAI, RateLimitError, APIConnectionError, APITimeoutError
from getpass import getpass
ep = os.environ.get("AZURE_OPENAI_ENDPOINT") or input("Azure endpoint: ")
if not ep.lower().startswith("http"): ep = "https://" + ep
client = AzureOpenAI(api_key=os.environ.get("AZURE_OPENAI_API_KEY") or getpass("API key: "),
                     api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
                     azure_endpoint=ep)
DEPLOYMENT = os.environ.get("AZURE_OPENAI_COMPLETION_DEPLOYMENT", "gpt-5-2025-08-07")

def classify(notes):
    um = USER_PROMPT_TEMPLATE.format(formatted_notes=format_notes(notes))
    for a in range(MAXRETRY):
        try:
            r = client.chat.completions.create(model=DEPLOYMENT,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": um}],
                response_format={"type": "json_schema", "json_schema": {"name": "ChartClassification", "strict": True, "schema": SCHEMA}},
                max_completion_tokens=MAXTOK)
            return ChartClassification.model_validate_json(r.choices[0].message.content)
        except (RateLimitError, APIConnectionError, APITimeoutError):
            time.sleep((2 ** a) + random.random())
    raise RuntimeError("retries exhausted")

n_pass = 0
for k in sorted(enc_notes):
    exp = expected[str(k)]
    p = classify(enc_notes[k])
    mode_ok = p.transport_mode.value == exp["transport_mode"]
    rule_ok = p.rule_fired.value in exp["acceptable_rules"]
    status = "PASS" if (mode_ok and rule_ok) else ("PASS*" if mode_ok else "FAIL")
    n_pass += mode_ok
    print(f"[{status}] {k} ({exp['tests']})")
    print(f"        got {p.transport_mode.value} / {p.rule_fired.value} / conf={p.confidence}"
          f"  | expected {exp['transport_mode']} / {exp['acceptable_rules']}")
    if not mode_ok:
        print(f"        evidence: {p.evidence_quote!r}")
print(f"\n{n_pass}/{len(enc_notes)} transport_mode expectations met "
      "(PASS* = correct category via an unlisted but plausible rule)")
