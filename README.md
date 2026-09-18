# LEO_ED_Classifier

LLM classifier separating police transport, police–EMS co-transport, and
false-positive flags among LEO-flagged emergency department encounters, using
enum-constrained structured outputs against free-text ED notes. Companion code
repository for the methods manuscript (citation/DOI added on publication).
Design, operational definitions, and validation strategy: `docs/approach.md`.

## Locked artifacts (v1.0)

Every reported result was produced by the locked v1.0 pipeline. Verify any
artifact by re-hashing:

| artifact | value |
|---|---|
| system prompt (`classifier_lib.SYSTEM_PROMPT`) SHA-256 | `e20bb0a1b06ac15e295356ad788fd0a93ca3d99ee1011b160006a1dc205b18dd` |
| output schema SHA-256 | `abf88ec6a6ca22456f1602a87781bbe8856991e2fd6257dc6da9c896faeda6d7` |
| model deployment | `gpt-5-2025-08-07` (Azure OpenAI, institutional BAA) |
| API version | `2024-08-01-preview` |
| sampling | none passed — API defaults (GPT-5 does not accept `temperature`) |
| split seed | `20260524` (50 dev / 259 eval; all 62 double-coded charts in eval) |

The prompt, schema, model deployment, and split seed were locked together
before any evaluation-set chart was scored; a tried-and-reverted prompt
revision (v1.1) is documented in `docs/approach.md`.

## Repository contents

| file | purpose |
|---|---|
| `classifier_lib.py` | locked v1.0 source of truth: schema, verbatim system prompt, decision rules R1–R8 |
| `docs/approach.md` | design document: definitions, validation strategy, prompt-version history |
| `run_classifier_dev.ipynb` | dev-set runner (cost-capped, eval-locked) |
| `run_classifier_dev_STANDALONE.ipynb` | single-file variant with `classifier_lib.py` embedded (regenerate via `make_standalone.py`) |
| `transport_classifier.ipynb` | viewer: prints the schema, prompt, and note formatting exactly as used at runtime |
| `deploy_cell.py` | checkpointed, resumable full-cohort deployment cell |
| `stability_check_cell.py` | repeated-inference stability check (seeded 50-chart resample) |
| `keyword_baseline_cell.py` | pre-specified keyword baselines scored on the same charts |
| `score_predictions.py` | regenerates the manuscript's performance tables from any predictions JSONL |
| `make_split_manifest.py` | emits the public salted-hash dev/eval split manifest |
| `make_eval_metadata.py` | reconstructs the evaluation-run metadata JSON (documented as reconstructed) |
| `make_figure1.py` | regenerates the manuscript's pipeline/validation-design figure |
| `fixtures/` | synthetic charts + expected outputs (no PHI) — see below |

The salted-hash split manifest and the evaluation-run metadata JSON are
generated on the secure platform by the two `make_*` scripts above. Both are
run artifacts derived from the study cohort and are not published here; they
are held on-platform and available from the corresponding author on
reasonable request.

## Verify the pipeline without patient data

`fixtures/synthetic_notes.csv` contains six fully synthetic encounters, one per
decision-rule pathway (police-only, both R3 co-transport branches, EMS-only,
police-initiated-EMS, officer-as-patient). With Azure OpenAI credentials:

```bash
python run_fixtures.py     # ~$0.50; checks outputs against fixtures/expected_outputs.json
```

This exercises the identical prompt, schema, strict-mode structured outputs,
and note formatting used for every reported result.

## Reproduce the reported analyses (data access required)

Patient notes and encounter-level labels contain PHI and remain on the
institution's secure high-performance computing platform; they cannot be
shared. With approved access:

```bash
python score_predictions.py "outputs/predictions_EVAL_v1.0_*.jsonl" \
    <gold-standard>.csv --label-col "LEO-Transport?"
```

regenerates the manuscript's performance tables exactly. `make_split_manifest.py`
reproduces the development/evaluation split from the same seed, and its salted
hashes are verifiable against the raw identifiers with the salt held on-platform.

## Secure-platform quickstart

The repo is code-only. Before anything runs on the secure platform:

1. **Environment** (once):

   ```bash
   git clone https://github.com/eeabbott23/LEO_ED_Classifier.git
   cd LEO_ED_Classifier
   module load python
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Secrets** — `cp .env.example .env`, fill in the Azure OpenAI
   key/endpoint/deployment. Never commit `.env`.

3. **Data** (gitignored, never leaves the secure platform): the notes file
   (`NOTES_FILE`), the gold-standard master (`GOLD_STANDARD_FILE`), and the
   double-coded key list (`DOUBLE_CODED_FILE`).

PHI and row-level data stay on the secure platform. Headers, counts, hashes,
and aggregate statistics may be shared off-platform; values may not. The
split salt (`.split_salt`) stays on-platform.

## License

MIT — see `LICENSE`.
