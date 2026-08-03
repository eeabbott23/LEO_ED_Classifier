# Pipeline Approach

This document specifies the design of the LLM-based classifier for police / EMS transport mode in emergency department (ED) encounters. It is the source of truth for operational definitions, schema, prompt, validation strategy, and cost discipline.

The single source of truth for the schema, prompt, decision-rule cascade, and `format_notes` helper is `classifier_lib.py`. Both `transport_classifier.ipynb` (viewer) and `run_classifier_dev.ipynb` (runner) import from it. **Any change to the rules below must be mirrored in `classifier_lib.py` and this document in the same commit** — otherwise PPV calculations across runs are uninterpretable.

## Study context

The pipeline supports a descriptive analysis of how patients flagged as law-enforcement-officer (LEO) transports arrive at the ED. The primary outcome is a three-category transport-mode variable applied to chart-abstracted encounters. The LLM classifier acts as a first-pass abstractor against which a human gold standard is compared.

The study is built to support three findings:

1. **How well the source-system LEO flag performs.** Every chart in the cohort carries the flag; the primary adjudicator's chart adjudication differentiates true police transports (codes 1 and 2) from false positives (code 0). The flag's PPV is simply the share of the 312 coded 1 or 2 — computable from the gold standard alone, no LLM required.
2. **Whether the LLM classifier is accurate enough to leverage for current research** — established by its PPV against the gold standard on the held-out eval set, read against the human ceiling (kappa = 0.761).
3. **Whether this approach and methods generalize to other, larger datasets** — supported by the methods-paper framing (iterative development, locked prompt, held-out evaluation, auditable rule/evidence outputs) and the 4,000-encounter deployment demonstrating scale.

Separately from this pipeline, the manuscript's descriptive results recode encounter ICD-10 codes using established classification systems (ICD-10 chapter classifications and AHRQ CCSR diagnostic categories), handled by the primary adjudicator outside this repository.

## Cohorts

Two cohorts, both LEO-flagged in the source system, with different roles:

**Validation cohort (N = 312).** Manually abstracted by the primary adjudicator, using numeric codes 0 = not police transport, 1 = police transport, 2 = police transport and EMS (mapped to the classifier categories via `GOLD_LABEL_MAP` in `classifier_lib.py`). A 62-encounter subset (an initial 32 plus an additional 30) was independently re-reviewed by the secondary adjudicator to assess inter-rater agreement on the human gold standard. **IRR result (definitive, 2026-07): Cohen's kappa = 0.761 (95% CI 0.606–0.896), overall percent agreement 85.5% (53/62) on the double-coded charts — substantial agreement; no further double-coding planned.** The primary adjudicator's master file was finalized after a small recode following the FDNY clarification. The 312 is where classifier quality is measured.

The original design specified N = 300 with a ~100-chart overlap; the realized numbers (312 / 62) supersede those throughout this document.

**Deployment cohort (N = 3,121; design estimate was "4,000").** The full LEO-flagged encounter set (`fullcohort_with_diagnosis_ccsr` — 4,366 rows because the file is diagnosis-level, one row per CCSR-coded diagnosis; 3,121 unique encounters). The note pull covers 3,097 of the 3,121. The locked classifier runs across all encounters with notes to produce descriptive estimates at scale. No human gold-standard labels exist for the deployment cohort; trust in the classifier output rests entirely on PPV established in the validation cohort.

**Realized data shapes (probe, 2026-07-08).** Notes: 36,197 rows / 23,538 notes (long notes are chunked across rows sharing a NOTE_ID; the runner stitches them, and chunks never cross encounters) across 3,097 encounters, median 10 note rows per encounter (max 162). Gold label distribution: 1 = 132, 2 = 122, 0 = 57, plus 1 unlabeled chart (dropped) and 2 charts without notes — realized validation universe ≈ 309. The note pull contains **all** ED chart note types (ED Psychiatric, ED Notes, ED Provider, Triage/Intake, nursing, consults, ...), not provider notes only; the runner excludes boilerplate types (Discharge Instructions, Attestation) and the prompt describes the actual input (v1.0 amendment #3). A `discordant_cases` file holds the 9 adjudicator disagreements from the 62 double-coded charts (leo1/leo2 columns) — the concordant 53 keys are still needed to complete the double-coded list.

Because both cohorts are restricted to LEO-flagged encounters, the design supports **positive predictive value (PPV)** of the LLM classifier against the human gold standard, but **not** sensitivity, specificity, or NPV. Detecting LEO transports the source-system flag missed would require abstracting non-LEO-flagged charts, which is outside current IRB approval. This is stated as a limitation in the manuscript alongside the related limitation that the source-system flag's own sensitivity is unknown.

## Outcome definition

Each encounter is classified into exactly one of three categories based on documented transportation and custody at arrival.

### `only_police`

Patient brought to the ED by law enforcement or in police custody, with no indication of EMS or ambulance transport. Qualifying language includes "BIB NYPD," "under arrest," and "in police custody" — without ambulance involvement.

### `police_and_ems`

**Locked definition (v1.0 — the prompt that scored the evaluation set).** Police *and* EMS were both involved in the arrival. This is satisfied when police actually transported alongside EMS, when police co-transported/accompanied the ambulance (escorted or rode with it, or the patient was in police custody during the EMS transport), **or when police were physically present with the patient at arrival / triage alongside EMS.** Police merely *initiating* the EMS call (e.g., "NYPD called EMS") without transporting or being present at arrival does **not** qualify, and police arriving at the ED *separately from the patient* (meeting them at the hospital later) does **not** qualify. This maps to decision rule R3, which fires either `co_transport_police_and_ems` (both transported) or `police_present_at_arrival_with_ems` (EMS transported, police present at arrival). Qualifying language includes "BIB EMS with NYPD" and "PD escorted ambulance."

> **Note on the reverted v1.1 narrowing.** A candidate v1.1 revision restricted this category to *transport-level* involvement only — excluding "police present at arrival" — but it performed worse on the dev set (3-way κ 0.519→0.433) and was **reverted**; see "Prompt versions" below. v1.0 (broad, present-at-arrival qualifies) is the locked prompt that produced every result in the manuscript. Earlier drafts of this section carried the narrow v1.1 wording in error; it has been corrected to match what was actually scored.

The adjudicators' operational definition maps to the categories as: 0 = no police involvement (`no_police`), 1 = NYPD transport (`only_police`), 2 = police and EMS co-transport (`police_and_ems`).

### `no_police`

No evidence of LEO involvement in transport or custody. Includes EMS-only and self / family / other transport.

Edge case: if the patient is themselves a police officer (e.g., an officer presenting for injury) but was not transported or detained by police, classify as `no_police`.

### Out of scope

The following situations are assumed not to occur and are **not** weighted toward a police category:

- Post-arrival custody changes (patient arrives independently and is later placed under arrest at bedside).
- Officer-as-patient transported by a fellow officer.

## Exploratory custody field (`custody_context`)

Alongside the validated transport-mode outcome, the classifier assigns an **exploratory** secondary field capturing *why* police were involved at arrival — a question of direct clinical and policy interest (how many patients are in police custody presenting for medical clearance versus transported for trauma or medical complaints):

- `criminal_custody` — arrest-related custody: under arrest, handcuffed, prisoner, medical clearance for booking/arraignment, precinct or corrections custody.
- `involuntary_psych_hold` — involuntary psychiatric removal or hold: EDP transport, 70/10, MHL §9.41 removal. Officers may use custody-like language for these removals ("taken into custody for evaluation"); they are classified here, **not** as criminal custody.
- `police_involved_no_custody` — police transported or were present, but no custody or hold is documented (e.g., assault victim driven in by officers).
- `not_applicable` — required when `transport_mode = no_police`.

Custody is orthogonal to transport mode: a handcuffed patient arriving by ambulance with officers is `police_and_ems` + `criminal_custody`.

**Validation status: none.** The human adjudicators coded transport mode only, so `custody_context` has no gold standard, no PPV, and is excluded from the validation analysis. It is reported **descriptively** on the deployment cohort, audited via `custody_evidence_quote` spot-checks, and explicitly labeled in the manuscript as not human-validated. The decision not to have the adjudicators re-review charts for custody was made deliberately (2026-07-08) to avoid re-review burden.

## Pipeline

```
chart notes (one row per note CHUNK; all ED note types except boilerplate)
       │
       │  drop Discharge Instructions + Attestation rows,
       │  stitch chunks sharing a NOTE_ID, group by EncounterKey,
       │  sort by NOTE_ID, format with labeled section headers
       │  (Note i of N, note type as role)
       ▼
per-encounter prompt (USER_PROMPT_TEMPLATE)
       │
       │  Azure OpenAI under institutional BAA
       │  + SYSTEM_PROMPT (codebook + cohort framing + decision rules + confidence rubric)
       │  + structured-output schema derived from ChartClassification
       ▼
ChartClassification (Pydantic)
   - transport_mode          (only_police | police_and_ems | no_police)
   - rule_fired              (controlled vocabulary; which decision rule produced the answer)
   - evidence_quote          (short verbatim grounding span)
   - confidence              (high | medium | low, per the rubric in the prompt; transport_mode only)
   - custody_context         (EXPLORATORY: criminal_custody | involuntary_psych_hold |
                              police_involved_no_custody | not_applicable)
   - custody_evidence_quote  (verbatim grounding span for custody_context)
       │
       │  joined back to the encounter table on EncounterKey
       ▼
classifier output (one row per encounter)  →  compare vs human adjudication
```

Each encounter is classified independently. No batching across encounters. No explicit chain-of-thought is requested; the schema's `evidence_quote` plus `rule_fired` together force the model to ground its decision in the note text and tag which decision rule applied.

## Prompt architecture (context engineering)

The prompt is deliberately **not** zero-shot. It embeds:

1. **Operational codebook** — the three-category definitions and out-of-scope items above are lifted verbatim into the system prompt, so the LLM and the human adjudicators are working from identical text.
2. **Cohort and setting framing** — NYC ED, EHR-LEO-flagged encounters, all ED chart note types (provider, psychiatric, triage/intake, nursing, consults; boilerplate excluded), and NYC-specific abbreviations (BIB, EDP, 70/10, perp, ESU, FDNY) so the model anchors in the expected language. FDNY is explicitly glossed as the operator of NYC 911 ambulances — "BIB FDNY" is EMS transport, not law enforcement — mirroring the adjudicators' FDNY clarification.
3. **Structured input** — chronological provider notes are formatted as labeled sections (`[Note i of N — timestamp — role]`) rather than concatenated, so the model can attend to arrival-relevant notes over discharge text.
4. **Decision-rule cascade (R1–R8)** — explicit if-then rules that disambiguate the hard cases (`police_initiated_ems_only` → `no_police`, `officer_as_patient_no_transport` → `no_police`, post-arrival custody out-of-scope, etc.). Applied in order; first match wins.
5. **Controlled-vocabulary `rule_fired` field** — every classification is tagged with the rule that produced it, giving an auditable error-analysis layer aligned to the cascade.
6. **Confidence rubric** — `high` / `medium` / `low` are defined in the prompt, not by feel, so the calibration analysis on the 300 is interpretable.

**No few-shot exemplars in v1.0.** Exemplars are added only during iteration on the 50-chart development set (see *Validation*), with rationale logged per prompt version. Few-shots drawn from the evaluation 262 — or from the secondary-adjudicator overlap — would leak the test set and are prohibited.

The `Keywords` column from the gold-standard data shell (which keyword triggered each chart's inclusion) is **not exposed** to the LLM. This keeps classification blind to the source-system flag and avoids anchoring bias toward a police category.

### Prompt versioning

`PROMPT_VERSION` is captured in run metadata and incremented for every change. The protocol:

- v1.0 (initial baseline): codebook + cohort framing + R1–R8 + confidence rubric + structured input. Zero few-shot exemplars. Amended pre-first-run (2026-07-08, before any scored run existed, so the version number is unchanged) to (a) add FDNY to the abbreviation list per the adjudicators' FDNY clarification, and (b) add the exploratory `custody_context` + `custody_evidence_quote` fields (see *Exploratory custody field*).
- **v1.1 (2026-07-09) — tried and reverted, not adopted.** Sharpened `police_and_ems` to strict co-transport (excluding mere police presence / bare EMS mention). On the same 50-chart dev set it performed **worse** (3-way κ 0.519→0.433; binary police-vs-none κ 0.606→0.532): it pushed the model to credit police *less*, but the primary adjudicator credits police *more* — they code police-activated/accompanied EMS psych transports (AOT/EDP pickups) as category 1, not 2, contradicting the literal "co-transport" wording. The 1-vs-2 boundary is rater-idiosyncratic and genuinely ambiguous, and at n=50 the change was below the sampling-noise floor (a same-prompt v1.0 rerun swung binary κ 0.61→0.73 across two 50-chart draws). **Reverted to v1.0, which is the locked prompt for the held-out evaluation.** v1.1 text preserved in git history (commit 72a62fe).

### Dev-set finding: report the binary flag-validation separately

Dev-set error analysis established that the classifier's disagreements with the primary adjudicator concentrate entirely on the **police-only vs. police-and-EMS (1 vs. 2)** boundary — which is rater-specific, ambiguous, and where the two human adjudicators themselves disagree (κ = 0.761). This boundary is **irrelevant to study goal 1** (validating the source LEO flag = distinguishing true police transport [1 or 2] from false-positive flags [0]). On that binary question the classifier is markedly stronger than on the 3-way (dev-set: binary agreement 0.84–0.90, κ 0.61–0.73 vs. 3-way κ 0.52). The manuscript therefore reports **both**: the binary police-vs-none agreement as the primary flag-validation result, and the 3-way κ with the explicit caveat that residual disagreement is concentrated on genuinely ambiguous co-transport documentation where the classifier's calls are largely defensible. No further prompt tuning of the 1-vs-2 boundary is performed, to avoid overfitting a single rater's idiosyncrasy at a sample size below the noise floor.
- Each subsequent version logs a rationale, the specific disagreement(s) it addresses, and the dev-set charts whose error analysis drove the change.
- The final version is **locked before any of the 262-chart evaluation set is scored**. The locked prompt is what runs on the 4,000.

## Provider, models, BAA boundary

- All inference runs on **Azure OpenAI** under the institutional Business Associate Agreement. No chart text leaves the BAA boundary.
- No consumer OpenAI endpoints, no third-party API providers, no public hosted notebooks.
- The specific model is selected at run time from those available under the BAA. The model name, deployment, and a content hash of the prompts are captured in run metadata so a run can be reproduced.

## Cost discipline

Hard ceilings:

| Cohort | Ceiling |
|---|---|
| ~312 (validation) | **$10** total |
| ~3,100 (deployment — full LEO-flagged set) | **$50** total |

Each run prints projected token cost before sending requests and aborts if projected cost exceeds the ceiling for the current cohort size. Actual cost is logged alongside outputs.

## Validation

Quality is measured **only** on the 312-encounter validation cohort:

- **Primary:** PPV of the LLM against the primary adjudicator's labels across all 312 encounters, computed per category and overall, reported with bootstrap 95% CIs.
- **Source-flag PPV (descriptive, no LLM involved):** the share of the 312 gold-standard charts coded 1 or 2 — i.e., how often the source-system LEO flag marks a true police transport. This addresses study goal 1 directly.
- **Secondary:** inter-rater agreement between the primary and secondary adjudicators on the 62-chart overlap subset, reported as Cohen's kappa. **Result: kappa = 0.761, 85.5% agreement (53/62)** — this is the human ceiling against which LLM PPV is read.
- **Confidence calibration:** check whether the LLM's self-rated `confidence` tracks correctness on the 312 — used to flag low-confidence cases for review on the 4,000.
- Disagreements (LLM vs primary adjudicator, and primary vs secondary adjudicator) are surfaced as a review table that includes the LLM's `evidence_quote` for context.

The 4,000-encounter deployment run produces descriptive estimates only; it has no gold-standard comparison. Spot-checks of the `evidence_quote` field are the auditability mechanism at deployment scale.

Sensitivity and specificity are **not reported** on this cohort design; the manuscript states this explicitly in limitations.

To prevent prompt overfitting against the validation cohort, the prompt, model deployment, split seed, and `PROMPT_VERSION` tag are **locked before scoring any of the evaluation set** (262 by design; 259 realized of the 309 analyzable). No sampling parameters are passed: the GPT-5 reasoning deployment does not accept `temperature`, so requests run at API defaults — confirmed by the archived run metadata, which records no temperature field for any scored run. (An earlier draft of this paragraph said "temperature (0)"; that was never what the runner sent and has been corrected.) Iteration during development uses synthetic fixtures plus a **50-chart dev set** drawn from the 312; the remaining **262 — including the full 62-chart secondary-adjudicator overlap** — serve as the locked-prompt evaluation set. Mechanically, the runner reads the double-coded EncounterKeys from `DOUBLE_CODED_FILE` and excludes them from the dev-eligible pool before drawing the seeded 50-chart dev sample, so the entire overlap is guaranteed to land in eval. The 50 / 262 split is larger on the dev side than a minimal pilot would be, deliberately, because edge cases will be discovered through iteration rather than pre-specified.

**The dev draw is stratified by the gold-standard label** (`GOLD_LABEL_COL`): a floor of 10 charts per category (0 / 1 / 2), remaining slots filled at random, all seeded and deterministic. The source-system LEO flag cannot stratify anything — every chart carries it — so the primary adjudicator's codes are the only differentiated labels available pre-classifier. Rationale: with a plain random 50, the rarest category (likely police + EMS) could contribute only a handful of dev charts, leaving the hardest distinction (police present vs. police merely called EMS) under-exercised before prompt lock. Using dev labels to compose the dev set leaks nothing: those labels are already consulted during iteration, the label never enters the LLM's context, and the eval set is the untouched remainder, so its composition stays representative.

The methods-paper framing is iterative prompt development, not zero-shot: *"We developed the prompt iteratively on a 50-chart development set with error-driven exemplar selection and rule-cascade refinement, then locked the prompt and evaluated on a held-out 262 (including the full 62-chart secondary-adjudicator overlap)."*

## Data handling

- Real chart text lives on the lead investigator's machine. It is **never** committed to this repository.
- Only **synthetic fixtures** — hand-crafted notes that resemble each transport-mode category — are used for in-repo tests.
- The classifier notebook reads real data via a path variable pointing outside the repo. Classifier outputs and run metadata may be written to the local working directory but are gitignored.

## Repo layout

```
.
├── README.md
├── .gitignore                       # keeps .env, data/, outputs/ out of git
├── .env.example                     # template for Minerva env vars (no secrets)
├── classifier_lib.py                # SOURCE OF TRUTH: schema, prompt, format_notes
├── transport_classifier.ipynb       # viewer: prints schema, prompt, demo
├── run_classifier_dev.ipynb         # runner: loads notes -> classifies dev set
└── docs/
    └── approach.md                   # this document
```

Minerva-local (gitignored, never copied off Minerva):

- `data/` — populated full-cohort notes table + gold-standard labels.
- `outputs/` — per-run JSONL predictions + JSON metadata sidecars + dev/eval split CSV.
- `.env` — secrets and paths; copied from `.env.example` and filled in.

Anticipated additions (not yet committed):

- `validate.ipynb` — joins classifier output to human adjudication; computes PPV with bootstrap CIs, Cohen's kappa, and confidence calibration.
- `run_classifier_eval.ipynb` — locked-prompt scoring of the 250-eval set (created only after the dev iteration is complete and the prompt is locked).
- `run_classifier_deploy.ipynb` — runs the locked classifier over the 4,000-encounter deployment cohort.
- `fixtures/` — synthetic chart text covering each category and the documented edge cases.

## Runner protocol

`run_classifier_dev.ipynb` enforces:

- **Env-driven config** — `AZURE_OPENAI_API_KEY` / `_ENDPOINT` / `_COMPLETION_DEPLOYMENT` / `_API_VERSION` read from `.env`. Missing vars raise on startup.
- **Validation universe from the gold standard** — the notes file is the full-cohort shell, so the runner intersects its EncounterKeys with the gold-standard file's keys and warns if the result differs from the expected 312. The split is drawn only from that intersection.
- **Deterministic stratified split** — fixed `SPLIT_SEED` + sorted EncounterKey list produces the same 50 / 262 assignment every time. Double-coded EncounterKeys (`DOUBLE_CODED_FILE`) are removed from the dev-eligible pool first so the full overlap lands in eval; the dev draw is then stratified by `GOLD_LABEL_COL` (floor of 10 per category, remainder random). Changing the seed after first run is prohibited.
- **Eval lock** — refuses to score any encounter outside the dev partition unless `UNLOCK_EVAL = True` is set explicitly (the eval notebook will be the only place this is true).
- **Reasoning-model conventions** — uses `max_completion_tokens` (not `max_tokens`); does not pass `temperature` (GPT-5 forces default).
- **Retry** — exponential backoff with jitter on `RateLimitError` / `APIConnectionError` / `APITimeoutError`, up to `MAX_RETRIES`.
- **Cost guard** — pre-flight character-based token estimate projects the next call's worst-case cost; aborts the run before sending if cumulative spend would exceed `COST_CEILING_USD`. Actual cost is summed from `response.usage` and written into run metadata.
- **Reproducible output** — JSONL predictions + JSON metadata sidecar capturing deployment, API version, prompt SHA-256, schema SHA-256, split seed, token totals, cost, and timing.

## Change control

If the classification rules, decision-rule cascade, or controlled vocabularies change, update **all** of the following in the same commit:

1. The definitions, decision-rule cascade, and confidence-rubric sections of this document (`docs/approach.md`).
2. `classifier_lib.py`:
   - `TransportMode` enum (if categories change).
   - `RuleFired` enum (any new / renamed / removed values).
   - `CustodyContext` enum (exploratory field categories).
   - `ChartClassification` field descriptions.
   - `SYSTEM_PROMPT` constant (codebook section, R1–R8, confidence rubric).
   - `PROMPT_VERSION` constant (increment) and `PROMPT_VERSION_NOTES` (rationale + which disagreement the change addresses).
3. The corresponding internal notes used by the human adjudicators.

`transport_classifier.ipynb` and `run_classifier_dev.ipynb` import from `classifier_lib.py`, so they pick up changes automatically — but re-run the viewer to confirm the printed output matches your edits before scoring.

If any of these drift out of sync, PPV across runs becomes uninterpretable and error analysis by `rule_fired` becomes unreliable.
