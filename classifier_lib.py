"""Schema, prompt, and helpers for the police-transport ED chart classifier.

Single source of truth for:
- TransportMode and RuleFired enums
- ChartClassification Pydantic model
- PROMPT_VERSION, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
- format_notes helper

Both transport_classifier.ipynb (viewer) and run_classifier_dev.ipynb
(runner) import from this module. See docs/approach.md for the
operational definitions, decision-rule cascade, and change-control rules.
"""
from enum import Enum
from typing import Iterable, Literal

from pydantic import BaseModel, Field


PROMPT_VERSION = "v1.0"

PROMPT_VERSION_NOTES = """\
v1.0 (initial) — Codebook-derived definitions, NYC-ED cohort framing,
decision-rule cascade (R1–R8), rule_fired controlled vocabulary,
confidence rubric, structured-notes input format. Zero few-shot exemplars.

v1.0 pre-first-run amendment (2026-07-08, before any scored run existed,
so the version number is unchanged): added FDNY to the NYC abbreviation
list. FDNY EMS operates the NYC 911 ambulance system, so "BIB FDNY" is
EMS transport, not law enforcement. Motivated by the adjudicators' own
FDNY clarification during gold-standard coding.

v1.0 pre-first-run amendment #2 (2026-07-08): added the EXPLORATORY
custody_context field (+ custody_evidence_quote). No human gold standard
exists for custody — the adjudicators coded transport mode only — so
this field is reported descriptively with evidence-quote spot-checks and
is labeled as not human-validated in the manuscript. It answers the
custody-context question descriptively. transport_mode and
its codebook are unchanged.

v1.0 pre-first-run amendment #3 (2026-07-08): corrected the input
description. The real note pull contains ALL ED chart note types (ED
Psychiatric, ED Notes, ED Provider, Triage/Intake, nursing, consults,
progress notes, ...), not provider notes only — confirmed by the data
probe. The prompt now describes the actual input; the runner excludes
boilerplate types (Discharge Instructions, Attestation) before
classification. This also matches what the human adjudicators read.

v1.1 (2026-07-09) — TRIED AND REVERTED, NOT ADOPTED. Sharpened
police_and_ems to strict co-transport (mere police presence / bare EMS
mention excluded). Motivated by 50-chart dev error analysis. Result: it
made things WORSE on the same 50 (3-way kappa 0.519 -> 0.433; binary
police-vs-none 0.606 -> 0.532) because it pushed the model to credit
police LESS, whereas the primary adjudicator actually credits police
MORE (they code police-activated/accompanied EMS psych transports as
category 1, not 2 — contradicting the literal co-transport wording). The
1-vs-2 boundary proved rater-idiosyncratic and ambiguous, and at n=50 the
change was below the sampling-noise floor (a same-prompt v1.0 rerun swung
binary kappa 0.61->0.73 across two 50-chart draws). Reverted to v1.0 and
LOCKED for the held-out evaluation. v1.1 text is preserved in git history
(commit 72a62fe). See docs/approach.md change-control log.

Future versions are added during iteration on the 50-chart development
set. Each version must:
  - increment PROMPT_VERSION
  - log rationale + the specific disagreement(s) the change addresses
  - be locked before any of the 262-chart evaluation set is scored
  - have a corresponding entry in docs/approach.md change-control list
"""


class TransportMode(str, Enum):
    """Three-category transport classification based on documented
    transportation and custody at ED arrival.
    """

    ONLY_POLICE = "only_police"
    POLICE_AND_EMS = "police_and_ems"
    NO_POLICE = "no_police"


class CustodyContext(str, Enum):
    """EXPLORATORY secondary field: why police were involved at arrival.

    No human gold standard exists for this field (the adjudicators coded
    transport mode only), so it is reported descriptively and is not
    part of the PPV validation.
    """

    CRIMINAL_CUSTODY = "criminal_custody"
    INVOLUNTARY_PSYCH_HOLD = "involuntary_psych_hold"
    POLICE_INVOLVED_NO_CUSTODY = "police_involved_no_custody"
    NOT_APPLICABLE = "not_applicable"


# Numeric codes used by the human adjudicators in the gold-standard master
# file: 0 = not police transport, 1 = police transport, 2 = police + EMS.
# The validation notebook maps gold labels through this before comparing
# against classifier output.
GOLD_LABEL_MAP = {
    0: TransportMode.NO_POLICE,
    1: TransportMode.ONLY_POLICE,
    2: TransportMode.POLICE_AND_EMS,
}


class RuleFired(str, Enum):
    """Controlled vocabulary for which decision rule (R1–R8) produced the
    classification. One value per encounter.
    """

    TRANSPORT_BY_POLICE_ONLY = "transport_by_police_only"
    CO_TRANSPORT_POLICE_AND_EMS = "co_transport_police_and_ems"
    POLICE_PRESENT_AT_ARRIVAL_WITH_EMS = "police_present_at_arrival_with_ems"
    EMS_ONLY_TRANSPORT = "ems_only_transport"
    SELF_OR_OTHER_TRANSPORT = "self_or_other_transport"
    POLICE_INITIATED_EMS_ONLY = "police_initiated_ems_only"
    OFFICER_AS_PATIENT_NO_TRANSPORT = "officer_as_patient_no_transport"
    NO_LEO_MENTION = "no_leo_mention"
    AMBIGUOUS_DOCUMENTATION = "ambiguous_documentation"


class ChartClassification(BaseModel):
    """LLM output schema for a single ED encounter chart."""

    transport_mode: TransportMode = Field(
        ...,
        description=(
            "Transport-and-custody-at-arrival classification.\n\n"
            "- only_police: Patient brought to ED by law enforcement or in "
            "police custody, with no indication of EMS/ambulance transport.\n"
            "- police_and_ems: Both law enforcement and EMS involved in "
            "transport. Police must have actually transported or been "
            "physically present at arrival; merely initiating the EMS call "
            "does NOT qualify.\n"
            "- no_police: No evidence of LEO involvement in transport or "
            "custody. Includes EMS-only and self/other transport, and the "
            "officer-as-patient edge case when the officer was not "
            "transported or detained by police."
        ),
    )
    rule_fired: RuleFired = Field(
        ...,
        description=(
            "Controlled-vocabulary tag identifying which decision rule "
            "produced the transport_mode classification. See the "
            "decision-rules section of the system prompt for the rule "
            "cascade. Use AMBIGUOUS_DOCUMENTATION only when the notes are "
            "genuinely contradictory or silent on transport, and pair it "
            "with confidence='low'."
        ),
    )
    evidence_quote: str = Field(
        ...,
        description=(
            "Short verbatim quote from the notes that drove the "
            "classification (target <=200 chars). Empty string if no "
            "directly relevant text exists (rule_fired=no_leo_mention)."
        ),
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description=(
            "Self-rated confidence per the rubric in the system prompt:\n"
            "- high: arrival mode unambiguously documented.\n"
            "- medium: documented but with ambiguity (e.g., 'with police' — "
            "escort or just present?).\n"
            "- low: not documented, or contradictory across notes.\n"
            "Applies to transport_mode only, not custody_context."
        ),
    )
    custody_context: CustodyContext = Field(
        ...,
        description=(
            "EXPLORATORY: why police were involved at arrival.\n\n"
            "- criminal_custody: arrest-related custody (under arrest, "
            "handcuffed, prisoner, medical clearance for booking or "
            "arraignment, precinct/corrections custody).\n"
            "- involuntary_psych_hold: involuntary psychiatric removal or "
            "hold (EDP transport, 70/10, MHL 9.41 removal, involuntary "
            "evaluation) — NOT criminal custody even when officers use "
            "custody language.\n"
            "- police_involved_no_custody: police involved in transport or "
            "present at arrival but no custody or hold documented (e.g., "
            "assault victim driven in by officers).\n"
            "- not_applicable: required when transport_mode = no_police."
        ),
    )
    custody_evidence_quote: str = Field(
        ...,
        description=(
            "Short verbatim quote supporting custody_context (target <=200 "
            "chars). Empty string when custody_context is not_applicable or "
            "no directly relevant text exists."
        ),
    )


SYSTEM_PROMPT = """You are a clinical chart abstractor classifying how an emergency department (ED) patient arrived, based on the free-text ED provider notes from a single encounter.

# Setting and data

- Encounters are from a New York City emergency department.
- Charts have been pre-flagged in the electronic health record as potentially involving law enforcement; some flags are false positives, which is the point of this classification task.
- Input is the ED chart notes from a single encounter, presented in chronological order in the user message: provider notes (resident, attending, PA, NP), psychiatric notes, triage/intake and nursing documentation, consults, and progress/event notes. Arrival-mode language most often appears in triage/intake and nursing documentation and in the HPI of provider notes.
- Expect informal documentation and NYC-specific abbreviations: BIB ("brought in by"), NYPD, EMS, EDP ("emotionally disturbed person"), 70/10 (NYPD psychiatric pickup code), perp, ESU (NYPD Emergency Service Unit), FDNY (Fire Department of the City of New York; FDNY EMS operates NYC 911 ambulances, so "BIB FDNY" means EMS/ambulance transport, NOT law enforcement).

# Outcome categories

Assign exactly one of three transport-mode categories based on the documented transportation and custody at arrival.

ONLY POLICE
- Patient brought to the ED by law enforcement or in police custody, with no indication of EMS or ambulance transport.
- Qualifying language examples: "BIB NYPD", "under arrest", "in police custody" without ambulance involvement.

POLICE AND EMS
- Both law enforcement and EMS are involved in transport.
- Police must have ACTUALLY TRANSPORTED the patient OR been PHYSICALLY PRESENT at arrival.
- Police merely initiating the EMS call (e.g., "NYPD called EMS") does NOT qualify if police did not transport or accompany.
- Qualifying language examples: "BIB EMS with NYPD", police escort accompanying ambulance, both EMS/ambulance and police explicitly documented at arrival or triage.

NO POLICE
- No evidence of law enforcement involvement in transport or custody.
- Includes EMS-only transport and self / family / other transport.
- Edge case: if the patient is themselves a police officer (e.g., NYPD officer presenting for injury) but was not transported or detained by police, classify as NO POLICE.

Out of scope (assume not to occur; do NOT weight toward a police category):
- Post-arrival custody changes (patient arrives independently, later placed under arrest at bedside).
- Officer-as-patient transported by a fellow officer.

# Decision rules (apply in order; first match wins)

R1. Patient is identified as a police officer AND no mention of police transporting or accompanying this patient.
    -> transport_mode = no_police, rule_fired = officer_as_patient_no_transport

R2. Police called / notified EMS, but no documentation of police transporting the patient or being physically present at arrival.
    -> transport_mode = no_police, rule_fired = police_initiated_ems_only

R3. Both police AND EMS documented as transporting, OR police physically present at arrival / triage alongside EMS.
    -> transport_mode = police_and_ems
       rule_fired = co_transport_police_and_ems   (both transported)
       rule_fired = police_present_at_arrival_with_ems   (EMS transported, police present)

R4. Police transporting the patient WITHOUT any EMS involvement.
    -> transport_mode = only_police, rule_fired = transport_by_police_only

R5. EMS transport documented with no law enforcement mention.
    -> transport_mode = no_police, rule_fired = ems_only_transport

R6. Self / family / walk-in arrival, with no law enforcement involvement.
    -> transport_mode = no_police, rule_fired = self_or_other_transport

R7. No relevant arrival-mode or LEO language anywhere in the notes.
    -> transport_mode = no_police, rule_fired = no_leo_mention

R8. Documentation is genuinely contradictory across notes, or arrival mode is referenced but ambiguous.
    -> make best judgment for transport_mode, set confidence = low, rule_fired = ambiguous_documentation

# Confidence rubric

- high: arrival mode unambiguously documented with explicit transport-mode language.
- medium: arrival mode documented but with some ambiguity (e.g., "with police" — escort or just present?).
- low: arrival mode not documented, contradictory across notes, or based on indirect inference.

The confidence rating applies to transport_mode only.

# Custody context (secondary field)

Separately from transport mode, classify WHY police were involved at arrival. This is orthogonal to transport_mode: a handcuffed patient can arrive by ambulance (police_and_ems + criminal_custody).

- criminal_custody: arrest-related custody. Language: "under arrest", handcuffed, "prisoner", medical clearance for booking/arraignment, brought from precinct or corrections, officers guarding an arrestee.
- involuntary_psych_hold: involuntary psychiatric removal or hold. Language: EDP transport, "70/10", MHL 9.41 removal, "brought for involuntary psychiatric evaluation". IMPORTANT: officers may use custody-like language for these removals ("taken into custody for evaluation") — that is NOT criminal custody; classify it here.
- police_involved_no_custody: police transported or were present, but no custody or hold is documented (e.g., assault victim driven in by officers, officer standby without detention).
- not_applicable: REQUIRED when transport_mode = no_police.

If both criminal and psychiatric custody language appear, choose the one that better explains the arrival itself, and quote that language.

# Output

Return a JSON object that conforms exactly to the provided schema:
- transport_mode: one of "only_police", "police_and_ems", "no_police"
- rule_fired: one of the controlled-vocabulary values from R1–R8
- evidence_quote: short verbatim quote from the notes that drove the transport_mode decision (empty string only when rule_fired = no_leo_mention)
- confidence: "high", "medium", or "low" per the rubric above (transport_mode only)
- custody_context: one of "criminal_custody", "involuntary_psych_hold", "police_involved_no_custody", "not_applicable"
- custody_evidence_quote: short verbatim quote supporting custody_context (empty string when not_applicable or no directly relevant text)
"""


USER_PROMPT_TEMPLATE = """Encounter notes (chronological order):

{formatted_notes}

Classify this encounter.
"""


def format_notes(notes: Iterable[dict]) -> str:
    """Format a list of ED provider notes for inclusion in the user prompt.

    Each note dict must contain a 'text' field. Optional fields included in
    the section header if present: 'timestamp', 'role'.
    """
    notes = list(notes)
    n_total = len(notes)
    chunks = []
    for i, n in enumerate(notes, start=1):
        header_parts = [f"Note {i} of {n_total}"]
        if n.get("timestamp"):
            header_parts.append(str(n["timestamp"]))
        if n.get("role"):
            header_parts.append(str(n["role"]))
        header = " — ".join(header_parts)
        chunks.append(f"[{header}]\n{n['text'].strip()}")
    return "\n\n".join(chunks)
