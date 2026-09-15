#!/usr/bin/env python3
"""Phase 6: assemble a results.json from the pipeline's intermediate outputs.

For the challenge's actual subject (SIREN 480489707) this validates against
challenges/actes/schema/results.schema.json — whose siren field is pinned to that
SIREN — and writes to the repo root, exactly as the submission requires. For any
other SIREN in the corpus (run via scripts/run_pipeline.sh <siren> first) it builds
the same shape for exploration, skips the siren-pinned schema check, and writes to
output/<siren>/results.json instead of the root, so it never collides with the real
submission file.

Usage:
    python scripts/build_results.py                    # ARCHEAN, writes ./results.json
    python scripts/build_results.py --siren 499979540   # any other corpus company
    python scripts/build_results.py --siren 499979540 -o output/499979540/results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHALLENGE_SIREN = "480489707"

ARCHEAN_NOTES = """\
Pipeline: rule-based extraction (regex calibrated on this corpus' actual phrasing) \
first, an LLM (Groq, openai/gpt-oss-120b) only as a validator for low-confidence \
candidates and a fallback reader for documents where the rules found nothing — see \
scripts/rules.py, extract_events.py, graph.py. capital_timeline is produced by a \
deterministic chronological replay (graph.py), not by an LLM narrating the answer.

Known gaps, left as gaps rather than guessed:
- The 1,130 new shares from the 2005-05-17 increase have no named subscriber in the \
corpus; the next hard evidence (2005-08-16) already shows the post-cession state, so \
the 2005-05-17 snapshot's shares_total is left null rather than shown stale.
- The 2008-06-27 "Actions A" tranche (17,241 shares, evt_09) has no subscriber named \
anywhere in the 17 filed acts and is carried through to 2018 as an explicit UNKNOWN \
holder rather than assumed to be HADEAN.
- HADEAN's acquisition of 100% of ARCHEAN TECHNOLOGIES (first evidenced 2008-06-27, \
"Associée Unique") has no transfer/apport document in this corpus at all — \
reconstructed by diffing the cap table before/after that filing opens, the same rule \
applied to any SHAREHOLDER_END with no explicit cession text.
- A 20-share discrepancy exists between the Aug-2005 répartition table (BLANCO 823 / \
AUMONT 617) and the attendance sheet of the Oct-2006 AGE for the same two \
shareholders (BLANCO 803 / AUMONT 637) — no document in the corpus records that \
movement. This pipeline uses the Aug-2005 figures as its base. A cross-check against \
HADEAN's own 2007 incorporation act (SIREN 499979540, outside this challenge's \
scope) states Xavier AUMONT contributed exactly 742 ARCHEAN shares, which matches \
the attendance-sheet-based figure (637+330-225) rather than this pipeline's \
(617+330-225=722) — the attendance-sheet numbers are likely the more accurate ones; \
not corrected here since only the répartition table is captured by the extraction code.
- The 2018-03-23 increase's amount is internally inconsistent in its own source \
document: the operative text reads "185,759 €" but only "182,759 €" reconciles the \
known before/after capital (217,241 -> 400,000) exactly. 182,759 is used; 185,759 is \
kept on the event as a flagged conflicting_amount_alternates entry — likely a single \
OCR-plausible digit slip (5<->2), not corrected in the source itself.
- The 4 institutional Actions-B subscribers (FPCI SECURITE, FIP GALIA PME 4, GALIA \
VENTURE, FPCI FINANCIERE DE BRIENNE) are never named in the 2008 increase's own \
filing — only in the 2017 buyback list, whose total (150,861) matches that tranche's \
size exactly. Their 2008 entry is inferred from that later document, flagged as such \
on the allocation, not asserted as directly sourced.

Cross-validation: an independent blind re-read of the same 17 documents (a separate \
agent, no access to this code or its output) converged on the same event set and, \
notably, found the same 185,759-vs-182,759 arithmetic inconsistency on its own.

Group bonus: see the top-level "group" field, built by scripts/extract_group.py \
(regex over the DGFiP Cerfa filiales/participations template, an LLM fallback for the \
free-text and attendance-sheet cases regex can't parse, entity resolution against this \
corpus' own 20 known SIRENs). Chain found: HADEAN (100% owner, via its own annual \
filings' 2033-G form) -> ARCHEAN TECHNOLOGIES -> ARCHEAN LABS / ARCHEAN MOTION (100% \
subsidiaries, via ARCHEAN's 2059-G form) -> and one hop further back, AIR SYSTEM \
SERVICE (~7.9% of HADEAN, visible only in HADEAN's own 2008 acte and a 2019 \
attendance sheet — never in anything ARCHEAN itself filed). ARCHEAN MOTION's own \
filing lists it under the parent's SIREN rather than its own; resolved:false rather \
than trusting that value. A few group.edges are exact duplicates (same relation, same \
date, corroborated by more than one source) not yet deduplicated — see README.md.

Corpus-wide: this results.json covers ARCHEAN TECHNOLOGIES' own 17 actes only. \
CAPITAL_DUAL_CLASS events are included where grounded but are not scored per \
event_codes.json.
"""


def load(siren: str, name: str) -> dict:
    return json.load(open(os.path.join(REPO_ROOT, "output", siren, name), encoding="utf-8"))


def clean_event(e: dict) -> dict:
    out = {
        "event_id": e["event_id"],
        "event_code": e["event_code"],
        "event_date": e["event_date"],
        "payload": e["payload"],
        "source": e["source"],
    }
    if e.get("n_corroborations"):
        out["n_corroborations"] = e["n_corroborations"]
    if e.get("alternate_sources"):
        out["alternate_sources"] = e["alternate_sources"]
    if e.get("llm_note"):
        out["extraction_note"] = e["llm_note"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--siren", default=CHALLENGE_SIREN)
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()
    siren = args.siren
    is_challenge_subject = siren == CHALLENGE_SIREN

    events_data = load(siren, "events_candidates.json")
    timeline_data = load(siren, "capital_timeline.json")

    events = []
    for e in events_data["events"]:
        if not e.get("event_date"):
            print(f"dropping {e['event_id']} ({e['event_code']}): no resolvable event_date", file=sys.stderr)
            continue
        events.append(clean_event(e))

    result = {
        "siren": siren,
        "events": events,
        "capital_timeline": timeline_data["capital_timeline"],
        "notes": ARCHEAN_NOTES.strip() if is_challenge_subject else (
            f"Exploratory run for SIREN {siren} — same pipeline as the ARCHEAN "
            f"TECHNOLOGIES (480489707) submission, not itself a challenge deliverable."
        ),
    }

    if is_challenge_subject:
        group_path = os.path.join(REPO_ROOT, "output", CHALLENGE_SIREN, "group.json")
        if os.path.exists(group_path):
            result["group"] = json.load(open(group_path, encoding="utf-8"))
            print(
                f"merged group.json: {len(result['group']['nodes'])} nodes, "
                f"{len(result['group']['edges'])} edges", file=sys.stderr,
            )

    if is_challenge_subject:
        schema_path = os.path.join(REPO_ROOT, "challenges", "actes", "schema", "results.schema.json")
        schema = json.load(open(schema_path, encoding="utf-8"))
        try:
            import jsonschema
            jsonschema.validate(result, schema)
            print("schema: OK", file=sys.stderr)
        except ImportError:
            print("jsonschema not installed — skipping validation (pip install jsonschema)", file=sys.stderr)
        except Exception as e:
            print(f"SCHEMA VALIDATION FAILED: {e}", file=sys.stderr)
            return 1
    else:
        print(f"siren {siren} != {CHALLENGE_SIREN} — schema's 'siren' const only fits the challenge "
              f"subject, skipping validation", file=sys.stderr)

    default_out = os.path.join(REPO_ROOT, "results.json") if is_challenge_subject else \
        os.path.join(REPO_ROOT, "output", siren, "results.json")
    out_path = args.out or default_out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(result, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"wrote {out_path}: {len(events)} events, {len(result['capital_timeline'])} timeline snapshots",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
