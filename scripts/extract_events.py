#!/usr/bin/env python3
"""Phase 2 orchestration: turn rule-based candidates (rules.py) into a deduplicated
set of capital/shareholder events, using the LLM only where rules are weak — never as
the primary extractor.

Documents are processed in chronological (dateDepot) order, and before each LLM call
the running cap-table state — built by replaying everything extracted from
earlier-filed documents so far, via graph.py — is injected as context: capital social
(total share capital), how shares are divided (total outstanding + who holds what),
and the value per share (nominal). This is the "context graph" design: the model
reading document N is told what's known as of document N-1, the way a human analyst
reading chronologically would, so it can catch reconciliation problems while reading
and resolve vague references ("the minority shareholders") against known holders —
not just verify arithmetic once at the very end. An earlier version of this pipeline
processed each document blind (no context) and only reconciled in Phase 4's replay;
that worked but only caught the narrow case a fixed arithmetic formula can check.

Two LLM passes, both budget-conscious:
  - VALIDATOR: reviews this document's low-confidence candidates (missing date, date
    inferred from a PV header rather than found locally, amount that doesn't
    reconcile against the known prior capital) against the running context.
  - FALLBACK: one call for a document where the rules found nothing, given the same
    running context so it can resolve vague references and sanity-check amounts.
    Most out-of-scope documents (CAC reports, address/object/fiscal-year changes)
    never reach this because they have no capital/shareholder keywords at all.

Usage:
    python scripts/extract_events.py --siren 480489707 -o output/480489707/events_candidates.json
    python scripts/extract_events.py --siren 480489707 --no-llm   # rules only, for debugging
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph as graph_module
from boilerplate import strip_boilerplate
from grounding import find_snippet_bbox
from llm_client import call_llm, extract_json
from rules import extract_document_candidates

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCORED_CODES = {
    "CAPITAL_INCREASE", "CAPITAL_DECREASE", "SHAREHOLDER_ENTRY",
    "SHAREHOLDER_END", "SHAREHOLDER_SHARE_TRANSFER", "CAPITAL_DUAL_CLASS",
}
# Rule patterns that produce informational context, not a directly-scored event on
# their own — kept for Phase 4 (cross-checking / cap-table replay), not emitted here.
INFO_CODES = {
    "CAP_TABLE_SNAPSHOT", "CAPITAL_SNAPSHOT", "CAPITAL_DECREASE_CONTEXT",
    "CAPITAL_NOMINAL_SPLIT", "CAP_TABLE_ALLOCATION", "CAP_TABLE_BUYBACK_LIST",
}
# Not informational in the same sense as the above — this one gets expanded into
# real SHAREHOLDER_ENTRY/END events during replay (graph.py), once the prior holders
# are known, which rules.py's page-local view can't determine. Kept in its own list.
TRANSITION_CODES = {"OWNERSHIP_TRANSITION_TO_SOLE"}

CAPITAL_KEYWORDS = (
    "capital social", "parts sociales", "actions", "cession", "associé", "actionnaire",
    "souscri", "augmentation", "réduction", "apport",
)

# A document having ANY rule hit is not proof it has EVERY relevant fact — the same
# mistake showed up in both directions on two different companies: on ARCHEAN, a
# clean capital-*amount* match (50k->200k) sat next to a differently-worded share
# transfer ("...autorise Monsieur Xavier AUMONT à céder 225 actions à Monsieur
# Michel CAPGRAS") the regex never anticipated — an earlier version of this gate
# skipped the fallback whenever *any* pattern fired, and silently missed it. Fixing
# that by only skipping once a *shareholder*-specific pattern had fired backfired the
# other way on HADEAN: its founding cap table matches apport_attribution_table on
# every document, which suppressed the fallback for all 9 of its filings and missed
# every capital/shareholder-movement event between snapshots (17,066 of Xavier
# AUMONT's shares vanish between two consecutive snapshots with zero event explaining
# it). No fixed rule pattern is reliable evidence that *every* other pattern is also
# covered, so the gate no longer tries to guess that at all — it runs the fallback
# whenever the document merely *mentions* capital/shareholder vocabulary at all
# (CAPITAL_KEYWORDS below), regardless of what rules.py already found. The keyword
# check itself stays a genuine token-cost lever: it is what actually filters out
# CAC-nomination/address-change/fiscal-year documents with no such vocabulary.


def load_reconstructed_docs(siren: str) -> list[dict]:
    paths = sorted(glob.glob(os.path.join(REPO_ROOT, "output", siren, "actes", "*.json")))
    return [json.load(open(p, encoding="utf-8")) for p in paths]


def load_meta(siren: str) -> dict[str, dict]:
    meta_by_id = {}
    for p in glob.glob(os.path.join(REPO_ROOT, "data", siren, "actes", "meta", "*.json")):
        m = json.load(open(p, encoding="utf-8"))
        meta_by_id[m["id"]] = m
    return meta_by_id


def to_source(cand: dict) -> dict:
    return {
        "inpi_id": cand["doc_id"],
        "page": cand["page"],
        "bbox": cand["bbox_norm"] or [0.0, 0.0, 1.0, 1.0],
        "snippet": cand["snippet"][:400],
    }


def _coerce_float(x) -> float | None:
    """LLM-fallback payloads aren't schema-checked before reaching here — a model has
    returned amount_eur as a French-formatted string ("300 000") instead of a number,
    which crashed round() downstream. Numbers-as-strings get parsed; anything else
    (a stray "non précisé" the model left instead of null) becomes None rather than
    propagating a crash three call-frames deep."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = re.sub(r"[^\d.,\-]", "", x.replace("\xa0", "").replace(" ", "")).replace(",", ".")
        try:
            return float(s) if s not in ("", "-", ".") else None
        except ValueError:
            return None
    return None


def dedup_key(cand: dict) -> tuple:
    if cand["event_code"] in ("CAPITAL_INCREASE", "CAPITAL_DECREASE"):
        amt = _coerce_float(cand["payload"].get("amount_eur"))
        after = _coerce_float(cand["payload"].get("capital_after_eur"))
        return (cand["event_code"], round(amt) if amt else None, round(after) if after else None)
    if cand["event_code"] == "SHAREHOLDER_END":
        return (cand["event_code"], cand["payload"].get("holder_name"))
    if cand["event_code"] == "CAPITAL_DUAL_CLASS":
        return (cand["event_code"], cand["payload"].get("class_name"))
    return (cand["event_code"], json.dumps(cand["payload"], sort_keys=True))


SOURCE_KIND_RANK = {"resolution": 0, "unknown": 1, "recap": 2}
DATE_CONF_RANK = {"local": 0, "inferred_from_pv_header": 1, "missing": 2}


def candidate_rank(cand: dict) -> tuple:
    return (
        SOURCE_KIND_RANK.get(cand.get("source_kind"), 1),
        DATE_CONF_RANK.get(cand.get("date_confidence"), 2),
        cand["doc_id"],
    )


def merge_duplicates(candidates: list[dict]) -> list[dict]:
    """Group candidates describing the same real-world change, pick the best-grounded
    one as primary, and keep the rest as corroborating alternates (with their dates,
    in case they disagree — e.g. AGE authorization vs. president's confirming act)."""
    groups: dict[tuple, list[dict]] = {}
    for c in candidates:
        groups.setdefault(dedup_key(c), []).append(c)

    merged = []
    for key, group in groups.items():
        group.sort(key=candidate_rank)
        primary = dict(group[0])
        alternates = [
            {"doc_id": g["doc_id"], "page": g["page"], "event_date": g["event_date"],
             "source_kind": g["source_kind"]}
            for g in group[1:]
            if g["doc_id"] != primary["doc_id"] or g["event_date"] != primary["event_date"]
        ]
        # The best-grounded snippet (source_kind="resolution") doesn't always carry
        # the most precise date locally — e.g. the president's confirming sentence
        # for the 2017 reduction has no date of its own, while a same-document recap
        # clause states it exactly ("... en date du 21 février 2017 ..."). Borrow the
        # best available date from the group rather than settle for a PV-header guess.
        if primary["date_confidence"] != "local":
            better = next((g for g in group[1:] if g["date_confidence"] == "local"), None)
            if better:
                primary["event_date"] = better["event_date"]
                primary["date_confidence"] = "borrowed_from_alternate"
        # The best-grounded candidate isn't always the richest one — a regex pattern
        # (e.g. recap_increase) never fills `method`, while a same-fact LLM-sourced
        # candidate elsewhere in the group often does. Backfill null payload fields
        # from any corroborating alternate that has them, rather than let which
        # candidate happens to win the primary slot silently drop information.
        for field, value in primary["payload"].items():
            if value is not None:
                continue
            richer = next(
                (g for g in group[1:] if g["payload"].get(field) is not None), None
            )
            if richer:
                primary["payload"][field] = richer["payload"][field]
        primary["n_corroborations"] = len(group)
        primary["alternate_sources"] = alternates
        merged.append(primary)
    return merged


def merge_rachat_context(events: list[dict], info_candidates: list[dict]) -> None:
    """Fold a rachat_price info-candidate into the CAPITAL_DECREASE it documents,
    when they share a document (buyback price/share-count context, not its own event)."""
    for info in info_candidates:
        if info["pattern"] != "rachat_price":
            continue
        for ev in events:
            if ev["event_code"] != "CAPITAL_DECREASE":
                continue
            same_doc = ev.get("doc_id") == info["doc_id"] or any(
                a["doc_id"] == info["doc_id"] for a in ev.get("alternate_sources", [])
            )
            # The buyback-and-cancel context often lives in a *separate* filing (the
            # AGE authorization) from the president's confirming decrease — match on
            # content too: at 1€ nominal, shares bought back == the amount reduced.
            same_amount = (
                info["payload"].get("shares") is not None
                and info["payload"]["shares"] == ev["payload"].get("amount_eur")
            )
            if same_doc or same_amount:
                ev["payload"]["buyback_shares"] = info["payload"].get("shares")
                ev["payload"]["buyback_price_eur"] = info["payload"].get("price_eur")


def consolidate(events: list[dict]) -> list[dict]:
    """Second dedup pass, after per-field merge_duplicates(): two candidates can
    describe the same real change while disagreeing on `amount_eur` (typically an
    LLM-fallback digit slip, e.g. 182,759 vs. the correct 185,759 for the same
    2018-03-23 increase to 400,000) — `capital_after_eur` is the more stable anchor
    since every source agrees on the resulting state, so group on that instead of on
    the possibly-noisy delta. Also collapses CAPITAL_DUAL_CLASS's many near-duplicate
    class-name spellings from the LLM fallback sweep into one entry per date.
    """
    capital_events = [e for e in events if e["event_code"] in ("CAPITAL_INCREASE", "CAPITAL_DECREASE")]
    dual_events = [e for e in events if e["event_code"] == "CAPITAL_DUAL_CLASS"]
    other_events = [e for e in events
                    if e["event_code"] not in ("CAPITAL_INCREASE", "CAPITAL_DECREASE", "CAPITAL_DUAL_CLASS")]

    groups: dict[tuple, list[dict]] = {}
    for e in capital_events:
        after = e["payload"].get("capital_after_eur")
        key = (e["event_code"], e["event_date"], round(after) if after is not None else None)
        groups.setdefault(key, []).append(e)

    merged_capital = []
    for group in groups.values():
        if len(group) == 1:
            merged_capital.append(group[0])
            continue
        group.sort(key=lambda e: (
            0 if e.get("date_confidence") != "llm_fallback" else 1,
            0 if e["payload"].get("amount_eur") is not None else 1,
        ))
        primary = group[0]
        conflicts = [
            {"amount_eur": g["payload"].get("amount_eur"), "doc_id": g["doc_id"], "page": g["page"]}
            for g in group[1:]
            if g["payload"].get("amount_eur") != primary["payload"].get("amount_eur")
        ]
        if conflicts:
            primary["payload"]["conflicting_amount_alternates"] = conflicts
        for field, value in primary["payload"].items():
            if value is not None:
                continue
            richer = next((g for g in group[1:] if g["payload"].get(field) is not None), None)
            if richer:
                primary["payload"][field] = richer["payload"][field]
        merged_capital.append(primary)

    dual_by_date: dict[str, list[dict]] = {}
    for e in dual_events:
        dual_by_date.setdefault(e["event_date"] or "unknown", []).append(e)
    merged_dual = []
    for group in dual_by_date.values():
        group.sort(key=lambda e: 0 if e.get("date_confidence") != "llm_fallback" else 1)
        best = dict(group[0])
        names = sorted({g["payload"].get("class_name", "") for g in group} - {""})
        if names:
            best["payload"]["class_name"] = ", ".join(names)[:150]
        merged_dual.append(best)

    return merged_capital + merged_dual + other_events


def summarize_state(scored_so_far: list[dict], info_so_far: list[dict],
                     transitions_so_far: list[dict]) -> str:
    """The running cap-table state from everything extracted in earlier-filed
    documents, replayed by graph.py and phrased explicitly for an LLM prompt — the
    context passed before every document: capital social, how shares are divided
    among holders, and the value per share."""
    snapshots = [c for c in info_so_far if c["pattern"] in ("repartition_table", "apport_attribution_table")]
    allocations = [c for c in info_so_far if c["pattern"] == "subscription_table"]
    nominal_splits = [c for c in info_so_far if c["pattern"] == "nominal_split"]
    buyback_lists = [c for c in info_so_far if c["pattern"] == "buyback_list"]

    if not scored_so_far and not snapshots:
        return "No capital or shareholder information established yet — this is the earliest-filed document processed so far."

    merged = consolidate(merge_duplicates(scored_so_far))
    for i, e in enumerate(merged, start=1):
        e.setdefault("event_id", f"tmp_{i:02d}")
    data = {
        "events": merged, "cap_table_snapshots": snapshots, "cap_table_allocations": allocations,
        "nominal_splits": nominal_splits, "buyback_lists": buyback_lists,
        "sole_holder_transitions": transitions_so_far,
    }
    result = graph_module.replay(data)
    tl = result["capital_timeline"]
    if not tl:
        return "No capital or shareholder information established yet — this is the earliest-filed document processed so far."
    last = tl[-1]

    lines = [f"Known state as of {last['as_of']} (from {len(merged)} events found in earlier-filed documents):"]
    lines.append(
        f"- Capital social (total share capital): {last['capital_eur']} EUR"
        if last["capital_eur"] is not None else "- Capital social: not yet established"
    )
    lines.append(
        f"- Division of shares (total shares outstanding): {last['shares_total']}"
        if last["shares_total"] is not None else "- Total shares outstanding: not yet established"
    )
    lines.append(
        f"- Value per share (nominal): {last['nominal_eur']} EUR/share"
        if last["nominal_eur"] is not None else "- Value per share: not yet established"
    )
    if last["holders"]:
        lines.append("- Sócios/shareholders known, with their share of the division: " + "; ".join(
            f"{h['name']} ({h['kind']}, {h['shares']} shares, {h['pct']}%)" for h in last["holders"]
        ))
    else:
        lines.append("- Sócios/shareholders: none named yet")
    if nominal_splits:
        last_split = nominal_splits[-1]
        lines.append(
            f"- A par-value split has occurred (factor x{last_split['payload']['factor']}) — share "
            f"counts were multiplied and nominal value divided accordingly; the figures above already "
            f"reflect it."
        )
    if result.get("notes"):
        lines.append("- Open gaps/uncertainties flagged so far: " + " | ".join(result["notes"][-3:]))
    return "\n".join(lines)


CONTEXTUAL_VALIDATOR_SYSTEM = """You validate capital/shareholder events extracted from OCR'd French corporate \
filings (procès-verbaux, statuts) — some by regex, some by an LLM reading a document the regex found nothing in, \
so treat every candidate as unverified regardless of its source. You are given (1) the KNOWN STATE as of just \
before this document — capital social, how shares are divided among holders, and the value per share, built \
from every earlier-filed document already processed — and (2) candidate events this document's text produced, \
each with its parsed fields and exact snippet. Mark a CAPITAL_INCREASE/CAPITAL_DECREASE invalid — do not just \
reconcile its numbers — when the snippet is actually a routine profit allocation ("affecter le bénéfice", \
"réserve légale", "report à nouveau", "poste Autres Réserves") rather than an actual decision to change share \
capital ("augmenter/réduire le capital social"): moving profit into a reserves account is ordinary annual \
bookkeeping and does not touch capital_eur, even when the snippet mentions a "réserves" line matching the \
payload's amount_eur by coincidence — only "incorporation de réserves" (an explicit resolution capitalizing \
reserves into new shares) is a real capital increase. For everything that does describe a real capital change, \
check whether the stated amount actually reconciles with the known prior capital (capital_before + amount_eur \
should equal capital_after_eur); if it does not, but a small OCR-plausible digit change would fix it, propose \
that corrected amount. Reply ONLY a JSON array, one object per candidate id: {"id": "...", "valid": true|false, \
"corrected_event_date": "YYYY-MM-DD" or null, "corrected_amount_eur": number or null (only if you are \
correcting a reconciliation mismatch), "note": "short reason, especially if valid=false or something was \
corrected"}. No prose outside the JSON array."""

CONTEXTUAL_FALLBACK_SYSTEM = """You extract capital/shareholder events from a French corporate legal filing \
(procès-verbal or statuts) that regex rules already checked and found nothing in — read carefully for wording \
rules would miss. You are given the KNOWN STATE as of just before this document (capital social, how shares \
are divided among holders, value per share — from every earlier-filed document already processed) — use it to \
resolve vague references ("the minority shareholders", "the associates") to specific known holders when the \
context makes the referent unambiguous, and to sanity-check any amount against the known prior capital. Only \
these event codes count, exactly as named: CAPITAL_INCREASE (amount_eur, capital_after_eur, method), \
CAPITAL_DECREASE (amount_eur, capital_after_eur, method), SHAREHOLDER_ENTRY (holder_name, holder_siren?, \
shares?), SHAREHOLDER_END (holder_name, holder_siren?, shares?), SHAREHOLDER_SHARE_TRANSFER (from_name, \
to_name, shares, price_eur?), CAPITAL_DUAL_CLASS (class_name, description). event_date is when the decision \
took effect, NOT the filing/deposit date. Ignore auditors, officers, address, name, or corporate-object \
changes — out of scope. If there is truly nothing in scope, return an empty array. Reply ONLY a JSON array of \
objects: {"event_code": "...", "payload": {...}, "event_date": "YYYY-MM-DD" or null, "page": <int>, \
"snippet": "the exact quote you based this on, verbatim from the text"}."""


def contextual_validate(candidates: list[dict], context_str: str) -> None:
    to_check = [
        c for c in candidates
        if c.get("date_confidence") != "local"
        or (c["event_code"] in ("CAPITAL_INCREASE", "CAPITAL_DECREASE") and c["payload"].get("amount_eur") is None)
    ]
    if not to_check:
        return
    for i, c in enumerate(to_check):
        c["_vid"] = f"v{i}"
    payload = [
        {"id": c["_vid"], "event_code": c["event_code"], "payload": c["payload"],
         "event_date": c["event_date"], "date_confidence": c["date_confidence"], "snippet": c["snippet"][:400]}
        for c in to_check
    ]
    user = f"KNOWN STATE BEFORE THIS DOCUMENT:\n{context_str}\n\nCANDIDATES FROM THIS DOCUMENT:\n" + \
        json.dumps(payload, ensure_ascii=False)
    try:
        raw = call_llm(CONTEXTUAL_VALIDATOR_SYSTEM, user, max_tokens=2000)
        results = {r["id"]: r for r in extract_json(raw)}
    except Exception as e:
        print(f"contextual validator call failed, leaving candidates as-is: {e}", file=sys.stderr)
        for c in to_check:
            del c["_vid"]
        return
    for c in to_check:
        r = results.get(c["_vid"])
        if r:
            c["llm_valid"] = r.get("valid")
            c["llm_note"] = r.get("note")
            if r.get("corrected_event_date"):
                c["event_date"] = r["corrected_event_date"]
                c["date_confidence"] = "llm_context_corrected"
            if r.get("corrected_amount_eur") is not None:
                c["payload"]["amount_eur_source_text"] = c["payload"].get("amount_eur")
                c["payload"]["amount_eur"] = _coerce_float(r["corrected_amount_eur"])
                c["llm_context_amount_correction"] = True
        del c["_vid"]


def contextual_fallback(doc: dict, context_str: str) -> list[dict]:
    # Strip statute boilerplate (denomination/siège/durée/gouvernance articles,
    # letterhead) before it reaches the prompt — see boilerplate.py. Confirmed by a
    # corpus-wide frequency scan to repeat verbatim and never carry a capital/
    # shareholder decision, so this only ever removes dead weight from the prompt,
    # never a candidate event; rules.py's own regex passes still run on the
    # unstripped text.
    reduced_text, _stats = strip_boilerplate(doc["full_text"])
    user = f"KNOWN STATE BEFORE THIS DOCUMENT:\n{context_str}\n\nDOCUMENT TEXT:\n" + reduced_text[:12000]
    try:
        raw = call_llm(CONTEXTUAL_FALLBACK_SYSTEM, user, max_tokens=1500)
        hits = extract_json(raw)
    except Exception as e:
        print(f"contextual fallback call failed for {doc['doc_id']}: {e}", file=sys.stderr)
        return []
    if isinstance(hits, dict):
        hits = hits.get("events") or hits.get("candidates") or ([hits] if hits.get("event_code") else [])
    if not isinstance(hits, list):
        print(f"contextual fallback returned unexpected shape for {doc['doc_id']}: {type(hits)}", file=sys.stderr)
        return []
    extra = []
    for h in hits:
        if not isinstance(h, dict) or h.get("event_code") not in SCORED_CODES:
            continue
        page = h.get("page") or 1
        page_obj = next((p for p in doc["pages"] if p["page"] == page), None)
        bbox = None
        if page_obj and h.get("snippet"):
            bbox, _method = find_snippet_bbox(page_obj, h["snippet"])
        payload = dict(h.get("payload") or {})
        # Model output isn't schema-checked — coerce the numeric fields graph.py and
        # dedup_key() expect to actually be numbers (a French-formatted string like
        # "300 000" has been observed from the claude-cli fallback model).
        for key in ("amount_eur", "capital_after_eur", "shares", "price_eur"):
            if key in payload:
                payload[key] = _coerce_float(payload[key])
        extra.append({
            "doc_id": doc["doc_id"], "page": page, "pattern": "llm_fallback_contextual",
            "event_code": h["event_code"], "payload": payload,
            "event_date": h.get("event_date"), "source_kind": "llm_fallback_contextual",
            "date_confidence": "llm_fallback_contextual", "snippet": h.get("snippet", ""),
            "bbox_norm": bbox,
        })
    return extra


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--siren", default="480489707")
    ap.add_argument("--no-llm", action="store_true", help="rules only, skip both LLM passes")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    docs = load_reconstructed_docs(args.siren)
    meta_by_id = load_meta(args.siren)
    docs_sorted = sorted(
        docs, key=lambda d: (meta_by_id.get(d["doc_id"]) or {}).get("dateDepot") or "9999-99-99"
    )

    all_candidates: list[dict] = []
    scored_so_far: list[dict] = []
    info_so_far: list[dict] = []
    transitions_so_far: list[dict] = []
    processing_log: list[dict] = []

    for doc in docs_sorted:
        dep = (meta_by_id.get(doc["doc_id"]) or {}).get("dateDepot")
        context_str = summarize_state(scored_so_far, info_so_far, transitions_so_far)

        cands = extract_document_candidates(doc)
        all_candidates.extend(cands)
        doc_scored = [c for c in cands if c["event_code"] in SCORED_CODES]
        doc_info = [c for c in cands if c["event_code"] in INFO_CODES]
        doc_transitions = [c for c in cands if c["event_code"] in TRANSITION_CODES]

        low_text = doc["full_text"].lower()
        n_fallback = 0
        if not args.no_llm and any(k in low_text for k in CAPITAL_KEYWORDS):
            fb = contextual_fallback(doc, context_str)
            doc_scored.extend(fb)
            all_candidates.extend(fb)
            n_fallback = len(fb)

        # Validate AFTER the fallback sweep, over the combined set — a fallback-
        # produced candidate is exactly as unverified as a low-confidence rule one
        # (more so: nothing has checked its snippet at all yet). Running validation
        # only on rule candidates, before the fallback ran, left fallback output
        # completely unchecked — which is how a routine "affecter le bénéfice ...
        # au poste Autres Réserves" (profit allocation to reserves, capital
        # untouched) got mislabeled CAPITAL_INCREASE with an invented
        # capital_after_eur nowhere in the snippet.
        if not args.no_llm:
            contextual_validate(doc_scored, context_str)
            rejected = [c for c in doc_scored if c.get("llm_valid") is False]
            for c in rejected:
                print(f"  rejected by validator: {c['event_code']} {c['payload']} — {c.get('llm_note')}",
                      file=sys.stderr)
            doc_scored = [c for c in doc_scored if c.get("llm_valid") is not False]

        scored_so_far.extend(doc_scored)
        info_so_far.extend(doc_info)
        transitions_so_far.extend(doc_transitions)

        processing_log.append({
            "doc_id": doc["doc_id"], "dateDepot": dep, "context_given": context_str,
            "n_rule_scored": len(doc_scored) - n_fallback, "n_llm_fallback": n_fallback,
        })
        print(f"[{dep}] {doc['doc_id'][:8]}: {len(doc_scored) - n_fallback} rule events, "
              f"{n_fallback} contextual-fallback events", file=sys.stderr)

    scored = scored_so_far
    info = info_so_far
    transitions = transitions_so_far

    events = merge_duplicates(scored)
    events = consolidate(events)
    merge_rachat_context(events, info)

    snapshots = [c for c in info if c["pattern"] in ("repartition_table", "apport_attribution_table")]
    allocations = [c for c in info if c["pattern"] == "subscription_table"]
    nominal_splits = [c for c in info if c["pattern"] == "nominal_split"]
    buyback_lists = [c for c in info if c["pattern"] == "buyback_list"]

    # A holder-table's date is backfilled from doc-level PV headers (rules.py), which
    # can false-positive on an illustrative date buried in boilerplate statutes text.
    # A filing cannot legitimately describe a cap-table state dated after its own
    # deposit, so that's used as a sanity bound — fall back to it when the inferred
    # date fails it.
    for s in snapshots + allocations + nominal_splits + buyback_lists + transitions:
        dep = (meta_by_id.get(s["doc_id"]) or {}).get("dateDepot")
        if dep and (s["event_date"] is None or s["event_date"] > dep):
            s["event_date"] = dep
            s["date_confidence"] = "fallback_deposit_date"

    # Identify-the-tranche inference: a buyback list can name exactly who is being
    # bought out, but that only tells us who they are as of the buyback's date. When
    # an earlier CAPITAL_INCREASE has no subscriber allocation (would go to an
    # UNKNOWN pseudo-holder in graph.py) and its share delta matches a buyback list's
    # total exactly, the buyback list almost certainly names that increase's original
    # subscribers too. Synthesize a retroactive allocation for that increase, sourced
    # to the document where the names actually appear, clearly flagged as inferred.
    for bl in buyback_lists:
        total = sum(h["shares"] or 0 for h in bl["payload"]["holders"])
        candidate_increase = next(
            (e for e in events
             if e["event_code"] == "CAPITAL_INCREASE"
             and e["payload"].get("amount_eur") is not None
             and abs(e["payload"]["amount_eur"] - total) < 1
             and not any(a["event_date"] == e["event_date"] for a in allocations)),
            None,
        )
        if candidate_increase:
            allocations.append({
                "doc_id": bl["doc_id"], "page": bl["page"], "pattern": "subscription_table",
                "event_code": "CAP_TABLE_ALLOCATION",
                "payload": {"allocations": [
                    {"name": h["name"], "new_shares": h["shares"]} for h in bl["payload"]["holders"]
                ]},
                "event_date": candidate_increase["event_date"],
                "source_kind": "inferred_from_later_buyback_list",
                "date_confidence": "inferred_retroactive",
                "snippet": bl["snippet"],
                "bbox_norm": bl["bbox_norm"],
                "note": (
                    f"Names not stated in the {candidate_increase['event_date']} increase's own filing — "
                    f"inferred from the buyback list in {bl['doc_id']} ({bl['event_date']}), whose total "
                    f"({total:.0f}) matches this increase's share delta exactly."
                ),
            })

    events.sort(key=lambda e: e["event_date"] or "9999-99-99")
    for i, e in enumerate(events, start=1):
        e["event_id"] = f"evt_{i:02d}"
        e["source"] = to_source(e)
        e["meta_hint"] = (meta_by_id.get(e["doc_id"]) or {}).get("typeRdd")
    for s in snapshots + allocations + nominal_splits + buyback_lists + transitions:
        s["source"] = to_source(s)

    out_path = args.out or os.path.join(REPO_ROOT, "output", args.siren, "events_candidates.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result = {
        "siren": args.siren,
        "mode": "chronological_context_aware",
        "n_documents": len(docs),
        "n_raw_candidates": len(all_candidates),
        "n_events": len(events),
        "events": events,
        "cap_table_snapshots": snapshots,
        "cap_table_allocations": allocations,
        "nominal_splits": nominal_splits,
        "buyback_lists": buyback_lists,
        "sole_holder_transitions": transitions,
        "processing_log": processing_log,
    }
    json.dump(result, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"wrote {out_path}: {len(events)} events, {len(snapshots)} cap-table snapshots, "
          f"{len(allocations)} subscription allocations, {len(buyback_lists)} buyback lists, "
          f"{len(transitions)} sole-holder transitions "
          f"(from {len(all_candidates)} raw candidates across {len(docs)} documents)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
