#!/usr/bin/env python3
"""Phase 4: replay events chronologically into capital_timeline snapshots.

The entity model is a small networkx graph — nodes are holders (deduplicated by
normalized name) plus the company itself, edges are "shareholder_of" with a share
count as of the latest replay step. At this scale (a handful of holders, ~15 events)
a graph database would be pure overhead; networkx gives node dedup and edge
attributes for free with zero infrastructure, and the graph is what actually gets
"replayed" — walking events in date order and updating edge weights — rather than
just a static picture drawn at the end.

Design decisions baked into the replay (see PLAN.md for the full reasoning):
  - A cap-table SNAPSHOT (an explicit "répartition suivante" table in the OCR) is
    ground truth and overwrites current holdings outright.
  - A CAPITAL_INCREASE's new shares go to a same-date subscription ALLOCATION table
    when one exists; failing that, if the method is an incorporation of reserves
    (pro-rata by definition — no subscription choice involved) they are distributed
    proportionally to current holders; failing that, they are recorded against an
    explicit "UNKNOWN" pseudo-holder rather than silently dropped, so
    sum(holders.shares) always reconciles to shares_total and the gap is visible
    instead of hidden.
  - A CAPITAL_NOMINAL_SPLIT multiplies every holder's share count (and divides the
    nominal) — capital is unchanged, so no share/holder gap opens up here.
  - SHAREHOLDER_SHARE_TRANSFER moves shares between two named holders directly.
    SHAREHOLDER_ENTRY/END are recorded as the events they are, but the actual holder
    dict is driven by transfers/snapshots/allocations — they are confirmation, not a
    second mechanism (this is the "two sides of the same movement" the brief warns
    about: we do not double-apply a transfer and an entry/end as if they were
    independent quantity changes).
  - A CAPITAL_DECREASE by buyback-and-cancel is applied against the "UNKNOWN"
    pseudo-holder first when the bought-back share count matches an unresolved
    tranche exactly (see the 2017 event: 150,861 shares repurchased "from minority
    shareholders" is exactly the size of the unallocated 2008 Actions-B tranche) —
    a documented inference, not a guess: it is flagged in the snapshot's note either
    way.

Usage:
    python scripts/graph.py --siren 480489707 -o output/480489707/capital_timeline.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date as _date
from itertools import groupby

import networkx as nx

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Known OCR-noise variants of the same person, observed in this corpus (see
# challenges/actes exploration notes — "Xavien AUMONT", stray civility prefixes,
# etc.). Exact mapping beats fuzzy clustering at this scale (five known people):
# it's auditable, and fuzzy matching on short French surnames risks false merges.
NAME_FIXES = {
    "Xavien AUMONT": "Xavier AUMONT",
    "Monsiduß Xavier AUMONT": "Xavier AUMONT",
    "Xavier AuMONT": "Xavier AUMONT",
    "Monsieur Xavier AUMONT": "Xavier AUMONT",
    "Monsieur Antonio BLANCO MARINA": "Antonio BLANCO MARINA",
    "Antonio BLANCO": "Antonio BLANCO MARINA",
    "Monsieur Franck GICQUEL": "Franck GICQUEL",
    "Franck, Charles GICQUEL": "Franck GICQUEL",
    "Monsieur Michel CAPGRAS": "Michel CAPGRAS",
}


def normalize_name(name: str) -> str:
    n = " ".join(name.split()).replace("’", "'")
    return NAME_FIXES.get(n, n)


def guess_kind(name: str) -> str:
    """French filings write a natural person as 'Firstname SURNAME' (mixed case) and
    an entity — companies, funds, the holding itself — in full capitals ('FPCI
    SECURITE', 'HADEAN'). A name with no lowercase letters at all is reliably one of
    the latter in this corpus; every real person we've extracted has mixed case."""
    letters = "".join(c for c in name if c.isalpha())
    return "COMPANY" if letters and letters.isupper() else "PERSON"


def load_data(siren: str) -> dict:
    path = os.path.join(REPO_ROOT, "output", siren, "events_candidates.json")
    return json.load(open(path, encoding="utf-8"))


def reconcile_doc_dates(events: list[dict], items: list[dict]) -> None:
    """A same-document auxiliary item (nominal split, buyback list, ...) can end up
    dated differently from the capital event it actually belongs to — its own date is
    extracted from page stamps/headers that are less reliable than the multi-document
    corroboration a CAPITAL_INCREASE/DECREASE gets (see extract_events.py). Concretely:
    the 2017 buyback list is dated to the AGE *authorization* (2017-01-20) while the
    matching CAPITAL_DECREASE is dated to the président's *confirming* decision
    (2017-02-21) — same document, different date, so grouped into different replay
    steps unless reconciled here. When an item shares its document with a capital
    event, adopt that event's date.

    Matched by nearest event_date, not by doc_id: a restated statutes document
    narrates the company's ENTIRE capital history in its "ARTICLE 6 - APPORTS" recap,
    so the same doc_id legitimately corroborates many different dated events — a
    doc_id-keyed lookup binds to whichever date is encountered first and silently
    mis-reconciles every other item that happens to share that document. The item's
    own extracted date is already roughly right (from a PV header or deposit-date
    fallback, typically within days of the true date); nearest-by-date is robust to
    which specific document ends up "primary" for a given event between runs, since
    it doesn't depend on doc_id matching at all. Capped at 45 days so an item never
    silently jumps to an unrelated, distant event.
    """
    capital_events = [e for e in events if e["event_code"] in ("CAPITAL_INCREASE", "CAPITAL_DECREASE")]
    for item in items:
        if not item["event_date"]:
            continue
        item_date = _date.fromisoformat(item["event_date"])
        closest = min(
            capital_events,
            key=lambda e: abs((_date.fromisoformat(e["event_date"]) - item_date).days),
            default=None,
        )
        if closest and abs((_date.fromisoformat(closest["event_date"]) - item_date).days) <= 45:
            item["event_date"] = closest["event_date"]


def build_ops(data: dict) -> list[tuple[str, list[dict]]]:
    ops = []
    for e in data["events"]:
        if e["event_date"]:
            ops.append({"date": e["event_date"], "kind": "event", "item": e})
    for s in data["cap_table_snapshots"]:
        if s["event_date"]:
            ops.append({"date": s["event_date"], "kind": "snapshot", "item": s})
    for a in data["cap_table_allocations"]:
        if a["event_date"]:
            ops.append({"date": a["event_date"], "kind": "allocation", "item": a})
    for n in data["nominal_splits"]:
        if n["event_date"]:
            ops.append({"date": n["event_date"], "kind": "nominal_split", "item": n})
    for b in data.get("buyback_lists", []):
        if b["event_date"]:
            ops.append({"date": b["event_date"], "kind": "buyback_list", "item": b})
    for t in data.get("sole_holder_transitions", []):
        if t["event_date"]:
            ops.append({"date": t["event_date"], "kind": "sole_holder_transition", "item": t})
    ops.sort(key=lambda o: o["date"])
    return [(d, list(g)) for d, g in groupby(ops, key=lambda o: o["date"])]


UNKNOWN_KEY_TMPL = "UNKNOWN (unallocated shares from {event_id}, {date})"


def is_pro_rata_method(method: str | None) -> bool:
    if not method:
        return False
    m = method.lower()
    return "incorporation" in m and ("réserve" in m or "reserve" in m)


def replay(data: dict) -> dict:
    events = data["events"]
    reconcile_doc_dates(events, data["nominal_splits"])
    reconcile_doc_dates(events, data.get("buyback_lists", []))
    grouped = build_ops(data)

    g = nx.DiGraph()
    g.add_node("ARCHEAN TECHNOLOGIES", kind="COMPANY")

    holders: dict[str, float] = {}
    holder_kind: dict[str, str] = {}
    capital_eur = None
    shares_total = None
    nominal_eur = None

    timeline = []
    global_notes: list[str] = []

    def set_holder(name: str, shares: float, kind: str | None = None) -> None:
        kind = kind or guess_kind(name)
        holders[name] = shares
        holder_kind[name] = kind
        if not g.has_node(name):
            g.add_node(name, kind=kind)
        g.add_edge(name, "ARCHEAN TECHNOLOGIES", relation="shareholder_of", shares=shares)

    def add_holder(name: str, delta: float, kind: str | None = None) -> None:
        set_holder(name, holders.get(name, 0) + delta, kind)

    for date, items in grouped:
        caused_by: list[str] = []
        step_notes: list[str] = []

        for op in items:
            if op["kind"] == "snapshot":
                s = op["item"]
                holders.clear()
                for h in s["payload"]["holders"]:
                    name = normalize_name(h["name"])
                    set_holder(name, h["shares"] or 0)
                shares_total = sum(holders.values()) or shares_total
                if capital_eur and shares_total:
                    nominal_eur = round(capital_eur / shares_total, 4)
                caused_by.append(f"snapshot:{s['doc_id']}:p{s['page']}")

        for op in items:
            if op["kind"] == "nominal_split":
                ns = op["item"]
                factor = ns["payload"]["factor"]
                for k in list(holders):
                    holders[k] *= factor
                    g[k]["ARCHEAN TECHNOLOGIES"]["shares"] = holders[k]
                if shares_total:
                    shares_total *= factor
                if nominal_eur:
                    nominal_eur = nominal_eur / factor
                caused_by.append(f"nominal_split:{ns['doc_id']}:p{ns['page']}")
                step_notes.append(f"par-value split by {factor}: share counts multiplied, capital unchanged.")

        buyback_list_for_date = next((o["item"] for o in items if o["kind"] == "buyback_list"), None)
        # SHAREHOLDER_ENTRY and SHAREHOLDER_SHARE_TRANSFER are the "two sides of the
        # same movement" the brief warns about — when both fire the same date for the
        # same incoming holder (a transfer that happens to also be that holder's
        # first entry, e.g. CAPGRAS receiving AUMONT's 225 shares), applying ENTRY's
        # own `shares` figure AND the transfer's add_holder() would double it. The
        # transfer is the mechanism; let it own the quantity, ENTRY only confirms identity.
        transfer_targets_this_date = {
            normalize_name(o["item"]["payload"]["to_name"])
            for o in items
            if o["kind"] == "event" and o["item"]["event_code"] == "SHAREHOLDER_SHARE_TRANSFER"
        }

        for op in items:
            if op["kind"] == "sole_holder_transition":
                t = op["item"]
                new_name = t["payload"]["new_sole_holder_name"]
                new_siren = t["payload"].get("new_sole_holder_siren")
                prior_total = sum(holders.values())
                prior_holders = list(holders.keys())
                holders.clear()
                set_holder(new_name, prior_total, kind="COMPANY")
                g.nodes[new_name]["siren"] = new_siren
                caused_by.append(f"sole_holder_transition:{t['doc_id']}:p{t['page']}")
                step_notes.append(
                    f"{new_name} (SIREN {new_siren}) recorded as sole shareholder, replacing "
                    f"{', '.join(prior_holders) if prior_holders else 'an unknown prior cap table'} "
                    f"({prior_total:.0f} shares total) — the transfer/apport document itself is not "
                    f"in this corpus; reconstructed by diffing the cap table before/after this filing, "
                    f"same as SHAREHOLDER_END's documented inference rule."
                )

        for op in items:
            if op["kind"] != "event":
                continue
            e = op["item"]
            code = e["event_code"]
            caused_by.append(e["event_id"])

            if code == "CAPITAL_INCREASE":
                amt = e["payload"].get("amount_eur")
                after = e["payload"].get("capital_after_eur")
                method = e["payload"].get("method")
                prev_total = shares_total
                prev_capital = capital_eur

                # Arithmetic cross-check: amount_eur should equal capital_after -
                # capital_before. Where it doesn't, but a recorded conflicting
                # alternate (see extract_events.py's consolidate()) does reconcile
                # exactly, the alternate is almost certainly the accurate reading —
                # e.g. the 2018-03-23 increase's recap text reads "185 759" but only
                # "182 759" satisfies 217,241 -> 400,000 exactly (a one-digit OCR
                # slip the text-only extraction had no way to catch on its own).
                if amt is not None and after is not None and prev_capital is not None:
                    expected = after - prev_capital
                    if abs(amt - expected) > 1:
                        alt = next(
                            (a["amount_eur"] for a in e["payload"].get("conflicting_amount_alternates", [])
                             if a["amount_eur"] is not None and abs(a["amount_eur"] - expected) <= 1),
                            None,
                        )
                        if alt is not None:
                            step_notes.append(
                                f"amount_eur corrected from {amt:.0f} to {alt:.0f} to reconcile "
                                f"{prev_capital:.0f} -> {after:.0f} exactly (the source text's literal "
                                f"reading, {amt:.0f}, is kept on the event as a flagged alternate)."
                            )
                            e["payload"]["amount_eur_source_text"] = amt
                            amt = e["payload"]["amount_eur"] = alt
                        else:
                            step_notes.append(
                                f"amount_eur ({amt:.0f}) does not reconcile with capital before/after "
                                f"({prev_capital:.0f} -> {after:.0f}, expected {expected:.0f}) — left as-is, "
                                f"corpus/OCR inconsistency."
                            )

                if after is not None:
                    capital_eur = after
                if nominal_eur and after is not None:
                    shares_total = round(after / nominal_eur)
                elif amt is not None and nominal_eur:
                    shares_total = (prev_total or 0) + round(amt / nominal_eur)
                elif after is not None:
                    # Capital moved but the nominal value per share isn't established
                    # yet (this only happens before the first holder-level snapshot,
                    # e.g. right after the founding increase and before the transfer
                    # that first gives us named, summable holdings) — leaving the old
                    # shares_total in place would silently show a stale, wrong number;
                    # None here is honest, and the next snapshot re-derives it exactly.
                    shares_total = None
                    step_notes.append(
                        "shares_total unknown at this point: capital changed but the nominal "
                        "value per share was not yet established from a holder-level snapshot."
                    )
                delta = (shares_total - (prev_total or 0)) if (shares_total is not None and prev_total is not None) else None

                # Match the allocation to THIS increase by its share total, not just
                # by date — two increases can land on the same date (the 2008-06-27
                # Actions A and Actions B tranches both did), and a shared
                # `allocation_for_date` would double-apply one allocation to both.
                matching_allocation = next(
                    (o["item"] for o in items if o["kind"] == "allocation"
                     and delta and abs(sum(a["new_shares"] or 0 for a in o["item"]["payload"]["allocations"]) - delta) < 1),
                    None,
                )

                if delta and delta > 0:
                    if matching_allocation:
                        for a in matching_allocation["payload"]["allocations"]:
                            add_holder(normalize_name(a["name"]), a["new_shares"] or 0)
                        step_notes.append(
                            f"{delta:.0f} new shares allocated per the subscription table in "
                            f"{matching_allocation['doc_id']}."
                        )
                    elif is_pro_rata_method(method) and holders:
                        base = sum(holders.values()) or 1
                        ratio = (base + delta) / base
                        for k in list(holders):
                            holders[k] *= ratio
                        step_notes.append(
                            f"{delta:.0f} new shares from incorporation of reserves — "
                            f"distributed pro-rata to existing holders."
                        )
                    else:
                        key = UNKNOWN_KEY_TMPL.format(event_id=e["event_id"], date=date)
                        add_holder(key, delta, kind="UNKNOWN")
                        step_notes.append(
                            f"{delta:.0f} new shares issued — no subscription record found in "
                            f"the corpus for this tranche; recorded against '{key}'."
                        )
                elif amt is None or after is None:
                    step_notes.append(
                        "increase reported without a confirmed amount in the corpus "
                        "(proposal/authorization only) — capital/shares not updated."
                    )

            elif code == "CAPITAL_DECREASE":
                amt = e["payload"].get("amount_eur")
                after = e["payload"].get("capital_after_eur")
                buyback_shares = e["payload"].get("buyback_shares")
                prev_total = shares_total
                if after is not None:
                    capital_eur = after
                if nominal_eur and after is not None:
                    shares_total = round(after / nominal_eur)
                delta = (prev_total - shares_total) if (prev_total is not None and shares_total is not None) else amt

                # Most precise first: a buyback list naming exactly who was bought
                # out (see rules.py's buyback_list pattern) — subtract each named
                # holder's own listed share count directly, rather than inferring
                # from a size match against an anonymous tranche. When the event
                # itself doesn't carry a buyback_shares figure (no rachat_price
                # context merged in), fall back to the share delta computed from
                # capital before/after — this is what makes a decrease that exactly
                # reverses an earlier unallocated increase (e.g. an authorized-then-
                # cancelled employee share plan) net out to zero instead of leaving
                # a phantom UNKNOWN balance that never gets removed.
                match_target = buyback_shares or delta
                unknown_matches = [
                    k for k in holders
                    if k.startswith("UNKNOWN") and match_target and abs(holders[k] - match_target) < 1
                ]
                if buyback_list_for_date:
                    removed = []
                    for h in buyback_list_for_date["payload"]["holders"]:
                        name = normalize_name(h["name"])
                        if name in holders:
                            holders[name] -= h["shares"] or 0
                            removed.append(f"{h['name']} ({h['shares']:.0f})")
                    step_notes.append(
                        f"buyback applied by name per the list in {buyback_list_for_date['doc_id']}: "
                        + ("; ".join(removed) if removed else "none of the named holders were found in the current cap table")
                    )
                elif unknown_matches:
                    k = unknown_matches[0]
                    del holders[k]
                    if g.has_node(k):
                        g.remove_node(k)
                    reason = ("the resolution states this buyback targets minority shareholders"
                              if buyback_shares else "the decrease exactly reverses this unallocated tranche")
                    step_notes.append(
                        f"{match_target:.0f}-share decrease matches the unallocated tranche "
                        f"'{k}' exactly — cancelled against it rather than a named holder ({reason})."
                    )
                elif delta:
                    step_notes.append(
                        f"{delta:.0f}-share capital decrease could not be matched to a specific "
                        f"holder or unallocated tranche — cap table below is NOT adjusted for it; "
                        f"treat holder shares as provisional for this snapshot."
                    )

            elif code == "SHAREHOLDER_SHARE_TRANSFER":
                frm = normalize_name(e["payload"]["from_name"])
                to = normalize_name(e["payload"]["to_name"])
                shares = e["payload"].get("shares") or 0
                if frm in holders:
                    holders[frm] -= shares
                    g[frm]["ARCHEAN TECHNOLOGIES"]["shares"] = holders[frm]
                add_holder(to, shares)
                g.add_edge(frm, to, relation="transferred_shares_to", shares=shares, as_of=date)

            elif code == "SHAREHOLDER_ENTRY":
                name = normalize_name(e["payload"]["holder_name"])
                shares = e["payload"].get("shares")
                if name in transfer_targets_this_date:
                    pass  # quantity owned by the same-date SHAREHOLDER_SHARE_TRANSFER — don't double it
                elif name not in holders and shares:
                    set_holder(name, shares)
                elif name not in holders:
                    set_holder(name, 0)
                    step_notes.append(f"{name} recorded as entering — share count not stated here.")

            elif code == "SHAREHOLDER_END":
                name = normalize_name(e["payload"]["holder_name"])
                if name in holders and holders[name] > 0:
                    step_notes.append(
                        f"{name} recorded as exiting, but still shows {holders[name]:.0f} shares "
                        f"in the replayed state — their prior holding was never captured (no "
                        f"snapshot/allocation names them before this point); treat as a known gap."
                    )
                holders.pop(name, None)
                if g.has_node(name):
                    g.remove_node(name)

        # zero/negative holders are noise from a partial reconciliation — drop, note
        for k in [k for k, v in holders.items() if v is not None and v <= 0]:
            step_notes.append(f"{k} reached {holders[k]:.0f} shares and was dropped from the cap table.")
            del holders[k]
            if g.has_node(k):
                g.remove_node(k)

        holders_out = [
            {
                "name": name,
                "kind": holder_kind.get(name, "UNKNOWN"),
                "shares": round(shares, 2),
                "pct": round(100 * shares / shares_total, 4) if shares_total else None,
                **({"siren": g.nodes[name]["siren"]} if g.has_node(name) and g.nodes[name].get("siren") else {}),
            }
            for name, shares in sorted(holders.items(), key=lambda kv: -kv[1])
        ]
        entry = {
            "as_of": date,
            "capital_eur": capital_eur,
            "shares_total": shares_total,
            "nominal_eur": nominal_eur,
            "holders": holders_out,
            "caused_by": caused_by,
        }
        if step_notes:
            entry["notes"] = step_notes
            global_notes.extend(f"{date}: {n}" for n in step_notes)
        timeline.append(entry)

    return {"capital_timeline": timeline, "notes": global_notes, "graph_nodes": g.number_of_nodes(),
            "graph_edges": g.number_of_edges()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--siren", default="480489707")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    data = load_data(args.siren)
    result = replay(data)

    # replay() can correct an event's amount_eur in place (arithmetic cross-check
    # against capital_before/after) — persist that back so Phase 6 (build_results.py)
    # reads the corrected figure, not the original.
    events_path = os.path.join(REPO_ROOT, "output", args.siren, "events_candidates.json")
    if os.path.exists(events_path):
        json.dump(data, open(events_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    out_path = args.out or os.path.join(REPO_ROOT, "output", args.siren, "capital_timeline.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(result, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"wrote {out_path}: {len(result['capital_timeline'])} snapshots, "
          f"{result['graph_nodes']} graph nodes, {len(result['notes'])} reconciliation notes",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
