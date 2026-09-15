"""Rule-based (regex) extraction of capital/shareholder events from reconstructed OCR
text — the primary extraction path. An LLM is only used downstream (extract_events.py)
as a fallback for documents these rules find nothing in, and as a validator over what
they do find. Every pattern here was calibrated against the actual phrasing found in
ARCHEAN TECHNOLOGIES' 17 actes, not guessed in the abstract:

- The recap sentence "Aux termes de l'assemblée générale extraordinaire du <date>, le
  capital social a été augmenté/réduit de <amount> euros afin d'être porté/ramené à
  <amount> euros." appears, verbatim or near-verbatim, in the restated "ARTICLE 6 -
  APPORTS" of every later statutes document — it is the single most reliable source for
  CAPITAL_INCREASE/CAPITAL_DECREASE, and it already carries the *decision* date, not the
  deposit date.
- The operative resolution text ("DEUXIEME RESOLUTION ... constate la réalisation
  définitive de l'augmentation de capital de 113 000 € ... pour porter le capital à
  150 000 €") is the primary source when the document IS the decision, vs. a later
  restatement narrating it.
- Shareholder tables ("... suivant la répartition suivante : Monsieur X N actions ...")
  appear both at constitution and after a transfer, in the same "Name N actions [pct%]"
  shape.
"""

from __future__ import annotations

import re

from grounding import bbox_for_char_range, flatten_page

FR_MONTHS = {
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5,
    "juin": 6, "juillet": 7, "août": 8, "aout": 8, "septembre": 9,
    "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12,
}

DATE_SLASH_RE = re.compile(r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b")
DATE_TEXT_RE = re.compile(
    r"\b(\d{1,2})\s?(?:er)?\s+(" + "|".join(FR_MONTHS) + r")\s+(\d{4})\b", re.IGNORECASE
)

AMOUNT_RE = r"[\d](?:[\d.,\s ]{0,14}\d)?"


def parse_fr_amount(raw: str) -> float | None:
    """Parse a French-formatted number: '150.000' / '150 000' / '368 102' / '2,72'.

    Heuristic: digits grouped in 3s with '.'/' ' are thousands separators (whole
    euros, the norm in these filings); a short 1-2 digit tail after ',' or '.' is a
    decimal (share prices like '2,72 euros par action').
    """
    s = raw.strip().replace(" ", " ")
    if re.fullmatch(r"\d{1,3}([.\s]\d{3})+", s):
        return float(s.replace(".", "").replace(" ", ""))
    if re.fullmatch(r"\d+[.,]\d{1,2}", s):
        return float(s.replace(",", "."))
    if re.fullmatch(r"\d+", s):
        return float(s)
    # mixed noise (e.g. OCR dropped a separator) — strip everything but digits as a
    # last resort, keeping it flagged as low-confidence by the caller.
    digits = re.sub(r"\D", "", s)
    return float(digits) if digits else None


def _find_all_dates(text: str) -> list[tuple[int, int, str]]:
    """Every date-like substring in `text` as (start, end, 'YYYY-MM-DD'), position order."""
    found = []
    for m in DATE_SLASH_RE.finditer(text):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            found.append((m.start(), m.end(), f"{y:04d}-{mo:02d}-{d:02d}"))
    for m in DATE_TEXT_RE.finditer(text):
        mo = FR_MONTHS[m.group(2).lower()]
        found.append((m.start(), m.end(), f"{int(m.group(3)):04d}-{mo:02d}-{int(m.group(1)):02d}"))
    found.sort(key=lambda t: t[0])
    return found


def parse_fr_date(text: str) -> str | None:
    found = _find_all_dates(text)
    return found[0][2] if found else None


def nearest_date(flat: str, pos: int, window: int = 450) -> str | None:
    """Look for the date closest to character offset `pos` — closest preceding date
    first, then closest following date.

    A generous window: French legal sentences routinely put the operative date 200+
    chars from the number it governs ("... en date du 16 août 2005" trailing well
    after "cession de la totalité des actions détenues par ..."). Picking the
    *closest* match (not just the first one in the window) matters because a page can
    stack several dated clauses back to back (a restated "ARTICLE 6" recaps five
    capital changes in a row) — the first date in a naive left-to-right window search
    would keep leaking into the wrong sentence.
    """
    before = flat[max(0, pos - window) : pos]
    dates = _find_all_dates(before)
    if dates:
        return dates[-1][2]  # rightmost = closest to pos
    after = flat[pos : pos + window]
    dates = _find_all_dates(after)
    return dates[0][2] if dates else None


# A PV (procès-verbal) bundle in these filings often stacks several assemblies at
# different dates in one document (see acte_2006-01-04, which bundles an AGO of
# 2005-09-02 and an AGE of 2005-08-08). A match's own page can be too far from the
# date that governs it for `nearest_date` to reach — so extract_document_candidates
# also builds a doc-wide "which PV header date is in force by this page" timeline.
PV_HEADER_RE = re.compile(
    r"\bLe\s+(\d{1,2})\s?(?:er)?\s+(" + "|".join(FR_MONTHS) + r")\s+(\d{4})\s*,",
    re.IGNORECASE,
)
PV_HEADER_SLASH_RE = re.compile(
    r"(?:ASSEMBL[ÉE]E G[ÉE]N[ÉE]RALE|DELIBERATIONS)[^\n]{0,40}?\bDU\s+(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})",
    re.IGNORECASE,
)
# Two ways an incidental date deep in prose gets mistaken for a genuine PV/assembly
# header: (1) a recap paragraph narrating old history ("Aux termes de l'assemblée
# générale extraordinaire du 17/05/2005, le capital social a été augmenté...") trips
# PV_HEADER_SLASH_RE even though it isn't a header — but it sits well into the page,
# while a real header opens it, so bounding matches to the page's first ~600 chars
# screens this out. (2) A founder's biography inside an apport-en-nature filing
# ("...aux termes de leur contrat de mariage ... le 5 septembre 1992, préalablement
# à leur union célébrée...") can trip PV_HEADER_RE even near a page's top — bounded
# by position alone it still gets through, so matches near these keywords are
# additionally excluded by content.
HEADER_MAX_POS = 1000  # generous enough to clear registry-stamp noise atop page 1
HEADER_EXCLUDE_CONTEXT = re.compile(
    r"mariage|notaire|union célébrée|contrat de mariage", re.IGNORECASE
)


def _looks_like_biographical_date(flat: str, start: int, end: int, window: int = 100) -> bool:
    ctx = flat[max(0, start - window) : end + 20]
    return bool(HEADER_EXCLUDE_CONTEXT.search(ctx))
# The notarial formula "L'an <year>, [...] le <day> <month>" (e.g. "L'an 2008, Et le
# 27 juin, A 9 heures 30 ..."), where the year is stated before the day/month rather
# than after them — DATE_TEXT_RE alone misses this ordering.
PV_HEADER_LAN_RE = re.compile(
    r"L['’]an\s+(\d{4})[^.]{0,60}?[Ll]e\s+(\d{1,2})\s?(?:er)?\s+("
    + "|".join(FR_MONTHS) + r")\b",
    re.IGNORECASE,
)


def find_pv_headers(doc: dict) -> list[tuple[int, str]]:
    """Return [(page, 'YYYY-MM-DD'), ...] for every PV/assembly header date found,
    in document (page) order."""
    headers: list[tuple[int, str]] = []
    for page in doc["pages"]:
        flat, _ = flatten_page(page)
        for m in PV_HEADER_RE.finditer(flat):
            if m.start() > HEADER_MAX_POS or _looks_like_biographical_date(flat, m.start(), m.end()):
                continue
            d = int(m.group(1))
            mo = FR_MONTHS[m.group(2).lower()]
            y = int(m.group(3))
            headers.append((page["page"], f"{y:04d}-{mo:02d}-{d:02d}"))
        for m in PV_HEADER_SLASH_RE.finditer(flat):
            if m.start() > HEADER_MAX_POS:
                continue
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            y = y + 2000 if y < 100 else y
            headers.append((page["page"], f"{y:04d}-{mo:02d}-{d:02d}"))
        for m in PV_HEADER_LAN_RE.finditer(flat):
            if m.start() > HEADER_MAX_POS:
                continue
            y = int(m.group(1))
            d = int(m.group(2))
            mo = FR_MONTHS[m.group(3).lower()]
            headers.append((page["page"], f"{y:04d}-{mo:02d}-{d:02d}"))
    headers.sort(key=lambda t: t[0])
    return headers


def date_in_force_at(headers: list[tuple[int, str]], page: int) -> str | None:
    """Most recent PV header date at or before `page`.

    Deliberately returns None — not some other header from later in the document —
    when nothing qualifies: a header appearing only on a much later page (e.g. a
    restated-statutes recap that happens to open near the top of its own page) is not
    "in force" at an earlier one, and guessing it anyway produced a real bug (a
    buyback list on page 3 silently bound to a recap header on page 10, years off).
    Returning None here lets the caller's dateDepot-bound fallback take over instead
    of a wrong guess.
    """
    candidates = [d for p, d in headers if p <= page]
    return candidates[-1] if candidates else None


RESOLUTION_MARKERS = ("décide", "decide", "constate la réalisation", "constate que")
RECAP_MARKER = "aux termes"


def classify_source_kind(flat: str, pos: int, window: int = 200) -> str:
    ctx = flat[max(0, pos - window) : pos + window].lower()
    if RECAP_MARKER in ctx:
        return "recap"
    if any(m in ctx for m in RESOLUTION_MARKERS):
        return "resolution"
    return "unknown"


# ---------------------------------------------------------------------------
# Pattern definitions. Each returns raw hits: (match, event_code, payload, verb_pos)
# ---------------------------------------------------------------------------

PAT_RECAP_INCREASE = re.compile(
    r"le capital social (?:a été|est)\s+augment[ée]\s+de\s+(?P<amount>" + AMOUNT_RE + r")\s*(?:€|euros)"
    r"[^.]{0,180}?port[ée]e?\s+à\s+(?P<after>" + AMOUNT_RE + r")\s*(?:€|euros)",
    re.IGNORECASE,
)

PAT_RECAP_DECREASE = re.compile(
    r"le capital social (?:a été|est)\s+réduit\s+de\s+(?P<amount>" + AMOUNT_RE + r")\s*(?:€|euros)"
    r"[^.]{0,180}?ramen[ée]e?\s+(?:de\s+(?P<before>" + AMOUNT_RE + r")\s*(?:€|euros)\s+à|à)"
    r"\s*(?P<after>" + AMOUNT_RE + r")\s*(?:€|euros)",
    re.IGNORECASE,
)

PAT_DIRECT_INCREASE = re.compile(
    r"(?:constate la réalisation(?: définitive)? de l['’]augmentation de capital|"
    r"augmentation de capital(?: en numéraire)? d['’]un montant(?: nominal)? de)\s+"
    r"de?\s*(?P<amount>" + AMOUNT_RE + r")\s*(?:€|euros)"
    r"(?:[^.]{0,220}?(?:pour porter|afin de porter|porter) le capital(?: social)? à\s+"
    r"(?P<after>" + AMOUNT_RE + r")\s*(?:€|euros))?",
    re.IGNORECASE,
)

# "décide ... d'augmenter le capital social d'une somme de X euros, pour le porter de
# Y euros à Z euros" — the phrasing an operative resolution uses when it (unlike the
# retrospective recap sentence) is the actual decision, not a restated summary of one.
PAT_DECIDE_INCREASE = re.compile(
    r"d['’]augmenter le capital social d['’]une somme de\s+(?P<amount>" + AMOUNT_RE + r")\s*(?:€|euros)"
    r"[^.]{0,120}?pour le porter de\s+(?P<before>" + AMOUNT_RE + r")\s*(?:€|euros)\s+à\s+"
    r"(?P<after>" + AMOUNT_RE + r")\s*(?:€|euros)",
    re.IGNORECASE,
)

PAT_NOMINAL_SPLIT = re.compile(
    r"divise[r]?\s+la valeur nominale des actions par\s+(?P<factor>\d+)", re.IGNORECASE
)

PAT_STATUTE_SNAPSHOT = re.compile(
    r"capital social est fixé à la somme de\s+(?P<amount>" + AMOUNT_RE + r")\s*"
    r"(?:\(?" + AMOUNT_RE + r"\)?\s*)?(?:€|euros)",
    re.IGNORECASE,
)

PAT_DUAL_CLASS = re.compile(
    r"[Cc]réation d['’](?:une|deux|des)? ?(?:nouvelles? )?actions? de préférence"
    r"(?: de catégorie)?\s*(?P<codes>[A-Z]'?(?:\s*(?:et|,)\s*[A-Z]'?)*)",
)

# A buyback-and-cancel decrease often names exactly who was bought out: "fixe la
# liste des associés concernés comme suit : - à FPCI SECURITE 64 655 actions - à ...".
# This is the mirror of a subscription table (shares leaving named holders, not
# arriving) — found by a blind independent re-read of this corpus that the original
# rachat_price pattern (price/count only) missed entirely.
PAT_BUYBACK_LIST_TRIGGER = re.compile(
    r"liste des (?:associés|actionnaires) concernés\s*(?:comme suit)?\s*:?", re.IGNORECASE
)

# Fund/entity names can end in a bare digit ("FIP GALIA PME 4") which collides with
# the share count that follows — a plain \d+[\s\d]* run would swallow both into one
# number. Requiring proper thousands-grouping (an optional leading 1-3 digits, then
# space-separated exact 3-digit groups) makes "12 931" unambiguous from "...PME 4".
BUYBACK_LINE_RE = re.compile(
    r"à\s+(?P<name>[A-Z][A-Z0-9À-Üa-zà-ÿ&\-\.\s]{2,45}?)\s+(?P<shares>\d{1,3}(?:\s\d{3})*)\s+actions"
)

# "La société HADEAN, société par actions simplifiée au capital de ..., sous le
# numéro 499 979 540, Associée Unique de la [Société]" — a filing can simply *open*
# by stating a new sole shareholder, with no separate document recording how the
# previous holders' shares got there. `.{0,200}?`/`.{0,80}?` (not `[^.]{0,N}?`)
# because French capital amounts use '.' as a thousands separator ("578.450 euros")
# right in the gap this pattern spans.
PAT_SOLE_HOLDER_TRANSITION = re.compile(
    r"[Ll]a société\s+(?P<name>[A-ZÀ-Ü][A-ZÀ-Üa-zà-ÿ]{1,30})\s*,\s*société par actions simplifiée"
    r".{0,200}?num[ée]ro\s+(?P<siren>[\d\s]{9,15}).{0,80}?[Aa]ssoci[ée]e?\s+[Uu]nique"
)

# A capital increase's new shares are often reserved to named subscribers in a
# following resolution: "réserver la souscription des N actions nouvelles au profit
# de : Monsieur X Pour 330 actions [bio...] Monsieur Y Pour 150 actions [bio...]".
# This is a *delta* (new shares subscribed), not an absolute post-state — different
# from PAT_REPARTITION_TRIGGER's "répartition" tables — so it is kept as its own
# pattern/candidate shape and consumed as an allocation, not a snapshot.
PAT_SOUSCRIPTION_TRIGGER = re.compile(
    r"(?:réserver la souscription des[^.]{0,60}?au profit de|souscription[^.]{0,40}?réservée au profit de)\s*:?",
    re.IGNORECASE,
)

SUBSCRIBER_LINE_RE = re.compile(
    # Civility is required (not optional, unlike HOLDER_LINE_RE) so the match anchors
    # exactly at "Monsieur"/"Madame" rather than greedily swallowing a capitalized
    # word from the preceding sentence ("... de nationalité Française Monsieur X").
    r"(?:Monsieur|Madame|Mademoiselle|Société|SAS|SARL|SA)\s+"
    r"(?P<name>[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’\-\.]*(?:\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’\-\.]*){0,4})\s+"
    r"[Pp]our\s+(?P<shares>\d[\d\s]{0,6}\d|\d)\s+actions"
)

PAT_REPARTITION_TRIGGER = re.compile(
    r"(?:suivant la répartition suivante|nouvelle répartition suivante entre les associés)\s*:?",
    re.IGNORECASE,
)

HOLDER_LINE_RE = re.compile(
    r"(?P<civ>Monsieur|Madame|Mademoiselle|Société|SAS|SARL|SA)?\s*"
    r"(?P<name>[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’\-\.]*(?:\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’\-\.]*){0,4})\s+"
    r"(?P<shares>\d[\d\s]{0,6}\d|\d)\s+actions"
    r"(?:\s+(?P<pct>\d{1,3}[.,]\d{1,2})\s*%)?"
)

# A founding-by-contribution-in-kind (a holding company constituted by apport en
# nature of another company's shares, e.g. HADEAN) has no single "répartition"
# table — each founder's allocation is its own numbered sub-article ("6.2- Apport de
# 742 actions ... par Monsieur X", "6.3- Apport ... par Monsieur Y"), each ending in
# its own sentence: "Monsieur X se voit attribuer N actions d'apports d'un montant
# de P € chacune." Every such sentence on a page is collected into one genesis
# snapshot for that page, same shape as PAT_REPARTITION_TRIGGER's table.
PAT_APPORT_ATTRIBUTION = re.compile(
    r"(?:Monsieur|Madame|Mademoiselle)\s+"
    r"(?P<name>[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’\-\.]*(?:\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’\-\.]*){0,4})\s+"
    r"se voit attribuer\s+(?P<shares>\d[\d\s]{0,6}\d|\d)\s+actions",
    re.IGNORECASE,
)

CIVILITY_PREFIX_RE = re.compile(
    r"^(?:Messieurs|Mesdames|Mesdemoiselles|Monsieur|Madame|Mademoiselle)\s+", re.IGNORECASE
)


def _clean_name(raw: str) -> str:
    return CIVILITY_PREFIX_RE.sub("", raw.strip()).strip()

PAT_CESSION_TOTALITE = re.compile(
    r"cession de la totalité des actions détenues par\s+(?P<names>.{0,200}?),\s*associés?",
    re.IGNORECASE,
)

PAT_RACHAT_PRICE = re.compile(
    r"rachat(?:\s+de)?\s+(?P<n>" + AMOUNT_RE + r")\s+actions[^.]{0,60}?"
    r"prix de\s+(?P<price>\d+[.,]\d{1,2})\s*euros? par action",
    re.IGNORECASE,
)


def _amt(raw: str | None) -> float | None:
    return parse_fr_amount(raw) if raw else None


def extract_page_candidates(doc_id: str, page: dict) -> list[dict]:
    """Run every rule pattern against one reconstructed page; return raw candidates."""
    flat, spans = flatten_page(page)
    if not flat.strip():
        return []
    out: list[dict] = []

    def add(match: re.Match, event_code: str, payload: dict, pattern: str):
        start, end = match.span()
        bbox = bbox_for_char_range(page, spans, start, end)
        date = nearest_date(flat, start)
        out.append({
            "doc_id": doc_id,
            "page": page["page"],
            "pattern": pattern,
            "event_code": event_code,
            "payload": payload,
            "event_date": date,
            "source_kind": classify_source_kind(flat, start),
            "snippet": flat[max(0, start - 20) : end + 20].strip(),
            "bbox_norm": bbox,
        })

    for m in PAT_RECAP_INCREASE.finditer(flat):
        add(m, "CAPITAL_INCREASE", {
            "amount_eur": _amt(m.group("amount")),
            "capital_after_eur": _amt(m.group("after")),
            "method": None,
        }, "recap_increase")

    for m in PAT_RECAP_DECREASE.finditer(flat):
        add(m, "CAPITAL_DECREASE", {
            "amount_eur": _amt(m.group("amount")),
            "capital_after_eur": _amt(m.group("after")),
            "capital_before_eur": _amt(m.group("before")) if m.group("before") else None,
            "method": None,
        }, "recap_decrease")

    for m in PAT_DECIDE_INCREASE.finditer(flat):
        add(m, "CAPITAL_INCREASE", {
            "amount_eur": _amt(m.group("amount")),
            "capital_after_eur": _amt(m.group("after")),
            "capital_before_eur": _amt(m.group("before")),
            "method": None,
        }, "decide_increase")

    for m in PAT_DIRECT_INCREASE.finditer(flat):
        after = m.group("after") if "after" in m.groupdict() else None
        add(m, "CAPITAL_INCREASE", {
            "amount_eur": _amt(m.group("amount")),
            "capital_after_eur": _amt(after) if after else None,
            "method": None,
        }, "direct_increase")

    for m in PAT_NOMINAL_SPLIT.finditer(flat):
        # A par-value split multiplies share count and divides nominal value with
        # total capital unchanged — it is not a CAPITAL_INCREASE/DECREASE in the
        # scored sense. Kept as context only (Phase 4 cross-check), never emitted as
        # its own scored event.
        add(m, "CAPITAL_NOMINAL_SPLIT", {
            "factor": int(m.group("factor")),
        }, "nominal_split")

    for m in PAT_STATUTE_SNAPSHOT.finditer(flat):
        add(m, "CAPITAL_SNAPSHOT", {  # informational only — used for cross-checking, not scored
            "capital_after_eur": _amt(m.group("amount")),
        }, "statute_snapshot")

    for m in PAT_DUAL_CLASS.finditer(flat):
        add(m, "CAPITAL_DUAL_CLASS", {
            "class_name": m.group("codes"),
            "description": flat[m.start(): m.end() + 60].strip(),
        }, "dual_class")

    for m in PAT_RACHAT_PRICE.finditer(flat):
        add(m, "CAPITAL_DECREASE_CONTEXT", {  # merged into a decrease event downstream
            "shares": _amt(m.group("n")),
            "price_eur": _amt(m.group("price").replace(",", ".")),
        }, "rachat_price")

    for m in PAT_CESSION_TOTALITE.finditer(flat):
        names_blob = m.group("names")
        names = re.split(r",| et ", names_blob)
        names = [_clean_name(n) for n in names if n.strip() and len(n.strip()) > 2]
        for name in names:
            add(m, "SHAREHOLDER_END", {"holder_name": name}, "cession_totalite")

    for trig in PAT_REPARTITION_TRIGGER.finditer(flat):
        window = flat[trig.end(): trig.end() + 600]
        holders = []
        for hm in HOLDER_LINE_RE.finditer(window):
            holders.append({
                "name": hm.group("name").strip(),
                "shares": _amt(hm.group("shares")),
                "pct": float(hm.group("pct").replace(",", ".")) if hm.group("pct") else None,
            })
        if holders:
            start = trig.end()
            end = trig.end() + max(hm.end() for hm in HOLDER_LINE_RE.finditer(window))
            bbox = bbox_for_char_range(page, spans, start, end)
            out.append({
                "doc_id": doc_id,
                "page": page["page"],
                "pattern": "repartition_table",
                "event_code": "CAP_TABLE_SNAPSHOT",  # informational; not a scored event code
                "payload": {"holders": holders},
                # Not nearest_date(): these multi-person blocks often embed unrelated
                # dates (birth dates, certificate dates) right next to a share count —
                # the doc-level PV header (backfilled in extract_document_candidates)
                # is the trustworthy source for when a holder *table* applies.
                "event_date": None,
                "source_kind": classify_source_kind(flat, start),
                "snippet": flat[start:end].strip(),
                "bbox_norm": bbox,
            })

    apport_matches = list(PAT_APPORT_ATTRIBUTION.finditer(flat))
    if apport_matches:
        holders = [
            {"name": _clean_name(m.group("name")), "shares": _amt(m.group("shares")), "pct": None}
            for m in apport_matches
        ]
        start, end = apport_matches[0].start(), apport_matches[-1].end()
        bbox = bbox_for_char_range(page, spans, start, end)
        out.append({
            "doc_id": doc_id,
            "page": page["page"],
            "pattern": "apport_attribution_table",
            "event_code": "CAP_TABLE_SNAPSHOT",  # informational; not a scored event code
            "payload": {"holders": holders},
            "event_date": None,  # doc-level PV header / dateDepot fallback, not local search
            "source_kind": classify_source_kind(flat, start),
            "snippet": flat[start:end].strip(),
            "bbox_norm": bbox,
        })

    for trig in PAT_SOUSCRIPTION_TRIGGER.finditer(flat):
        window = flat[trig.end(): trig.end() + 900]
        allocations = []
        for hm in SUBSCRIBER_LINE_RE.finditer(window):
            allocations.append({
                "name": _clean_name(hm.group("name")),
                "new_shares": _amt(hm.group("shares")),
            })
        if allocations:
            start = trig.end()
            end = trig.end() + max(hm.end() for hm in SUBSCRIBER_LINE_RE.finditer(window))
            bbox = bbox_for_char_range(page, spans, start, end)
            out.append({
                "doc_id": doc_id,
                "page": page["page"],
                "pattern": "subscription_table",
                "event_code": "CAP_TABLE_ALLOCATION",  # informational; not a scored event code
                "payload": {"allocations": allocations},
                "event_date": None,  # see repartition_table above: doc-level PV header, not local search
                "source_kind": classify_source_kind(flat, start),
                "snippet": flat[start:end].strip(),
                "bbox_norm": bbox,
            })

    for trig in PAT_BUYBACK_LIST_TRIGGER.finditer(flat):
        window = flat[trig.end(): trig.end() + 500]
        bought_back = []
        for hm in BUYBACK_LINE_RE.finditer(window):
            bought_back.append({
                "name": _clean_name(hm.group("name")),
                "shares": _amt(hm.group("shares")),
            })
        if bought_back:
            start = trig.end()
            end = trig.end() + max(hm.end() for hm in BUYBACK_LINE_RE.finditer(window))
            bbox = bbox_for_char_range(page, spans, start, end)
            out.append({
                "doc_id": doc_id,
                "page": page["page"],
                "pattern": "buyback_list",
                "event_code": "CAP_TABLE_BUYBACK_LIST",  # informational; not a scored event code
                "payload": {"holders": bought_back},
                "event_date": None,  # doc-level PV header / dateDepot fallback, not local search
                "source_kind": classify_source_kind(flat, start),
                "snippet": flat[start:end].strip(),
                "bbox_norm": bbox,
            })

    for m in PAT_SOLE_HOLDER_TRANSITION.finditer(flat):
        add(m, "OWNERSHIP_TRANSITION_TO_SOLE", {
            "new_sole_holder_name": m.group("name").strip(),
            "new_sole_holder_siren": re.sub(r"\s+", "", m.group("siren")),
        }, "sole_holder_transition")

    return out


def extract_document_candidates(doc: dict) -> list[dict]:
    candidates = []
    for page in doc["pages"]:
        candidates.extend(extract_page_candidates(doc["doc_id"], page))

    headers = find_pv_headers(doc)
    for c in candidates:
        if c["event_date"] is None and headers:
            c["event_date"] = date_in_force_at(headers, c["page"])
            c["date_confidence"] = "inferred_from_pv_header"
        else:
            c["date_confidence"] = "local" if c["event_date"] else "missing"

    return candidates
