"""Regex extraction of corporate-ownership signals — the group bonus. Every pattern
here was validated against real reconstructed text from this corpus (not drafted from
assumption) before being trusted — several early drafts turned out wrong against real
text and were rewritten or dropped; see the inline notes on each pattern below.

Confirmed structures in this corpus (see PLAN.md):
- the DGFiP/Cerfa 2033-F/G and 2059-F/G tax-form template (byte-identical field
  sequence across companies and fiscal years — "Forme juridique / Dénomination /
  N° SIREN (si société établie en France) / % de détention"),
- a cash-subscription resolution shape ("au profit de la société X, RCS <ville>
  n° <siren>... l'émission de N actions nouvelles") — this is how HADEAN's 2008 AGM
  recorded AIR SYSTEM SERVICE's 1.836-share (7.93%) subscription; the beneficiary's
  own SIREN is given inline, so this is a high-confidence source,
- the "La société X, ..., Associée Unique de la société Y" sole-holder opener
  (validated on ARCHEAN's real 2018 PV) — recall is partial, since real phrasing
  varies more than this (guillemets, "représentant légal de la société «X»,
  associée unique de la Société «Y»" in other companies' docs); anything this misses
  is still caught by find_relevant_pages.py's marker for the same phrase, for LLM
  fallback,
- a "Liste des filiales et participations" row (legal form + name + postal code +
  city, followed by a run of numbers: capital / capitaux propres / quote-part % /
  résultat) — validated against BERNACHON's real annex. The quote-part % is NOT
  reliably the same column position across filings (no literal "%" per row, unlike
  the section header), so FILIALE_ROW_RE only locates candidate rows; picking out
  which number is the percentage is left to the caller/LLM.
- AGM attendance sheets (feuille de présence) do NOT have a reliable per-row regex
  shape at all — real HADEAN data mixes name/address/representative/share-count into
  one ragged block with no consistent delimiter. A drafted ATTENDANCE_ROW_RE failed
  against real text and was dropped; find_relevant_pages.py's "feuille de présence"
  marker is the only signal here — these pages always go to LLM fallback.

Regex here runs on the FLATTENED text of a single page (reuse ocr_reconstruction's
reconstruct_page) after find_relevant_pages.py has already narrowed which pages are
worth reconstructing at all — this module never needs to see a whole document.
"""

from __future__ import annotations

import re

# --- Cerfa 2033-F/G / 2059-F/G shareholder/subsidiary row ------------------------
# Denomination is matched as an ALL-CAPS run (how these forms always render a
# company name) specifically so a stray lowercase OCR-noise line between fields
# (a real artifact seen in this corpus — a single garbled "e" line between
# "Dénomination ARCHEAN TECHNOLOGIES" and "N° SIREN") can't be swallowed into it;
# the flexible `.{0,20}?` gap after it absorbs exactly that kind of noise instead.
CERFA_ROW_RE = re.compile(
    r"Forme juridique\s+(?P<forme>SA|SAS|SARL|SASU|EURL|SCI|SNC)?\s*"
    r"D[ée]nomination\s+(?P<denomination>[A-ZÀ-Ü][A-ZÀ-Ü0-9'&.\- ]{1,58})"
    r"\s*.{0,20}?N°\s*SIREN\s*\(si soci[ée]t[ée] [ée]tablie en France\)\s*"
    r"(?P<siren>[\d ]{9,15})?\s*%\s*de d[ée]tention\s+(?P<pct>[\d]{1,3}[.,]\d{2,4})",
    re.IGNORECASE | re.DOTALL,
)

# A section is empty when its own "Néant" checkbox is marked — appears right after
# the section header, before any row. Two full real examples in this corpus
# (024057572, 504304205) have this; skip the page as a confirmed-empty result rather
# than attempt a row match that would find nothing anyway.
NEANT_RE = re.compile(r"\bN[ée]ant\b", re.IGNORECASE)

# "EXERCICE CLOS LE" gives the bilan's own as_of anchor — OCR often mangles the
# slashes into stray punctuation ("31.12.18", ".311218"), so match digit runs loosely
# rather than requiring literal slashes.
EXERCICE_CLOS_RE = re.compile(
    r"EXERCICE CLOS LE\D{0,5}(?P<d>\d{1,2})\D{0,3}(?P<m>\d{1,2})\D{0,3}(?P<y>\d{2,4})",
    re.IGNORECASE,
)

# --- cash-subscription resolution: "au profit de la société X, RCS <ville>
# n° <siren>, pour permettre l'émission de N actions nouvelles..." — validated
# against HADEAN's real 2008-04-30 PV recording AIR SYSTEM SERVICE's subscription
# (1.836 shares / 100.000€, out of 23.138 total = 7.93%). High confidence: gives the
# beneficiary's own SIREN inline, no name-resolution guesswork needed.
BENEFICIARY_COMPANY_RE = re.compile(
    r"au profit de la soci[ée]t[ée]\s+(?P<name>[A-ZÀ-Ü][A-ZÀ-Ü0-9'&.\- ]{1,50}?)\s*,\s*"
    r"RCS\s+[A-ZÀ-Üa-zà-ÿ\-]+\s*n[°ºo]\s*(?P<siren>[\d ]{9,15}).{0,200}?"
    r"[ée]mission de\s+(?P<shares>[\d][\d .]{0,10})\s*actions? nouvelles",
    re.IGNORECASE | re.DOTALL,
)

# Companion pattern for the same resolution's follow-up decision — "...attribuées à
# la société X en rémunération de son apport" — same real document, no SIREN given
# here (that's why BENEFICIARY_COMPANY_RE above is preferred when both are present),
# useful on its own for apport-en-nature cases where a company (not a person) is the
# beneficiary.
ATTRIBUTED_COMPANY_RE = re.compile(
    r"attribu[ée]es?\s+à\s+la\s+soci[ée]t[ée]\s+(?P<name>[A-ZÀ-Ü][A-ZÀ-Ü0-9'&.\- ]{1,50}?)\s+"
    r"en\s+r[ée]mun[ée]ration\s+de\s+son\s+apport",
    re.IGNORECASE,
)

# "La société X, ..., Associée Unique de la société Y" — validated against ARCHEAN's
# real 2018 PV (HADEAN as Associée Unique of ARCHEAN TECHNOLOGIES). Keyword parts are
# wrapped in inline (?i:...) rather than a blanket re.IGNORECASE flag, deliberately:
# IGNORECASE would let the [A-ZÀ-Ü] name classes match lowercase too, and an early
# draft of this pattern silently swallowed trailing lowercase prose ("ARCHEAN
# TECHNOLOGIES a pris les décisions...") into the captured name before this fix.
# Guillemets are optional since some companies' filings quote the names («X»).
SOLE_HOLDER_RE = re.compile(
    r"(?i:la soci[ée]t[ée])\s+«?(?P<owner>[A-ZÀ-Ü][A-ZÀ-Ü0-9'&.\- ]{1,40}?)»?\s*,"
    r".{0,200}?(?i:associ[ée]e?\s+unique\s+de\s+la\s+(?:soci[ée]t[ée]\s+)?)"
    r"«?(?P<owned>[A-ZÀ-Ü][A-ZÀ-Ü0-9'&.\- ]{1,40}?)»?(?=\s+(?:a\s|à\s|,|\.|\n)|$)",
    re.DOTALL,
)

# "Liste des filiales et participations" annex row (non-Cerfa, accounting-software
# format — validated against BERNACHON's real annex): legal form + name + postal
# code + city, followed by a run of numbers (capital / capitaux propres / quote-part
# % / résultat, in no fixed column order across filings, and with no literal "%"
# character per row — only the section header has one). This pattern therefore only
# locates candidate rows; picking the quote-part number out of `nums` is left to the
# caller (or the LLM, given the row as compact context) rather than guessed here.
FILIALE_ROW_RE = re.compile(
    r"(?P<forme>EURL|SARL|SASU|SAS|SCI|SNC|SA)\s+"
    r"(?P<name>[A-ZÀ-Ü][A-ZÀ-Ü0-9'&.\- ]{1,40}?)\s+"
    r"(?P<postal>\d{4,5})\s+(?P<city>[A-ZÀ-Ü][A-ZÀ-Üa-zà-ÿ\-]+)\s+"
    r"(?P<nums>[\d ,.\-]+)"
)


def parse_siren(raw: str | None) -> str | None:
    if not raw:
        return None
    digits = re.sub(r"\D", "", raw)
    return digits if len(digits) == 9 else None


def parse_exercice_date(text: str) -> str | None:
    m = EXERCICE_CLOS_RE.search(text)
    if not m:
        return None
    d, mo, y = int(m.group("d")), int(m.group("m")), int(m.group("y"))
    y = y + 2000 if y < 100 else y
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return None
    return f"{y:04d}-{mo:02d}-{d:02d}"


def extract_cerfa_rows(text: str) -> list[dict]:
    """All Cerfa-form shareholder/subsidiary rows on a page's flattened text."""
    return [
        {
            "denomination": " ".join(m.group("denomination").split()),
            "siren": parse_siren(m.group("siren")),
            "pct": float(m.group("pct").replace(",", ".")),
            "forme": m.group("forme"),
        }
        for m in CERFA_ROW_RE.finditer(text)
    ]
