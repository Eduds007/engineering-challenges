"""Locate pages worth reconstructing/sending to the group-extraction pipeline, by
scanning raw OCR text directly — no reading-order reconstruction needed for a keyword
match, since word order within a page doesn't matter for detecting a marker phrase.

This is the single biggest token-cost lever for the group bonus: a bilan can run to
hundreds of pages of balance-sheet/P&L detail, and — confirmed by direct inspection
of this corpus (see PLAN.md) — only 1-3 pages per document ever carry an ownership
form (the Cerfa 2033-F/G / 2059-F/G "filiales et participations" / "composition du
capital social" annexes) or an AGM attendance sheet. Everything else in a bilan is
irrelevant to the group graph and never needs reconstructing, let alone an LLM call.
"""

from __future__ import annotations

import glob
import json
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")

MARKERS: list[re.Pattern] = [
    re.compile(r"FILIALES ET PARTICIPATIONS", re.IGNORECASE),
    re.compile(r"COMPOSITION DU CAPITAL SOCIAL", re.IGNORECASE),
    re.compile(r"CAPITAL D[ÉE]TENU P[AÁ]R LES PERSONNES", re.IGNORECASE),
    re.compile(r"NOMBRE TOTAL DE FILIALES", re.IGNORECASE),
    re.compile(r"feuille de pr[ée]sence", re.IGNORECASE),
    re.compile(r"N°\s*SIREN\s*\(si soci[ée]t[ée]", re.IGNORECASE),
    re.compile(r"LA SOUSSIGN[ÉE]E", re.IGNORECASE),
    re.compile(r"apporte\s+à\s+la\s+Soci[ée]t[ée]", re.IGNORECASE),
    re.compile(r"[Aa]ssoci[ée]e?\s+[Uu]nique\s+de\s+la", re.IGNORECASE),
    re.compile(r"soci[ée]t[ée]\s+m[èe]re", re.IGNORECASE),
    # "au profit de la société X, RCS <ville> n° <siren>..." / "...attribuées à la
    # société X en rémunération de son apport" — a real cash/apport-en-numéraire
    # capital-increase resolution shape (confirmed on HADEAN's 2008 AGM recording
    # AIR SYSTEM SERVICE's subscription); none of the markers above matched this
    # page until this was added, so extract_group.py silently never looked at it.
    re.compile(r"au\s+profit\s+de\s+la\s+soci[ée]t[ée]", re.IGNORECASE),
    re.compile(r"attribu[ée]es?\s+à\s+la\s+soci[ée]t[ée]", re.IGNORECASE),
]


def page_raw_text(page_json_path: str) -> str:
    """Concatenate a page's OCR line texts in array order — fine for a keyword scan,
    unlike event extraction this doesn't need visual reading order."""
    data = json.load(open(page_json_path, encoding="utf-8"))
    return " ".join((line.get("text") or "") for line in (data.get("ocr") or []))


def find_relevant_pages(siren: str, kind: str, doc_id: str) -> list[dict]:
    """Return [{'page': N, 'matched': [marker pattern strings]}] for a single document."""
    ocr_dir = os.path.join(DATA_DIR, siren, kind, "ocr", doc_id)
    if not os.path.isdir(ocr_dir):
        return []
    hits = []
    for page_path in sorted(glob.glob(os.path.join(ocr_dir, "page_*.json"))):
        page_num = int(os.path.basename(page_path)[5:8])
        text = page_raw_text(page_path)
        matched = [m.pattern for m in MARKERS if m.search(text)]
        if matched:
            hits.append({"page": page_num, "matched": matched})
    return hits


def find_relevant_pages_for_company(siren: str) -> dict:
    """Scan every acte and bilan document for this SIREN. Returns
    {doc_id: {"kind": "actes"|"bilans", "pages": [...]}} for documents with >=1 hit."""
    results: dict[str, dict] = {}
    for kind in ("actes", "bilans"):
        base = os.path.join(DATA_DIR, siren, kind, "ocr")
        if not os.path.isdir(base):
            continue
        for doc_id in sorted(os.listdir(base)):
            hits = find_relevant_pages(siren, kind, doc_id)
            if hits:
                results[doc_id] = {"kind": kind, "pages": hits}
    return results


if __name__ == "__main__":
    import sys

    siren = sys.argv[1] if len(sys.argv) > 1 else "480489707"
    result = find_relevant_pages_for_company(siren)
    total_pages = sum(len(v["pages"]) for v in result.values())
    print(f"{siren}: {len(result)} documents with hits, {total_pages} relevant pages total",
          file=sys.stderr)
    print(json.dumps(result, indent=2, ensure_ascii=False))
