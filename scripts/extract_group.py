"""Group-bonus orchestrator: locate candidate pages -> regex extraction -> entity
resolution -> group.nodes/group.edges, written to output/480489707/group.json.

Deterministic pass (default, no API calls): find_relevant_pages_for_company() finds
candidate pages cheaply from raw OCR text, group_patterns.py's validated regexes pull
structured rows out of the ones worth reconstructing, and every name found is resolved
against a denomination index built directly from this corpus's own document metadata
(data/<siren>/{actes,bilans}/meta/*.json — no hardcoded SIREN/name table).

LLM fallback pass (--llm, sequential only, never run while another OpenRouter-calling
phase is in flight): for pages find_relevant_pages_for_company() flagged that no
regex above produced a structured hit on (loose free-text "filiale"/"détient" mentions,
attendance-sheet rows — validated against real text as NOT having a reliable regex
shape), send just that page's text with a compact running summary of nodes/edges found
so far, mirroring extract_events.py's context-graph design.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from find_relevant_pages import find_relevant_pages_for_company, DATA_DIR
from ocr_reconstruction import find_doc_paths, load_page_ocr, reconstruct_page
from grounding import find_snippet_bbox
from llm_client import call_llm, extract_json
import group_patterns as gp

REPO_ROOT = os.path.dirname(DATA_DIR)
ALL_SIRENS = [
    "015551401", "016850919", "024052656", "024057572", "026980508", "027080076",
    "035550318", "328024377", "352890354", "360500011", "401009741", "412000887",
    "445070311", "480489707", "499979540", "504304205", "560801045", "820561470",
    "843071218", "846650141",
]

_LEGAL_FORM_RE = re.compile(
    r"^\s*(?:SA|SAS|SASU|SARL|EURL|SCI|SNC)\s+|\s+(?:SA|SAS|SASU|SARL|EURL|SCI|SNC)\.?\s*$",
)
_JUNK_RE = re.compile(r"[.\-'&]")


def _normalize(name: str) -> str:
    name = _LEGAL_FORM_RE.sub(" ", name.upper())
    name = _JUNK_RE.sub(" ", name)
    return " ".join(name.split())


def build_denomination_index() -> dict[str, list[str]]:
    """{siren: [denomination variants, most-representative first]} straight from this
    corpus's own filing metadata — no guessed or hardcoded names. Variants are ranked
    by how often they occur (ties broken by length) so a one-off typo in the corpus
    itself (e.g. HADEAN's own metadata misspells it "ADEAN" in a single filing) can't
    become the canonical name just because it happens to sort first alphabetically."""
    counts: dict[str, dict[str, int]] = {}
    for siren in sorted(os.listdir(DATA_DIR)):
        for kind in ("actes", "bilans"):
            meta_dir = os.path.join(DATA_DIR, siren, kind, "meta")
            if not os.path.isdir(meta_dir):
                continue
            for f in glob.glob(os.path.join(meta_dir, "*.json")):
                try:
                    d = json.load(open(f, encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                den = (d.get("denomination") or "").strip()
                if den and den.upper() != "NON RENSEIGNE":
                    counts.setdefault(siren, {}).setdefault(den, 0)
                    counts[siren][den] += 1
    return {
        siren: sorted(denoms, key=lambda d: (-denoms[d], -len(d)))
        for siren, denoms in counts.items()
    }


def resolve_entity(
    name: str, denom_index: dict[str, list[str]], threshold: int = 82
) -> dict | None:
    """Fuzzy-match `name` against every known (siren, denomination) pair. Returns
    {'siren', 'denomination', 'score'} for the best match at or above `threshold`,
    or None — callers must accept None and emit resolved:false rather than guess."""
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return None
    target = _normalize(name)
    if not target:
        return None
    best = None
    for siren, denoms in denom_index.items():
        for den in denoms:
            score = fuzz.token_sort_ratio(target, _normalize(den))
            if best is None or score > best["score"]:
                best = {"siren": siren, "denomination": den, "score": score}
    if best and best["score"] >= threshold:
        return best
    return None


def _page_dict(siren: str, kind: str, doc_id: str, page_num: int) -> dict | None:
    try:
        pdf_path, ocr_dir = find_doc_paths(siren, doc_id, kind)
    except FileNotFoundError:
        return None
    if not os.path.exists(pdf_path):
        return None
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    doc = fitz.open(pdf_path)
    if not (1 <= page_num <= doc.page_count):
        return None
    lines = load_page_ocr(ocr_dir, page_num)
    page = doc[page_num - 1]
    return reconstruct_page(lines, page.rect.width, page.rect.height)


def _doc_meta(siren: str, kind: str, doc_id: str) -> dict:
    meta_path = os.path.join(DATA_DIR, siren, kind, "meta", f"*{doc_id}.json")
    matches = glob.glob(meta_path)
    if not matches:
        return {}
    try:
        return json.load(open(matches[0], encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


_SECTION_MARKERS = [
    # "P[AÁ]R": OCR misreads PAR as PÁR on at least one real filing (ARCHEAN LABS'
    # 2022-01-18 bilan) — confirmed against real text, not a hypothetical.
    (re.compile(r"CAPITAL D[ÉE]TENU P[AÁ]R LES PERSONNES MORALES", re.IGNORECASE), "held_by"),
    (re.compile(r"FILIALES ET PARTICIPATIONS", re.IGNORECASE), "holds"),
    (re.compile(r"COMPOSITION DU CAPITAL SOCIAL", re.IGNORECASE), "holds"),
]


def _nearest_section(text: str, pos: int) -> str:
    """Which Cerfa section a row at `pos` falls under, by nearest preceding header:
    'holds' (subject owns the listed company — the default) or 'held_by' (the listed
    company owns the subject). Defaults to 'holds' if no header precedes the row at
    all (shouldn't happen on a real form, but a missing header is not a reason to
    guess the rarer, direction-flipping case)."""
    best_pos, best_kind = -1, "holds"
    for pattern, kind in _SECTION_MARKERS:
        for hm in pattern.finditer(text[:pos]):
            if hm.start() > best_pos:
                best_pos, best_kind = hm.start(), kind
    return best_kind


def _make_source(doc_id: str, page_num: int, page: dict, snippet: str) -> dict:
    bbox, method = find_snippet_bbox(page, snippet)
    return {
        "inpi_id": doc_id,
        "page": page_num,
        "bbox": bbox,
        "snippet": snippet[:280],
        "grounding_method": method,
    }


def extract_candidates_from_page(
    siren: str, kind: str, doc_id: str, page_num: int, page: dict, doc_meta: dict
) -> list[dict]:
    """Run every validated regex from group_patterns.py against one page's flattened
    text; return raw candidate dicts (not yet entity-resolved)."""
    text = page["text"]
    as_of = doc_meta.get("dateDepot") or gp.parse_exercice_date(text)
    candidates: list[dict] = []

    subject_denom_norm = _normalize(doc_meta.get("denomination") or "")

    for m in gp.CERFA_ROW_RE.finditer(text):
        denomination = " ".join(m.group("denomination").split())
        # Cerfa pages carry the filer's own "identité de l'entreprise" block in the
        # same field sequence as an actual filiale/participation row — this is the
        # only reliable way to tell them apart (real bug found and fixed against
        # ARCHEAN's own bilan, which was otherwise producing a self-loop edge).
        if subject_denom_norm and _normalize(denomination) == subject_denom_norm:
            continue
        siren_raw = m.group("siren")
        pct = m.group("pct")
        # The Cerfa row's field template ("Forme juridique / Dénomination / SIREN /
        # % de détention") is IDENTICAL whether it sits under "FILIALES ET
        # PARTICIPATIONS" (subject holds % of the listed company) or under "CAPITAL
        # DÉTENU PAR LES PERSONNES MORALES" (the listed company holds % of the
        # subject) — the direction flips depending only on which section header
        # precedes it. Real bug found against ARCHEAN LABS' own bilan, which was
        # otherwise producing a backwards "ARCHEAN LABS holds 100% of ARCHEAN
        # TECHNOLOGIES" edge (the true relationship is the reverse).
        held_by_section = _nearest_section(text, m.start()) == "held_by"
        candidates.append({
            "kind": "cerfa_row", "subject_siren": siren, "other_name": denomination,
            "other_siren_hint": siren_raw, "pct": float(pct.replace(",", ".")) if pct else None,
            "as_of": as_of, "reversed": held_by_section,
            "source": _make_source(doc_id, page_num, page, m.group(0)),
        })

    m = gp.BENEFICIARY_COMPANY_RE.search(text)
    if m:
        shares = int(re.sub(r"\D", "", m.group("shares")))
        candidates.append({
            "kind": "subscription", "subject_siren": siren, "other_name": m.group("name"),
            "other_siren_hint": re.sub(r"\D", "", m.group("siren")), "shares": shares,
            "as_of": as_of, "source": _make_source(doc_id, page_num, page, m.group(0)),
        })

    m = gp.ATTRIBUTED_COMPANY_RE.search(text)
    if m and not any(c["kind"] == "subscription" for c in candidates):
        candidates.append({
            "kind": "attribution", "subject_siren": siren, "other_name": m.group("name"),
            "other_siren_hint": None, "as_of": as_of,
            "source": _make_source(doc_id, page_num, page, m.group(0)),
        })

    m = gp.SOLE_HOLDER_RE.search(text)
    if m:
        candidates.append({
            "kind": "sole_holder", "subject_siren": siren, "owner_name": m.group("owner"),
            "owned_name": m.group("owned"), "pct": 100.0, "as_of": as_of,
            "source": _make_source(doc_id, page_num, page, m.group(0)),
        })

    for fm in gp.FILIALE_ROW_RE.finditer(text):
        pct_candidates = re.findall(r"\d{1,3}[.,]\d{2}\b", fm.group("nums"))
        candidates.append({
            "kind": "filiale_row", "subject_siren": siren,
            "other_name": f"{fm.group('forme')} {fm.group('name')}".strip(),
            "other_siren_hint": None,
            "pct": float(pct_candidates[0].replace(",", ".")) if pct_candidates else None,
            "as_of": as_of, "source": _make_source(doc_id, page_num, page, fm.group(0)),
        })

    return candidates


def extract_group_for_company(siren: str, denom_index: dict[str, list[str]]) -> tuple[list[dict], list[dict]]:
    """Deterministic-only pass for one company. Returns (candidates, unmatched_pages)
    — unmatched_pages are find_relevant_pages hits with zero regex candidates, i.e.
    exactly the pages an --llm pass would need to look at."""
    hits = find_relevant_pages_for_company(siren)
    all_candidates: list[dict] = []
    unmatched: list[dict] = []
    for doc_id, info in hits.items():
        kind = info["kind"]
        doc_meta = _doc_meta(siren, kind, doc_id)
        for hit in info["pages"]:
            page_num = hit["page"]
            page = _page_dict(siren, kind, doc_id, page_num)
            if page is None:
                continue
            cands = extract_candidates_from_page(siren, kind, doc_id, page_num, page, doc_meta)
            if cands:
                all_candidates.extend(cands)
            else:
                unmatched.append({
                    "siren": siren, "kind": kind, "doc_id": doc_id, "page": page_num,
                    "matched_markers": hit["matched"],
                })
    return all_candidates, unmatched


def resolve_and_build(
    candidates: list[dict], denom_index: dict[str, list[str]]
) -> tuple[list[dict], list[dict]]:
    """Turn raw regex candidates into schema-shaped nodes/edges, resolving every
    name against the corpus's own denomination index."""
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def add_node(name: str, siren_hint: str | None, exclude_siren: str | None) -> str:
        resolved = None
        if siren_hint and siren_hint.strip() and len(re.sub(r"\D", "", siren_hint)) == 9:
            candidate_siren = re.sub(r"\D", "", siren_hint)
            # A row's own SIREN hint equal to the filer's own SIREN is a red flag, not
            # a resolution — real bug found on ARCHEAN's own bilan, where a row named
            # "ARCHEAN MOTION" carried SIREN 480489707 (ARCHEAN TECHNOLOGIES' own),
            # which without this guard silently merged it into the parent and produced
            # a self-loop edge. Falling through to fuzzy name resolution below is
            # correct here: "ARCHEAN MOTION" isn't one of the 20 known denominations,
            # so it should end up resolved:false, not merged into the wrong node.
            if candidate_siren in denom_index and candidate_siren != exclude_siren:
                resolved = {"siren": candidate_siren, "denomination": name, "score": 100}
        if resolved is None:
            resolved = resolve_entity(name, denom_index)
            if resolved and resolved["siren"] == exclude_siren:
                resolved = None
        if resolved:
            key = resolved["siren"]
            canonical = denom_index[key][0]
            nodes.setdefault(key, {"name": canonical, "siren": key, "resolved": True})
            return canonical
        key = f"unresolved:{_normalize(name)}"
        nodes.setdefault(key, {"name": name.strip(), "siren": None, "resolved": False})
        return nodes[key]["name"]

    for c in candidates:
        subject_siren = c["subject_siren"]
        subject_denoms = denom_index.get(subject_siren)
        subject_name = subject_denoms[0] if subject_denoms else subject_siren
        nodes.setdefault(subject_siren, {
            "name": subject_name, "siren": subject_siren, "resolved": True,
        })

        if c["kind"] in ("cerfa_row", "filiale_row", "llm_row"):
            other = add_node(c["other_name"], c.get("other_siren_hint"), subject_siren)
            if other == subject_name:
                continue
            from_, to_ = (other, subject_name) if c.get("reversed") else (subject_name, other)
            edges.append({
                "from": from_, "to": to_, "relation": "shareholder_of",
                "pct": c.get("pct"), "as_of": c.get("as_of"), "source": c["source"],
            })
        elif c["kind"] == "subscription":
            other = add_node(c["other_name"], c.get("other_siren_hint"), subject_siren)
            if other == subject_name:
                continue
            edges.append({
                "from": other, "to": subject_name, "relation": "shareholder_of",
                "pct": None, "as_of": c.get("as_of"), "source": c["source"],
                "note": f"{c['shares']} shares subscribed (percentage not computed here — "
                        f"needs the company's total share count at that date)",
            })
        elif c["kind"] == "attribution":
            other = add_node(c["other_name"], None, subject_siren)
            if other == subject_name:
                continue
            edges.append({
                "from": other, "to": subject_name, "relation": "shareholder_of",
                "pct": None, "as_of": c.get("as_of"), "source": c["source"],
            })
        elif c["kind"] == "sole_holder":
            owner = add_node(c["owner_name"], None, None)
            owned = add_node(c["owned_name"], None, None)
            if owner == owned:
                continue
            edges.append({
                "from": owner, "to": owned, "relation": "shareholder_of",
                "pct": 100.0, "as_of": c.get("as_of"), "source": c["source"],
            })

    return list(nodes.values()), edges


GROUP_FALLBACK_SYSTEM = """You extract COMPANY-TO-COMPANY ownership relations from one page of a French \
corporate legal filing (bilan annex or acte) that structured regex rules already checked and found nothing \
structured in. Look for any statement that one named company holds shares in, is a shareholder of, is a \
subsidiary of, or is the parent of another named company. A person holding shares in a company does NOT count \
— only company-to-company. If several such relations appear on the page, return only the clearest, most \
confident one. Reply ONLY JSON, no prose: {"found": true|false, "other_name": "...", "other_siren": "9-digit \
string or null", "pct": number or null, "direction": "subject_holds_other" or "other_holds_subject", "as_of": \
"YYYY-MM-DD or null", "snippet": "the exact quote you based this on, verbatim from the page text"}. If there is \
truly no company-to-company relation on this page, reply {"found": false}."""


def _crop_around_markers(text: str, matched_markers: list[str], radius: int = 900) -> str:
    """Crop the page text to a compact window around the first marker that matched
    find_relevant_pages.py, instead of sending the whole (often multi-thousand-char)
    page — keeps the fallback prompt cheap without losing the relevant sentence."""
    for pattern in matched_markers:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            start = max(0, m.start() - radius)
            end = min(len(text), m.start() + radius)
            return text[start:end]
    return text[:1800]


def llm_fallback_candidates(unmatched: list[dict], denom_index: dict[str, list[str]]) -> list[dict]:
    """Sequential-only (never parallel) LLM pass over pages find_relevant_pages
    flagged but no regex pattern matched. One call per page, minimal cropped context,
    same call_llm() used by the main timeline pipeline (OpenRouter first, this
    machine's own claude-cli as last resort — see llm_client.py)."""
    candidates: list[dict] = []
    for i, entry in enumerate(unmatched):
        siren, kind, doc_id, page_num = entry["siren"], entry["kind"], entry["doc_id"], entry["page"]
        page = _page_dict(siren, kind, doc_id, page_num)
        if page is None:
            continue
        doc_meta = _doc_meta(siren, kind, doc_id)
        subject_name = (denom_index.get(siren) or [siren])[0]
        crop = _crop_around_markers(page["text"], entry["matched_markers"])
        user = f"Subject company: {subject_name} (SIREN {siren})\n\nPage text:\n{crop}"
        print(f"  [{i+1}/{len(unmatched)}] llm fallback: {siren} {doc_id} p{page_num}", file=sys.stderr)
        try:
            raw = call_llm(GROUP_FALLBACK_SYSTEM, user, max_tokens=400)
            result = extract_json(raw)
        except Exception as e:
            print(f"    call failed: {e}", file=sys.stderr)
            continue
        if not isinstance(result, dict) or not result.get("found") or not result.get("other_name"):
            continue
        as_of = result.get("as_of") or doc_meta.get("dateDepot")
        candidates.append({
            "kind": "llm_row", "subject_siren": siren, "other_name": result["other_name"],
            "other_siren_hint": result.get("other_siren"), "pct": result.get("pct"),
            "as_of": as_of, "reversed": result.get("direction") == "other_holds_subject",
            "source": _make_source(doc_id, page_num, page, result.get("snippet") or result["other_name"]),
        })
    return candidates


def _canon1(name: str) -> str:
    return _normalize(name).replace(" ", "")


def _canon2(name: str) -> str:
    return "".join(sorted(_normalize(name).split()))


def dedupe_unresolved_nodes(nodes: list[dict], edges: list[dict]) -> tuple[list[dict], list[dict]]:
    """Regex and the LLM fallback sometimes find the SAME unresolved company under
    differently-formatted names — real cases found in this corpus: "SK2R SAS" (regex)
    vs "S.K2.R." (LLM, dotted-abbreviation spacing) for the same relation on different
    dates, and "SARL BERNACHON PARIS" (regex) vs "PARIS BERNACHON" (LLM, word order
    flipped). Union-find on two exact-match canonical forms — whitespace-collapsed
    (_canon1, catches the SK2R spacing case) and word-order-independent (_canon2,
    catches the BERNACHON reversal) — merges only these; a plain fuzzy-similarity
    threshold was tried first and rejected: at any threshold loose enough to catch
    SK2R (~80), it also wrongly merged "EURL BERNACHON PASSION" with "SARL BERNACHON
    PARIS" (score 81) — two real, different subsidiaries that just share a long common
    substring. Exact-canonical-form matching has no such false positive here."""
    unresolved = [n["name"] for n in nodes if not n["resolved"]]
    parent = {n: n for n in unresolved}

    def find(x: str) -> str:
        while parent[x] != x:
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    by_c1: dict[str, str] = {}
    by_c2: dict[str, str] = {}
    for n in unresolved:
        c1, c2 = _canon1(n), _canon2(n)
        if c1 in by_c1:
            union(n, by_c1[c1])
        by_c1[c1] = n
        if c2 in by_c2:
            union(n, by_c2[c2])
        by_c2[c2] = n

    # Canonical name per cluster: prefer the one carrying a legal-form marker (SARL/
    # SAS/...) since that's consistently the more complete/correct-looking variant in
    # every case seen so far; otherwise the longest.
    clusters: dict[str, list[str]] = {}
    for n in unresolved:
        clusters.setdefault(find(n), []).append(n)
    rename: dict[str, str] = {}
    for members in clusters.values():
        if len(members) < 2:
            continue
        with_form = [m for m in members if _LEGAL_FORM_RE.search(m.upper())]
        canonical = max(with_form or members, key=len)
        for m in members:
            if m != canonical:
                rename[m] = canonical

    if not rename:
        return nodes, edges
    for e in edges:
        e["from"] = rename.get(e["from"], e["from"])
        e["to"] = rename.get(e["to"], e["to"])
    nodes = [n for n in nodes if n["name"] not in rename]
    return nodes, edges


def build_group(sirens: list[str], out_path: str, use_llm: bool = False) -> dict:
    denom_index = build_denomination_index()
    all_candidates: list[dict] = []
    all_unmatched: list[dict] = []
    for siren in sirens:
        cands, unmatched = extract_group_for_company(siren, denom_index)
        all_candidates.extend(cands)
        all_unmatched.extend(unmatched)
        print(f"{siren}: {len(cands)} regex candidates, {len(unmatched)} pages need LLM fallback",
              file=sys.stderr)

    if use_llm and all_unmatched:
        print(f"\nrunning LLM fallback over {len(all_unmatched)} pages, sequentially...", file=sys.stderr)
        all_candidates.extend(llm_fallback_candidates(all_unmatched, denom_index))

    nodes, edges = resolve_and_build(all_candidates, denom_index)
    nodes, edges = dedupe_unresolved_nodes(nodes, edges)
    group = {"nodes": nodes, "edges": edges}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(group, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    status = "already sent to LLM fallback above" if use_llm else "candidates for --llm fallback, not yet run"
    print(
        f"\n{len(nodes)} nodes, {len(edges)} edges written to {out_path}\n"
        f"{len(all_unmatched)} pages flagged by find_relevant_pages but with no regex match ({status})",
        file=sys.stderr,
    )
    unmatched_path = out_path.replace(".json", ".unmatched_pages.json")
    json.dump(all_unmatched, open(unmatched_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return group


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sirens", nargs="*", default=ALL_SIRENS)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "output", "480489707", "group.json"))
    ap.add_argument("--llm", action="store_true", help="run the sequential LLM fallback pass too")
    args = ap.parse_args()
    build_group(args.sirens, args.out, use_llm=args.llm)
