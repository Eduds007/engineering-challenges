"""Strip statute boilerplate before text reaches an LLM prompt — a token-cost lever,
not an extraction change. Every restated "statuts" document in this corpus reprints
the FULL bylaws (denomination, siège, durée, forme/transmission des actions,
commissaires aux comptes, gouvernance...) on every single capital change, because
French filings restate the whole document rather than diff it. A word-frequency scan
across all 17 ARCHEAN documents confirmed this: the same handful of Article bodies
repeat near-verbatim across most of the 17 documents and account for ~22% of total
corpus characters, and none of them has ever produced a capital/shareholder match in
this corpus — rules.py's patterns only ever fire inside ARTICLE 6/7/8 (apports,
capital social, modifications du capital) or inside a PV's numbered résolutions.

Deliberately conservative: this only ever removes an ARTICLE body whose title matches
a known-irrelevant keyword (never guesses from position or length), and it always
leaves a marker behind so a reader — human or LLM — can see something was cut rather
than silently losing structure. rules.py's own regex passes run on the ORIGINAL,
unstripped text — this stripping only shrinks what the LLM validator/fallback prompts
receive, since the LLM has never yet been the one to find a genuine event inside a
governance article.
"""

from __future__ import annotations

import re

ARTICLE_RE = re.compile(r"ARTICLE\s+\d+\s*[-–—]\s*([A-ZÀ-Üa-zà-ÿ'\s]{3,50})", re.IGNORECASE)

# Titles confirmed, by frequency analysis over this corpus's own text, to repeat
# verbatim across documents and never carry a capital/shareholder decision. Matched
# as a substring of the article's title, case-insensitively.
IRRELEVANT_ARTICLE_KEYWORDS = (
    "denomination", "dénomination", "siege social", "siège social", "duree", "durée",
    "objet social", "liberation des actions", "libération des actions",
    "forme des actions", "exercice social", "commissaires aux comptes",
    "commissaire aux comptes", "president", "président", "direction generale",
    "direction générale", "conventions", "representation", "représentation",
    "transmission des actions", "droit de vote", "dissolution", "liquidation",
    "contestations", "conventions reglementees", "conventions réglementées",
)

# Lines that repeat verbatim as page letterhead/footer — carry zero decision content
# (company name/capital/address restated on every page) and safe to drop outright.
LETTERHEAD_LINE_RE = re.compile(
    r"^\s*(?:"
    r"ARCHEAN Technologies|ARCHEAN TECHNOLOGIES|"
    r"Soci[ée]t[ée] par Actions Simplifi[ée]e|"
    r"Si[èe]ge social\s*:.*|"
    r"au capital de[\d.,\s]+(?:€|[Ee]uros)\s*$"
    r")\s*$"
)


def strip_boilerplate(text: str) -> tuple[str, dict]:
    """Return (reduced_text, stats). Article bodies whose title matches a known
    irrelevant keyword are replaced by a one-line marker; letterhead lines are
    dropped outright. Resolution/PV text and ARTICLE 6/7/8 are always kept whole."""
    lines = text.split("\n")
    kept_lines = [ln for ln in lines if not LETTERHEAD_LINE_RE.match(ln)]
    reduced = "\n".join(kept_lines)

    matches = list(ARTICLE_RE.finditer(reduced))
    out_parts = []
    cut = 0
    pos = 0
    for i, m in enumerate(matches):
        title = m.group(1).strip().lower()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(reduced)
        out_parts.append(reduced[pos:m.end()])
        body = reduced[m.end():end]
        if any(kw in title for kw in IRRELEVANT_ARTICLE_KEYWORDS):
            out_parts.append(f" [body omitted — boilerplate, {len(body)} chars]\n")
            cut += len(body)
        else:
            out_parts.append(body)
        pos = end
    out_parts.append(reduced[pos:])

    result = "".join(out_parts)
    stats = {
        "original_chars": len(text),
        "reduced_chars": len(result),
        "letterhead_lines_dropped": len(lines) - len(kept_lines),
        "article_chars_omitted": cut,
    }
    return result, stats
