"""Map a span of text (a rule match, or a snippet an LLM returned) back to a
normalized bbox, by unioning the OCR lines it overlaps.

Works off the per-page {"text", "lines": [{"text", "bbox_norm"}]} shape produced by
scripts/ocr_reconstruction.py — lines are already in visual reading order there.
"""

from __future__ import annotations


def flatten_page(page: dict) -> tuple[str, list[tuple[int, int]]]:
    """Join a page's lines with single spaces; return (flat_text, [(start,end) per line]).

    Regex matching and LLM-snippet lookup both operate on this flattened string, so a
    character span found in it maps directly back to the lines (and bboxes) that produced it.
    """
    parts = []
    spans = []
    pos = 0
    for line in page["lines"]:
        t = line["text"]
        start = pos
        end = start + len(t)
        spans.append((start, end))
        parts.append(t)
        pos = end + 1  # +1 for the joining space
    return " ".join(parts), spans


def union_bbox(boxes: list[list[float]]) -> list[float]:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)]


def bbox_for_char_range(
    page: dict, spans: list[tuple[int, int]], start: int, end: int
) -> list[float] | None:
    """Union the bbox of every line whose [start,end) span overlaps the given range."""
    hit_boxes = [
        line["bbox_norm"]
        for (s, e), line in zip(spans, page["lines"])
        if s < end and e > start
    ]
    if not hit_boxes:
        return None
    return union_bbox(hit_boxes)


def find_snippet_bbox(
    page: dict, snippet: str, fuzzy_threshold: int = 80
) -> tuple[list[float] | None, str]:
    """Locate `snippet` inside a page's reconstructed text: exact substring first,
    then a fuzzy sliding-window match (needs rapidfuzz) for OCR-noisy or paraphrased
    LLM snippets. Returns (bbox_norm or None, method) with method in
    {"exact", "fuzzy", "none"}.
    """
    flat, spans = flatten_page(page)
    snippet = snippet.strip()
    if not snippet:
        return None, "none"

    idx = flat.find(snippet)
    if idx != -1:
        return bbox_for_char_range(page, spans, idx, idx + len(snippet)), "exact"

    try:
        from rapidfuzz import fuzz
    except ImportError:
        return None, "none"

    window = max(len(snippet), 20)
    step = max(window // 4, 5)
    best_span = None
    best_score = 0
    for i in range(0, max(len(flat) - window, 0) + 1, step):
        cand = flat[i : i + window]
        score = fuzz.partial_ratio(snippet, cand)
        if score > best_score:
            best_score, best_span = score, (i, i + window)
    if best_span and best_score >= fuzzy_threshold:
        return bbox_for_char_range(page, spans, *best_span), "fuzzy"
    return None, "none"
