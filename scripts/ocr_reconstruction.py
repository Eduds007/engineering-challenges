#!/usr/bin/env python3
"""Reconstruct visual reading order from the shipped OCR, using polygon coordinates
instead of the raw array order.

The OCR engine emits lines in detection order, not reading order — harmless for a
running paragraph, but it scrambles tables/lists (e.g. a founder's name and share
count landing on opposite sides of the array). This script clusters OCR lines into
visual rows by vertical (y) overlap, sorts each row left-to-right by x, and stitches
rows top-to-bottom. It also carries each line's normalized bbox through, so a later
grounding step can map an LLM-returned snippet back to a `[x0,y0,x1,y1]` box.

    pip install pymupdf

Usage:
    # by siren + doc id (looks up the pdf/ocr paths under data/<siren>/<kind>/)
    python scripts/ocr_reconstruction.py --siren 480489707 \
        --doc-id 63e9593b8be6eb9f9d257ec5 -o out.json

    # or point directly at a pdf + ocr dir
    python scripts/ocr_reconstruction.py --pdf <path.pdf> --ocr <ocr_dir> -o out.json

Output shape:
    {
      "doc_id": "...",
      "pdf": "acte_....pdf",
      "n_pages": 24,
      "pages": [
        {"page": 1, "text": "<reading-order text>", "lines": [{"text", "bbox_norm"}, ...]}
      ],
      "full_text": "[PAGE 1]\\n...\\n\\n[PAGE 2]\\n..."
    }
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from tools.bbox_viewer import polygon_to_norm  # reuse the exact px(300dpi) -> norm conversion

DATA_DIR = os.path.join(REPO_ROOT, "data")

ROW_OVERLAP_RATIO = 0.5  # fraction of the shorter line's height that must overlap to be "same row"


def find_doc_paths(siren: str, doc_id: str, kind: str = "actes") -> tuple[str, str]:
    """Resolve a document's PDF path and OCR directory from siren + doc_id."""
    base = os.path.join(DATA_DIR, siren, kind)
    pdf_matches = glob.glob(os.path.join(base, "pdf", f"*{doc_id}.pdf"))
    if not pdf_matches:
        raise FileNotFoundError(f"no PDF for siren={siren} doc_id={doc_id} under {base}/pdf")
    ocr_dir = os.path.join(base, "ocr", doc_id)
    return pdf_matches[0], ocr_dir


def load_page_ocr(ocr_dir: str, page: int) -> list[dict]:
    path = os.path.join(ocr_dir, f"page_{page:03d}.json")
    if not os.path.exists(path):
        return []
    data = json.load(open(path, encoding="utf-8"))
    return data.get("ocr") or []


def _y_extent(polygon: list[list[float]]) -> tuple[float, float]:
    ys = [p[1] for p in polygon]
    return min(ys), max(ys)


def _x_min(polygon: list[list[float]]) -> float:
    return min(p[0] for p in polygon)


def cluster_rows(ocr_lines: list[dict], overlap_ratio: float = ROW_OVERLAP_RATIO) -> list[list[dict]]:
    """Group OCR lines into visual rows by vertical overlap, ignoring array order."""
    enriched = []
    for line in ocr_lines:
        poly = line.get("polygon")
        text = (line.get("text") or "").strip()
        if not poly or not text:
            continue
        y0, y1 = _y_extent(poly)
        enriched.append({**line, "_y0": y0, "_y1": y1, "_yc": (y0 + y1) / 2})
    enriched.sort(key=lambda l: l["_yc"])

    rows: list[list[dict]] = []
    row_y0 = row_y1 = None
    current: list[dict] = []
    for line in enriched:
        if not current:
            current = [line]
            row_y0, row_y1 = line["_y0"], line["_y1"]
            continue
        overlap = min(row_y1, line["_y1"]) - max(row_y0, line["_y0"])
        min_height = min(row_y1 - row_y0, line["_y1"] - line["_y0"])
        if min_height > 0 and overlap / min_height > overlap_ratio:
            current.append(line)
            row_y0, row_y1 = min(row_y0, line["_y0"]), max(row_y1, line["_y1"])
        else:
            rows.append(current)
            current = [line]
            row_y0, row_y1 = line["_y0"], line["_y1"]
    if current:
        rows.append(current)
    return rows


def reconstruct_page(ocr_lines: list[dict], w_pt: float, h_pt: float) -> dict:
    """Reading-order text + per-line normalized bboxes for one page."""
    rows = cluster_rows(ocr_lines)
    ordered_lines: list[dict] = []
    row_texts: list[str] = []
    for row in rows:
        row_sorted = sorted(row, key=lambda l: _x_min(l["polygon"]))
        row_texts.append(" ".join(l["text"].strip() for l in row_sorted))
        for line in row_sorted:
            bbox_norm = polygon_to_norm(line["polygon"], w_pt, h_pt)
            ordered_lines.append({
                "text": line["text"].strip(),
                "bbox_norm": [round(v, 4) for v in bbox_norm],
            })
    return {"text": "\n".join(row_texts), "lines": ordered_lines}


def reconstruct_document(pdf_path: str, ocr_dir: str, doc_id: str | None = None) -> dict:
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz  # older PyMuPDF releases
        except ImportError:
            print("needs PyMuPDF:  pip install pymupdf", file=sys.stderr)
            raise

    doc = fitz.open(pdf_path)
    n_pages = len(doc)
    n_ocr_pages = len(glob.glob(os.path.join(ocr_dir, "page_*.json"))) if os.path.isdir(ocr_dir) else 0
    if os.path.isdir(ocr_dir) and n_ocr_pages != n_pages:
        print(f"note: {n_ocr_pages} OCR pages for a {n_pages}-page PDF — "
              f"some pages of this document have no OCR", file=sys.stderr)
    elif not os.path.isdir(ocr_dir):
        print(f"note: no OCR directory at {ocr_dir} — this document has no OCR at all", file=sys.stderr)

    pages_out = []
    full_text_parts = []
    for page_num in range(1, n_pages + 1):
        page = doc[page_num - 1]
        w_pt, h_pt = page.rect.width, page.rect.height
        ocr_lines = load_page_ocr(ocr_dir, page_num)
        if not ocr_lines:
            pages_out.append({"page": page_num, "text": "", "lines": []})
            full_text_parts.append(f"[PAGE {page_num}]\n(no OCR for this page)")
            continue
        result = reconstruct_page(ocr_lines, w_pt, h_pt)
        pages_out.append({"page": page_num, "text": result["text"], "lines": result["lines"]})
        full_text_parts.append(f"[PAGE {page_num}]\n{result['text']}")

    return {
        "doc_id": doc_id,
        "pdf": os.path.basename(pdf_path),
        "n_pages": n_pages,
        "pages": pages_out,
        "full_text": "\n\n".join(full_text_parts),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--siren")
    ap.add_argument("--doc-id")
    ap.add_argument("--kind", default="actes", choices=["actes", "bilans"])
    ap.add_argument("--pdf", help="direct path to the PDF (alternative to --siren/--doc-id)")
    ap.add_argument("--ocr", help="direct path to the OCR dir (alternative to --siren/--doc-id)")
    ap.add_argument("-o", "--out", help="write JSON here (default: stdout)")
    args = ap.parse_args()

    if args.pdf:
        pdf_path, ocr_dir, doc_id = args.pdf, args.ocr, args.doc_id
    elif args.siren and args.doc_id:
        pdf_path, ocr_dir = find_doc_paths(args.siren, args.doc_id, args.kind)
        doc_id = args.doc_id
    else:
        ap.error("pass either --siren/--doc-id or --pdf [--ocr]")
        return 2

    result = reconstruct_document(pdf_path, ocr_dir, doc_id)
    out = json.dumps(result, ensure_ascii=False, indent=2)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"wrote {args.out} ({result['n_pages']} pages)", file=sys.stderr)
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
