#!/usr/bin/env python3
"""Single entry point for the actes pipeline: OCR reconstruction -> event extraction
(rules + OpenRouter, context-aware) -> capital timeline replay -> results.json.

All content judgment — is this snippet really a capital event, does this amount
reconcile — happens inside the pipeline itself (regex in rules.py, OpenRouter calls
in extract_events.py's contextual_validate/contextual_fallback). This script only
orchestrates: it runs each phase as a subprocess and stops if one fails, exactly as
running them by hand in order would, just as one command.

Usage:
    python3 main.py                     # ARCHEAN TECHNOLOGIES (480489707), full pipeline
    python3 main.py --siren 499979540   # any other corpus company
    python3 main.py --no-llm            # rules only, skip every OpenRouter call
    python3 main.py --skip-ocr          # reuse existing output/<siren>/actes/*.json
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CHALLENGE_SIREN = "480489707"


def run(cmd: list[str], label: str) -> None:
    print(f"\n=== {label} ===", file=sys.stderr)
    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"\n{label} failed (exit {result.returncode}) after {time.time()-t0:.0f}s — stopping.",
              file=sys.stderr)
        raise SystemExit(result.returncode)
    print(f"({label} done in {time.time()-t0:.0f}s)", file=sys.stderr)


def phase1_ocr(siren: str) -> None:
    pdf_dir = os.path.join(REPO_ROOT, "data", siren, "actes", "pdf")
    out_dir = os.path.join(REPO_ROOT, "output", siren, "actes")
    os.makedirs(out_dir, exist_ok=True)
    pdfs = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    print(f"\n=== Phase 1: OCR reconstruction ({siren}, {len(pdfs)} documents) ===", file=sys.stderr)
    for pdf in pdfs:
        base = os.path.basename(pdf)[:-4]  # strip .pdf
        doc_id = base.split("_")[-1]
        date = base[len("acte_"):-(len(doc_id) + 1)]
        out_path = os.path.join(out_dir, f"{date}_{doc_id}.json")
        cmd = ["python3", "scripts/ocr_reconstruction.py", "--siren", siren, "--doc-id", doc_id, "-o", out_path]
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            print(f"OCR reconstruction failed for {base} (exit {result.returncode}) — stopping.", file=sys.stderr)
            raise SystemExit(result.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--siren", default=CHALLENGE_SIREN)
    ap.add_argument("--no-llm", action="store_true", help="rules only, skip every OpenRouter call")
    ap.add_argument("--skip-ocr", action="store_true", help="reuse existing output/<siren>/actes/*.json")
    args = ap.parse_args()

    t_start = time.time()

    if args.skip_ocr:
        print("=== Phase 1: OCR reconstruction — skipped (--skip-ocr) ===", file=sys.stderr)
    else:
        phase1_ocr(args.siren)

    events_path = os.path.join(REPO_ROOT, "output", args.siren, "events_candidates.json")
    extract_cmd = ["python3", "-u", "scripts/extract_events.py", "--siren", args.siren, "-o", events_path]
    if args.no_llm:
        extract_cmd.append("--no-llm")
    run(extract_cmd, "Phase 2: event extraction (rules + OpenRouter)")

    timeline_path = os.path.join(REPO_ROOT, "output", args.siren, "capital_timeline.json")
    run(["python3", "scripts/graph.py", "--siren", args.siren, "-o", timeline_path],
        "Phase 4: graph replay -> capital timeline")

    build_cmd = ["python3", "scripts/build_results.py", "--siren", args.siren]
    run(build_cmd, "Phase 6: assemble results.json")

    out_path = os.path.join(REPO_ROOT, "results.json") if args.siren == CHALLENGE_SIREN else \
        os.path.join(REPO_ROOT, "output", args.siren, "results.json")
    print(f"\nDone in {time.time()-t_start:.0f}s total. Output: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
