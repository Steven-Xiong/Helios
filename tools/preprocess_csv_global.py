#!/usr/bin/env python3
"""Preprocess Sekai CSV files: strip camera-movement descriptions from captions.

Applies the same strip_camera_motion() regex pipeline used to create the
yume_training_helios_latents_global .pt files, but operates directly on CSV
text — no tensor loading or model imports needed.

Usage:
    python tools/preprocess_csv_global.py \
        --input_csvs data/Sekai-Project/train/sekai-real-walking-hq.csv \
                     data/Sekai-Project/train/sekai-game-walking.csv \
                     data/Sekai-Project/train/sekai-game-drone.csv \
        --output_dir data/Sekai-Project/train_global

    # Or with default paths (all 3 Sekai CSVs):
    python tools/preprocess_csv_global.py

Outputs one CSV per input (same filename) into --output_dir.
The `caption` column is replaced by the cleaned global caption.
Original caption is preserved in a new `caption_original` column.
"""

import argparse
import csv
import os
import re
import sys

# ─── strip_camera_motion (copied from tools/offload_data/re_encode_prompt.py) ──
# Keep in sync with that file.  Only pure regex, no model dependencies.

_MOTION_VERBS = (
    r"(?:advances?|moves?|pans?|tilts?|glides?|shifts?|adjusts?"
    r"|continues?|transitions?|progresses|follows?|maintains?"
    r"|approaches|turns?|swings?|tracks?|sweeps?|zooms?"
    r"|pulls?|pushes?|ascends?|descends?|retreats?|emerges?"
    r"|navigates?|crosses?|provides?|offers?|reveals?"
    r"|captures?|remains?)"
)
_MOTION_ADVERBS = r"(?:smoothly |steadily |gently |briefly |subtly |occasionally |then |slowly |further )?"

_OPENING_LOCATION_RE = re.compile(
    r"^(?:"
    r"(?:The (?:video|footage|clip) (?:begins|starts|opens) with )"
    r"|(?:The first-person perspective (?:begins|starts) with )"
    r"|(?:The camera " + _MOTION_VERBS + r"\s+)"
    r")?"
    r"(?:a |the )?(?:first-person |FPV |steady |smooth )?"
    r"(?:perspective |view |forward movement |movement |motion )?"
    r"(?:(?:of someone\s+)?(?:moving|walking|advancing|progressing|traveling|gliding)\s+)?"
    r"(?:steadily |smoothly |slowly |forward )?"
    r"(?:along|through|across|over|down|into|past|forward along|ahead along)?\s+"
    r"(?P<location>[^.]*?)"
    r"\.\s*",
    re.IGNORECASE,
)

_CAMERA_SENTENCE_RE = re.compile(
    r"(?<=[.!?])\s*"
    r"(?:"
    r"(?:The (?:camera|viewer|movement|journey|walk|ascent|descent|advance|progression))"
    r"|(?:As the (?:viewer|person|camera|walk|journey|movement)\s+"
    r"(?:progresses|continues|advances|moves|proceeds))"
    r"|(?:The (?:entire )?(?:sequence|journey|video|walk)\s+"
    r"(?:maintains|unfolds|continues|captures))"
    r"|(?:Throughout the (?:entire )?(?:sequence|journey|video|walk|clip))"
    r"|(?:(?:The )?[Cc]ontinuing (?:onward|forward|straight))"
    r")"
    r"[^.!?]*[.!?]",
    re.IGNORECASE,
)

_CAMERA_CLAUSE_RE = re.compile(
    r",?\s*(?:the camera|the viewer(?:'s gaze)?|the perspective)\s+"
    + _MOTION_ADVERBS + _MOTION_VERBS
    + r"[^,.\n]*",
    re.IGNORECASE,
)

_LEADING_MOTION_RE = re.compile(
    r"^(?:The (?:video|footage|clip) (?:begins|starts|opens) with )?"
    r"(?:a |the )?(?:first-person |FPV |steady |smooth )?"
    r"(?:perspective|view|forward movement|movement|motion)"
    r"(?:\s+(?:of someone\s+)?(?:moving|walking|advancing|progressing|traveling|gliding))"
    r"[^.]*\.\s*",
    re.IGNORECASE,
)

_MOTION_PHRASE_RE = re.compile(
    r",\s*(?:moving|advancing|walking|progressing|traveling|gliding|proceeding)"
    r"(?:\s+(?:steadily |smoothly |slowly |forward |ahead )?"
    r"(?:forward|ahead|along|through|across|onward|deeper|further|into))"
    r"[^,.\n]*",
    re.IGNORECASE,
)

_FPV_MOTION_RE = re.compile(
    r"\s*with a (?:first-person|FPV|steady)\s+(?:perspective|view)\s+"
    r"(?:of someone\s+)?(?:moving|walking|advancing|progressing)[^,.]*",
    re.IGNORECASE,
)


def strip_camera_motion(caption: str) -> str:
    """Remove camera/viewer movement descriptions from a Sekai-style caption."""
    location_prefix = ""
    m = _OPENING_LOCATION_RE.match(caption)
    if m:
        loc = m.group("location").strip()
        if len(loc) > 15:
            location_prefix = loc.rstrip(",. ") + ". "

    text = caption
    text = _LEADING_MOTION_RE.sub("", text)
    text = _CAMERA_SENTENCE_RE.sub("", text)
    text = _CAMERA_CLAUSE_RE.sub("", text)
    text = _MOTION_PHRASE_RE.sub("", text)
    text = _FPV_MOTION_RE.sub("", text)

    text = re.sub(r"\s*,\s*,", ",", text)
    text = re.sub(r"\.\s*\.", ".", text)
    text = re.sub(r"(?<=\.)\s*(?:As|And|But|While)\s*,", ".", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAs\s*,\s*", "", text)
    text = re.sub(r"(?<=\.)\s*,\s*", ". ", text)
    text = re.sub(r"\.\s+(?:it|he|she|they)\s+" + _MOTION_VERBS + r"[^.]*\.", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(?:it|he|she|they)\s+" + _MOTION_VERBS + r"[^.]*\.\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*[,;.]\s*", "", text)
    text = re.sub(r"^\s*capturing\b", "Capturing", text)
    text = re.sub(r"\.\s*\.", ".", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip()

    if location_prefix:
        loc_parts = [p.strip() for p in location_prefix.rstrip(". ").split(",") if len(p.strip()) > 8]
        already_present = any(p.lower() in text[:150].lower() for p in loc_parts)
        if not already_present:
            text = location_prefix + text

    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    if not text or len(text) < 20:
        return caption

    return text


# ─── Main ─────────────────────────────────────────────────────────────────────

DEFAULT_CSVS = [
    "data/Sekai-Project/train/sekai-real-walking-hq.csv",
    "data/Sekai-Project/train/sekai-game-walking.csv",
    "data/Sekai-Project/train/sekai-game-drone.csv",
]
DEFAULT_OUTPUT_DIR = "data/Sekai-Project/train_global"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_csvs", nargs="+", default=DEFAULT_CSVS,
                        help="Sekai CSV files to process")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory for output CSVs")
    parser.add_argument("--caption_col", default="caption",
                        help="Column name containing the caption text")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for csv_path in args.input_csvs:
        if not os.path.exists(csv_path):
            print(f"SKIP: {csv_path} not found")
            continue

        rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames)
            for row in reader:
                rows.append(row)

        if args.caption_col not in fieldnames:
            print(f"SKIP: {csv_path} has no '{args.caption_col}' column")
            continue

        out_fields = fieldnames + (["caption_original"] if "caption_original" not in fieldnames else [])

        n_changed = 0
        for row in rows:
            original = row[args.caption_col]
            cleaned = strip_camera_motion(original)
            row["caption_original"] = original
            row[args.caption_col] = cleaned
            if cleaned != original:
                n_changed += 1

        out_path = os.path.join(args.output_dir, os.path.basename(csv_path))
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=out_fields)
            writer.writeheader()
            writer.writerows(rows)

        print(f"{os.path.basename(csv_path)}: {len(rows)} rows, {n_changed} captions cleaned → {out_path}")

        if rows:
            print(f"  Example original: {rows[0]['caption_original'][:120]}...")
            print(f"  Example cleaned:  {rows[0][args.caption_col][:120]}...")

    print("\nDone. Use the new CSVs in --image_prompt_csv_path for inference.")


if __name__ == "__main__":
    main()
