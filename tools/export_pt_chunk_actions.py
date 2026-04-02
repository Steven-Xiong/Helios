#!/usr/bin/env python3
"""Export per-chunk action annotations from .pt latent files into a CSV/TSV.

The .pt files produced by the data pipeline contain a `chunk_actions` field:
    chunk_actions: [{'keys': 'W+A', 'mouse': '→'}, {'keys': 'W', 'mouse': '·'}, ...]

This script scans a PT directory, builds a lookup table from video-stem →
chunk_actions, then joins it onto an existing CSV/TSV inference manifest
(e.g. the seadance TSV or a Sekai CSV) and writes out an augmented CSV with
a new `chunk_actions_json` column.

infer_helios.py reads this column (if present) with top priority over the
regex-parsed actions, giving ground-truth per-chunk actions at inference time.

Usage:
    python tools/export_pt_chunk_actions.py \\
        --pt_dir  data/helios/seadance2_v3_helios_latents \\
        --csv_in  data/seadance2_yume_v3/world_model_action12_train_3000_simple1cam_actionfirst.tsv \\
        --csv_out data/seadance2_yume_v3/train_with_chunk_actions.csv

    python tools/export_pt_chunk_actions.py \\
        --pt_dir  data/helios/yume_training_helios_latents_global \\
        --csv_in  data/Sekai-Project/train_global/sekai-game-walking.csv \\
        --csv_out data/Sekai-Project/train_global/sekai-game-walking-actions.csv
"""

import argparse
import csv
import json
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pt_dir", required=True,
                        help="Directory of .pt latent files with chunk_actions field")
    parser.add_argument("--csv_in", required=True,
                        help="Input CSV/TSV manifest")
    parser.add_argument("--csv_out", required=True,
                        help="Output CSV with chunk_actions_json column added")
    parser.add_argument("--video_col", default=None,
                        help="Column containing the video filename (auto-detected if omitted)")
    args = parser.parse_args()

    import torch

    print(f"Scanning PT files in {args.pt_dir} ...")
    pt_files = sorted(f for f in os.listdir(args.pt_dir) if f.endswith(".pt"))
    print(f"  Found {len(pt_files)} .pt files")

    # Build lookup: video_stem → list of {keys, mouse}
    chunk_actions_map = {}
    n_missing = 0
    for i, fname in enumerate(pt_files):
        if i % 2000 == 0:
            print(f"  Loading {i}/{len(pt_files)} ...")
        fpath = os.path.join(args.pt_dir, fname)
        try:
            data = torch.load(fpath, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  WARNING: skip {fname}: {e}")
            continue

        ca = data.get("chunk_actions")
        if ca is None:
            n_missing += 1
            del data
            continue

        # PT filename format: {video_stem}_{num_frames}_{h}_{w}.pt
        # Strip the last 3 numeric suffixes to recover video_stem
        base = fname[:-3]  # remove .pt
        parts = base.rsplit("_", 3)
        stem = parts[0] if len(parts) >= 4 else base

        # Normalize chunk_actions to list of [keys, mouse]
        if isinstance(ca, list):
            normalized = [[str(c.get("keys", "None")), str(c.get("mouse", "·"))]
                          if isinstance(c, dict) else list(c)
                          for c in ca]
        else:
            normalized = []

        chunk_actions_map[stem] = normalized
        del data

    print(f"  Built lookup: {len(chunk_actions_map)} entries, {n_missing} files missing chunk_actions")

    # Read input CSV/TSV
    sep = "\t" if args.csv_in.lower().endswith((".tsv", ".tab")) else ","
    rows = []
    with open(args.csv_in, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=sep)
        fieldnames = list(reader.fieldnames)
        for row in reader:
            rows.append(row)
    print(f"Input CSV: {len(rows)} rows, columns: {fieldnames}")

    # Auto-detect video column
    video_col = args.video_col
    if video_col is None:
        for candidate in ("videoFile", "video_name", "id"):
            if candidate in fieldnames:
                video_col = candidate
                break
    if video_col is None:
        raise ValueError(f"Cannot auto-detect video column from {fieldnames}. Pass --video_col.")
    print(f"Using video column: '{video_col}'")

    # Join chunk_actions
    out_fields = fieldnames + (["chunk_actions_json"] if "chunk_actions_json" not in fieldnames else [])
    n_matched = 0
    n_unmatched = 0
    for row in rows:
        vid = row.get(video_col, "")
        stem = os.path.splitext(os.path.basename(str(vid)))[0]
        ca = chunk_actions_map.get(stem)
        if ca is not None:
            row["chunk_actions_json"] = json.dumps(ca)
            n_matched += 1
        else:
            row["chunk_actions_json"] = ""
            n_unmatched += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.csv_out)), exist_ok=True)
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nOutput: {args.csv_out}")
    print(f"  Matched: {n_matched} / {len(rows)} rows")
    if n_unmatched:
        print(f"  Unmatched (chunk_actions_json will be empty): {n_unmatched}")

    if rows and rows[0]["chunk_actions_json"]:
        ex = json.loads(rows[0]["chunk_actions_json"])
        print(f"  Example ({video_col}={rows[0][video_col]}): {ex[:3]} ...")


if __name__ == "__main__":
    main()
