#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build evaluation subsets for Stage 2 LLM experiments.

This script:
  - Reads the full narrative file (stage2_narratives_xgb.txt)
  - Reads suspicious_sequences_xgb.csv with ground-truth labels (is_malicious)
  - Derives a label per (file_user, date) window from suspicious_sequences_xgb.csv
  - Matches each narrative block to its (user, date) window using the first summary line
  - Samples up to N positive windows (true malicious) and N negative windows (false positives)
  - Writes a single text file containing both:
      * stage2_narratives_xgb_eval_100.txt  (up to 50 positives then 50 negatives)

This does NOT modify the original narrative file.
"""

import pathlib
import re
from typing import List, Tuple, Dict

import pandas as pd


BLOCK_SEP = "=" * 80


def load_labels(suspicious_path: pathlib.Path) -> Dict[Tuple[str, str], int]:
    """
    Build a mapping (user_id, date_str 'YYYY-MM-DD') -> label {0,1},
    where label is derived from suspicious_sequences_xgb.csv (is_malicious).
    We aggregate over all rows for that (user, date) and treat the window as
    malicious if ANY row has is_malicious == 1.
    """
    df = pd.read_csv(suspicious_path)
    if "file_user" not in df.columns or "file_date" not in df.columns or "is_malicious" not in df.columns:
        raise ValueError("suspicious_sequences_xgb.csv must contain 'file_user', 'file_date', 'is_malicious' columns")

    df["file_date"] = pd.to_datetime(df["file_date"], errors="coerce")
    df["date_only"] = df["file_date"].dt.date.astype(str)

    grouped = df.groupby(["file_user", "date_only"])["is_malicious"].max().reset_index()
    mapping: Dict[Tuple[str, str], int] = {}
    for _, row in grouped.iterrows():
        key = (row["file_user"], row["date_only"])
        mapping[key] = int(row["is_malicious"])
    return mapping


def parse_header_user_date(line: str) -> Tuple[str, str] | None:
    """
    Parse user_id and date from a header line like:
      'Summary of suspicious behavior for user AAF0535 on 2010-01-05: ...'
    Returns (user_id, 'YYYY-MM-DD') or None if no match.
    """
    m = re.search(r"Summary of suspicious behavior for user\s+(\S+)\s+on\s+(\d{4}-\d{2}-\d{2})", line)
    if not m:
        return None
    user_id = m.group(1)
    date_str = m.group(2)
    return user_id, date_str


def split_blocks_with_labels(
    narrative_path: pathlib.Path,
    label_map: Dict[Tuple[str, str], int],
) -> Tuple[List[str], List[str]]:
    """
    Walk through stage2_narratives_xgb.txt, split into blocks,
    and assign each block to positive or negative based on label_map.
    Blocks without a matching label are ignored.
    """
    pos_blocks: List[str] = []
    neg_blocks: List[str] = []

    current_lines: List[str] = []
    current_key: Tuple[str, str] | None = None

    with narrative_path.open("r", encoding="utf-8") as f:
        for line in f:
            # Try to parse header (user/date) the first time we see a summary line in this block.
            if current_key is None and "Summary of suspicious behavior for user" in line:
                key = parse_header_user_date(line)
                current_key = key

            current_lines.append(line)

            if line.strip() == BLOCK_SEP:
                # End of block
                block_text = "".join(current_lines).strip()
                if current_key is not None and current_key in label_map:
                    label = label_map[current_key]
                    if label == 1:
                        pos_blocks.append(block_text)
                    else:
                        neg_blocks.append(block_text)
                # Reset for next block
                current_lines = []
                current_key = None

    return pos_blocks, neg_blocks


def main() -> None:
    base = pathlib.Path(__file__).parent
    narrative_path = base / "stage2_narratives_xgb.txt"
    suspicious_path = base / "suspicious_sequences_xgb.csv"

    if not narrative_path.exists():
        raise FileNotFoundError(f"Narrative file not found: {narrative_path}")
    if not suspicious_path.exists():
        raise FileNotFoundError(f"Suspicious sequences file not found: {suspicious_path}")

    print(f"Loading labels from {suspicious_path} ...")
    label_map = load_labels(suspicious_path)
    print(f"  Built label map for {len(label_map):,} (user, date) windows")

    print(f"Splitting narrative blocks from {narrative_path} ...")
    pos_blocks, neg_blocks = split_blocks_with_labels(narrative_path, label_map)
    print(f"  Found {len(pos_blocks):,} positive blocks and {len(neg_blocks):,} negative blocks")

    # Sample up to 50 from each side (without shuffling to keep deterministic order)
    N = 50
    pos_sample = pos_blocks[:N]
    neg_sample = neg_blocks[:N]

    # Combined file: first positives then negatives in one file
    combined_out = base / "stage2_narratives_xgb_eval_100.txt"
    combined_blocks = []
    for blk in pos_sample:
        combined_blocks.append(blk)
    for blk in neg_sample:
        combined_blocks.append(blk)

    with combined_out.open("w", encoding="utf-8") as f:
        f.write("\n\n".join(combined_blocks))
        if combined_blocks:
            f.write("\n")

    print(f"Wrote combined eval set ({len(pos_sample)} positives + {len(neg_sample)} negatives) to {combined_out.name}")


if __name__ == "__main__":
    main()

