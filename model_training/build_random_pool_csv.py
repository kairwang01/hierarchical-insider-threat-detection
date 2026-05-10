#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a "random pool" CSV with the same schema as Stage-1 suspicious_sequences_*.csv
for single-stage LLM baselines (proposal VI.D.1): same number of LLM calls as the
hierarchical Stage-2 pool, but windows drawn uniformly (or stratified) from the full
sequence table instead of top-k risk scores.

Workflow:
  1) python build_random_pool_csv.py --source ../integrated_sequences_labeled.csv \\
         --reference suspicious_sequences_xgb.csv --stratified --seed 42 \\
         --output suspicious_sequences_random_pool.csv
  2) python stage2_narrative.py --suspicious suspicious_sequences_random_pool.csv ... \\
         --output stage2_narratives_random.txt
  3) python llm_evaluator.py --input stage2_narratives_random.txt --output llm_random_pool.jsonl ...

Compare metrics to hierarchical pool using the same eval_keys (e.g. tfidf_eval_split test).
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import random
import sys
from typing import Dict, List, Tuple

from label_utils import file_date_to_date_only, parse_binary_label


def load_windows_with_labels(path: pathlib.Path) -> List[Dict[str, str]]:
    rows_out: List[Dict[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames or "file_user" not in fieldnames or "file_date" not in fieldnames:
            raise ValueError(f"{path} needs file_user, file_date")
        for row in reader:
            row = {k.strip() if isinstance(k, str) else k: v for k, v in row.items()}
            u = (row.get("file_user") or "").strip()
            d = file_date_to_date_only(row.get("file_date") or "")
            if not u or len(d) < 10:
                continue
            key = (u, d)
            if key in seen:
                continue
            seen.add(key)
            rows_out.append(row)
    return rows_out


def count_labels_in_reference(ref_path: pathlib.Path) -> Tuple[int, int, int]:
    n = n_mal = 0
    with ref_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            row = {k.strip() if isinstance(k, str) else k: v for k, v in row.items()}
            u = (row.get("file_user") or "").strip()
            d = file_date_to_date_only(row.get("file_date") or "")
            if not u or len(d) < 10:
                continue
            n += 1
            if parse_binary_label(row.get("is_malicious")):
                n_mal += 1
    return n, n_mal, n - n_mal


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample random (user,day) pool for single-stage LLM baseline.")
    parser.add_argument(
        "--source",
        type=str,
        default="../integrated_sequences_labeled.csv",
        help="Full labeled sequence table (same columns as features / integrated_sequences).",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="suspicious_sequences_xgb.csv",
        help="Reference suspicious CSV: default --n = its row count; --stratified matches its malicious/benign counts.",
    )
    parser.add_argument("--n", type=int, default=0, help="Override sample size (0 = use reference row count).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Match malicious and benign counts from --reference (requires enough rows in source).",
    )
    parser.add_argument("--output", type=str, default="suspicious_sequences_random_pool.csv")
    args = parser.parse_args()

    base = pathlib.Path(__file__).parent
    src = (base / args.source).resolve() if not pathlib.Path(args.source).is_absolute() else pathlib.Path(args.source)
    ref = (base / args.reference).resolve() if not pathlib.Path(args.reference).is_absolute() else pathlib.Path(args.reference)
    out = (base / args.output).resolve() if not pathlib.Path(args.output).is_absolute() else pathlib.Path(args.output)

    if not src.exists():
        print(f"ERROR: {src}", file=sys.stderr)
        sys.exit(1)
    if not ref.exists():
        print(f"ERROR: {ref}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading unique windows from {src.name} ...", flush=True)
    all_rows = load_windows_with_labels(src)
    mal_rows = [r for r in all_rows if parse_binary_label(r.get("is_malicious"))]
    ben_rows = [r for r in all_rows if not parse_binary_label(r.get("is_malicious"))]
    print(f"  Total unique windows: {len(all_rows):,} (malicious {len(mal_rows):,}, benign {len(ben_rows):,})", flush=True)

    rng = random.Random(args.seed)
    n_ref, n_mal_ref, n_ben_ref = count_labels_in_reference(ref)
    n_target = args.n if args.n > 0 else n_ref
    print(f"Reference {ref.name}: n={n_ref}, malicious={n_mal_ref}, benign={n_ben_ref}", flush=True)

    chosen: List[Dict[str, str]] = []
    if args.stratified:
        if n_mal_ref > len(mal_rows):
            print(f"ERROR: need {n_mal_ref} malicious windows but source has {len(mal_rows)}", file=sys.stderr)
            sys.exit(1)
        if n_ben_ref > len(ben_rows):
            print(f"ERROR: need {n_ben_ref} benign windows but source has {len(ben_rows)}", file=sys.stderr)
            sys.exit(1)
        chosen.extend(rng.sample(mal_rows, n_mal_ref))
        chosen.extend(rng.sample(ben_rows, n_ben_ref))
        if len(chosen) != n_ref:
            print("WARN: stratified count mismatch vs reference; check reference row count.", flush=True)
    else:
        if n_target > len(all_rows):
            print(f"ERROR: n={n_target} > available {len(all_rows)}", file=sys.stderr)
            sys.exit(1)
        chosen = rng.sample(all_rows, n_target)

    # Match suspicious CSV columns: use first chosen row keys + risk_score
    template = chosen[0]
    out_fields = [k for k in template.keys() if k]
    if "risk_score" not in out_fields:
        out_fields.append("risk_score")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        for r in chosen:
            row = {k: r.get(k, "") for k in out_fields if k != "risk_score"}
            row["risk_score"] = "0.0"
            w.writerow(row)

    print(f"Wrote {out} ({len(chosen):,} rows), seed={args.seed}, stratified={args.stratified}", flush=True)


if __name__ == "__main__":
    main()
