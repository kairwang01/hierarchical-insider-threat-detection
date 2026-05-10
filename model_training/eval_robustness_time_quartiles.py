#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Proposal VI.E (lightweight): measure score quality when eval windows are split by calendar date.
Same model / same scores; only the evaluation subset changes — a simple temporal slice / drift proxy.

Example (model_training/):

  python eval_robustness_time_quartiles.py \\
    --suspicious suspicious_sequences_xgb.csv \\
    --keys-json tfidf_eval_split.json --keys-subset test \\
    --llm-jsonl llm_predictions_test.jsonl

Prints per-quartile (by window_date): n, prevalence, PR-AUC, best-F1 @ sweep, P/R/F1 at that threshold.
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys
from typing import Dict, List, Tuple

import numpy as np

from label_utils import load_true_labels
from llm_eval_metrics import load_llm_risk_by_key
from plot_stage2_comparison import best_f1_sweep, load_eval_keys_from_split


def date_to_ordinal(date_str: str) -> int:
    s = (date_str or "")[:10]
    try:
        y, m, d = int(s[:4]), int(s[5:7]), int(s[8:10])
        return dt.date(y, m, d).toordinal()
    except (ValueError, TypeError):
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-date-quartile metrics on a fixed score file.")
    parser.add_argument("--suspicious", type=str, default="suspicious_sequences_xgb.csv")
    parser.add_argument("--keys-json", type=str, required=True)
    parser.add_argument("--keys-subset", choices=("test", "train"), default="test")
    parser.add_argument("--llm-jsonl", type=str, default="", help="LLM predictions JSONL")
    parser.add_argument(
        "--scores-csv",
        type=str,
        default="",
        help="Optional CSV file_user,window_date,score (e.g. TF-IDF) instead of LLM",
    )
    args = parser.parse_args()

    base = pathlib.Path(__file__).parent
    susp = (base / args.suspicious).resolve()
    split_p = (base / args.keys_json).resolve()
    if not susp.exists() or not split_p.exists():
        print("ERROR: missing suspicious or keys-json", file=sys.stderr)
        sys.exit(1)

    label_map = load_true_labels(susp)
    eval_keys = [k for k in load_eval_keys_from_split(split_p, args.keys_subset) if k in label_map]
    if not eval_keys:
        print("ERROR: no eval keys", file=sys.stderr)
        sys.exit(1)

    scores: Dict[Tuple[str, str], float]
    name = ""
    if args.scores_csv:
        p = (base / args.scores_csv).resolve()
        if not p.exists():
            print(f"ERROR: {p}", file=sys.stderr)
            sys.exit(1)
        import csv

        scores = {}
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                row = {k.strip() if isinstance(k, str) else k: v for k, v in row.items()}
                u = (row.get("file_user") or "").strip()
                d = (row.get("window_date") or "").strip()[:10]
                if len(d) >= 10:
                    k = (u, d)
                    try:
                        s = float(row.get("score") or 0)
                    except (TypeError, ValueError):
                        s = 0.0
                    scores[k] = max(scores.get(k, float("-inf")), s)
        name = args.scores_csv
    elif args.llm_jsonl:
        p = (base / args.llm_jsonl).resolve()
        if not p.exists():
            print(f"ERROR: {p}", file=sys.stderr)
            sys.exit(1)
        scores = load_llm_risk_by_key(p, progress_every=0)
        name = args.llm_jsonl
    else:
        print("ERROR: provide --llm-jsonl or --scores-csv", file=sys.stderr)
        sys.exit(1)

    ords = np.array([date_to_ordinal(k[1]) for k in eval_keys], dtype=int)
    q1, q2, q3 = np.percentile(ords[ords > 0], [25, 50, 75]) if np.any(ords > 0) else (0, 0, 0)

    def quartile_idx(od: int) -> int:
        if od <= 0:
            return -1
        if od <= q1:
            return 0
        if od <= q2:
            return 1
        if od <= q3:
            return 2
        return 3

    buckets: List[List[Tuple[str, str]]] = [[] for _ in range(4)]
    skipped = 0
    for k in eval_keys:
        qi = quartile_idx(date_to_ordinal(k[1]))
        if qi < 0:
            skipped += 1
            continue
        buckets[qi].append(k)

    print(f"Score source: {name}", flush=True)
    print(f"Eval keys: {len(eval_keys):,} (skipped no-date: {skipped})", flush=True)
    print(f"Date quartile cutoffs (ordinal): Q1={q1:.0f} Q2={q2:.0f} Q3={q3:.0f}", flush=True)
    print(
        f"{'Q':>3} {'n':>6} {'prev%':>8} {'PR-AUC':>8} {'F1*':>8} {'P*':>8} {'R*':>8} {'thr*':>8}",
        flush=True,
    )

    for qi, keys_q in enumerate(buckets):
        if len(keys_q) < 5:
            print(f"Q{qi+1} too few keys ({len(keys_q)}), skip", flush=True)
            continue
        y_pos = sum(1 for k in keys_q if label_map.get(k, 0) == 1)
        prev = 100.0 * y_pos / len(keys_q) if keys_q else 0.0
        if y_pos == 0 or y_pos == len(keys_q):
            print(
                f"Q{qi+1} {len(keys_q):6d} {prev:8.2f} {'n/a':>8} {'n/a':>8} {'n/a':>8} {'n/a':>8} {'n/a':>8}  (single-class slice)",
                flush=True,
            )
            continue
        bf1, bp, br, bt, ap, _ = best_f1_sweep(label_map, keys_q, scores)
        print(
            f"Q{qi+1} {len(keys_q):6d} {prev:8.2f} {ap:8.4f} {bf1:8.4f} {bp:8.4f} {br:8.4f} {bt:8.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
