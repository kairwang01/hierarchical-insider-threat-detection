#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 pool baseline: every window in suspicious_sequences_*.csv is "escalated" (pred=1).

This is the natural comparator for Stage 2: Stage 1 Top-K already flagged them;
precision = malicious_windows / total_windows, recall (within pool) = 1.0 for malicious present.

Usage:

    python stage1_pool_metrics.py --suspicious suspicious_sequences_xgb.csv
"""

import argparse
import pathlib

from label_utils import load_true_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-1 pool 'all escalate' baseline metrics.")
    parser.add_argument("--suspicious", type=str, default="suspicious_sequences_xgb.csv")
    args = parser.parse_args()

    base = pathlib.Path(__file__).parent
    susp_path = (base / args.suspicious).resolve()
    if not susp_path.exists():
        raise FileNotFoundError(susp_path)

    label_map = load_true_labels(susp_path)
    n = len(label_map)
    pos = sum(label_map.values())
    neg = n - pos

    # pred = 1 for all
    tp = pos
    fp = neg
    tn = 0
    fn = 0

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0

    print("Stage 1 pool baseline: predict MALICIOUS for every (user, day) in the suspicious CSV")
    print(f"  File: {susp_path}")
    print(f"  Unique windows: {n:,}  (malicious={pos:,}, benign={neg:,})")
    print("\nConfusion matrix (within this pool only):")
    print(f"  TP: {tp}\n  FP: {fp}\n  TN: {tn}\n  FN: {fn}")
    print("\nMetrics:")
    print(f"  Precision: {p:.4f}")
    print(f"  Recall:    {r:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print("\nInterpretation: Stage 2 (LLM / baselines) should improve Precision while keeping Recall high.")


if __name__ == "__main__":
    main()
