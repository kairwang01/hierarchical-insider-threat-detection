#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate window-level scores (CSV) against ground truth from suspicious_sequences_*.csv.

Use for Stage 2 baselines (rule / TF-IDF) and any method that outputs:
  file_user, window_date, score

Example:

    python eval_window_scores.py --scores baseline_rule_scores.csv --suspicious suspicious_sequences_xgb.csv --threshold 0.35
"""

import argparse
import csv
import pathlib
from typing import Dict, Tuple

from label_utils import load_true_labels


def load_scores_csv(
    path: pathlib.Path,
    user_col: str,
    date_col: str,
    score_col: str,
) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        for row in reader:
            row = {k.strip() if isinstance(k, str) else k: v for k, v in row.items()}
            u = (row.get(user_col) or "").strip()
            d = (row.get(date_col) or "").strip()[:10]
            if len(d) >= 10 and d[4] == "-" and d[7] == "-":
                pass
            else:
                continue
            try:
                s = float(row.get(score_col) or 0)
            except (TypeError, ValueError):
                s = 0.0
            key = (u, d)
            out[key] = max(out.get(key, float("-inf")), s)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate (user,date) scores vs ground truth.")
    parser.add_argument("--scores", type=str, required=True, help="CSV with file_user, window_date, score")
    parser.add_argument("--suspicious", type=str, default="suspicious_sequences_xgb.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--user-col", type=str, default="file_user")
    parser.add_argument("--date-col", type=str, default="window_date")
    parser.add_argument("--score-col", type=str, default="score")
    parser.add_argument(
        "--method-name",
        type=str,
        default="",
        help="Label printed in header (e.g. rule_baseline)",
    )
    args = parser.parse_args()

    base = pathlib.Path(__file__).parent
    score_path = (base / args.scores).resolve()
    susp_path = (base / args.suspicious).resolve()
    if not score_path.exists():
        raise FileNotFoundError(score_path)
    if not susp_path.exists():
        raise FileNotFoundError(susp_path)

    label_map = load_true_labels(susp_path)
    scores = load_scores_csv(score_path, args.user_col, args.date_col, args.score_col)

    name = args.method_name or score_path.name
    print(f"Method: {name}", flush=True)
    print(f"  Score file: {score_path}", flush=True)
    print(f"  Threshold: {args.threshold}", flush=True)

    tp = fp = tn = fn = 0
    missing_score = 0
    for key, true_y in label_map.items():
        s = scores.get(key)
        if s is None:
            missing_score += 1
            pred = 0
        else:
            pred = 1 if s >= args.threshold else 0
        if true_y == 1 and pred == 1:
            tp += 1
        elif true_y == 0 and pred == 1:
            fp += 1
        elif true_y == 0 and pred == 0:
            tn += 1
        else:
            fn += 1

    print("\nConfusion matrix (all keys in suspicious label map):", flush=True)
    print(f"  TP: {tp}\n  FP: {fp}\n  TN: {tn}\n  FN: {fn}", flush=True)
    if missing_score:
        print(f"\n  Windows with no score row (treated pred=0): {missing_score}", flush=True)

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    print("\nMetrics:", flush=True)
    print(f"  Precision: {p:.4f}\n  Recall:    {r:.4f}\n  F1-score:  {f1:.4f}", flush=True)


if __name__ == "__main__":
    main()
