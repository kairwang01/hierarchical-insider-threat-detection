#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Print a compact ablation table by running in-process metrics helpers.

Does NOT call the OpenAI API. Requires:
  - suspicious_sequences_xgb.csv
  - Optional: baseline_rule_scores.csv, baseline_tfidf_full_scores.csv (generate with stage2_baselines.py)
  - Optional: llm_predictions_xgb.jsonl (LLM eval uses same logic as llm_eval_metrics)

Usage (from model_training/):

    python stage2_baselines.py --mode rule --output baseline_rule_scores.csv
    python stage2_baselines.py --mode tfidf_lr --output baseline_tfidf_full_scores.csv --full-pool-fit
    python ablation_compare.py --suspicious suspicious_sequences_xgb.csv --llm-jsonl llm_predictions_xgb.jsonl --rule-csv baseline_rule_scores.csv --tfidf-csv baseline_tfidf_full_scores.csv
"""

import argparse
import pathlib
from typing import Dict, Tuple

from label_utils import load_true_labels
from llm_eval_metrics import load_llm_risk_by_key


def _metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def stage1_pool(label_map: Dict[Tuple[str, str], int]) -> Tuple[int, int, int, int]:
    """All windows predicted malicious: TP=malicious count, FP=benign count, TN=FN=0."""
    pos = sum(label_map.values())
    n = len(label_map)
    neg = n - pos
    return pos, neg, 0, 0


def eval_llm_jsonl(path: pathlib.Path, label_map: Dict, threshold: float) -> Tuple[int, int, int, int]:
    """One prediction per (user, date): max risk_score across duplicate JSONL lines."""
    risks = load_llm_risk_by_key(path, progress_every=0)
    tp = fp = tn = fn = 0
    for key, true_y in label_map.items():
        s = risks.get(key)
        pred = 0 if s is None else (1 if s >= threshold else 0)
        if true_y == 1 and pred == 1:
            tp += 1
        elif true_y == 0 and pred == 1:
            fp += 1
        elif true_y == 0 and pred == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def _max_score_in_scores_csv(path: pathlib.Path) -> float:
    import csv

    m = float("-inf")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            row = {k.strip() if isinstance(k, str) else k: v for k, v in row.items()}
            try:
                s = float(row.get("score") or 0)
            except (TypeError, ValueError):
                s = 0.0
            m = max(m, s)
    return float(m) if m > float("-inf") else 0.0


def eval_scores_csv(
    path: pathlib.Path,
    label_map: Dict,
    threshold: float,
) -> Tuple[int, int, int, int]:
    import csv

    scores: Dict[Tuple[str, str], float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            row = {k.strip() if isinstance(k, str) else k: v for k, v in row.items()}
            u = (row.get("file_user") or "").strip()
            d = (row.get("window_date") or "").strip()[:10]
            if len(d) < 10:
                continue
            try:
                s = float(row.get("score") or 0)
            except (TypeError, ValueError):
                s = 0.0
            scores[(u, d)] = max(scores.get((u, d), float("-inf")), s)

    tp = fp = tn = fn = 0
    for key, true_y in label_map.items():
        s = scores.get(key)
        pred = 0 if s is None else (1 if s >= threshold else 0)
        if true_y == 1 and pred == 1:
            tp += 1
        elif true_y == 0 and pred == 1:
            fp += 1
        elif true_y == 0 and pred == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def row(name: str, tp: int, fp: int, tn: int, fn: int) -> str:
    p, r, f1 = _metrics_from_counts(tp, fp, tn, fn)
    return f"{name:22s}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  (TP={tp} FP={fp} TN={tn} FN={fn})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Print ablation comparison table.")
    parser.add_argument("--suspicious", type=str, default="suspicious_sequences_xgb.csv")
    parser.add_argument("--threshold-llm", type=float, default=0.5)
    parser.add_argument("--threshold-rule", type=float, default=0.15)
    parser.add_argument("--threshold-tfidf", type=float, default=0.5)
    parser.add_argument("--llm-jsonl", type=str, default="", help="Optional LLM predictions JSONL")
    parser.add_argument("--rule-csv", type=str, default="", help="Optional rule baseline scores CSV")
    parser.add_argument("--tfidf-csv", type=str, default="", help="Optional TF-IDF scores CSV")
    args = parser.parse_args()

    base = pathlib.Path(__file__).parent
    susp_path = (base / args.suspicious).resolve()
    label_map = load_true_labels(susp_path)

    tp, fp, tn, fn = stage1_pool(label_map)
    # stage1_pool returns pos,neg - remap to TP,FP,TN,FN for all-pred-1
    pos, neg = tp, fp
    lines = [
        "Ablation (same label map = all windows in suspicious CSV)",
        row("Stage1_pool_all_1", pos, neg, 0, 0),
    ]

    if args.llm_jsonl:
        p = (base / args.llm_jsonl).resolve()
        if p.exists():
            t1, f1, t2, f2 = eval_llm_jsonl(p, label_map, args.threshold_llm)
            lines.append(row(f"Stage2_LLM@{args.threshold_llm}", t1, f1, t2, f2))
        else:
            lines.append(f"(skip LLM: not found {p})")

    if args.rule_csv:
        p = (base / args.rule_csv).resolve()
        if p.exists():
            mx = _max_score_in_scores_csv(p)
            if mx < args.threshold_rule:
                lines.append(
                    f"  WARNING: max score in {p.name} is {mx:.4f} < --threshold-rule "
                    f"{args.threshold_rule} → no positive predictions (lower threshold or rescale scores)."
                )
            t1, f1, t2, f2 = eval_scores_csv(p, label_map, args.threshold_rule)
            lines.append(row(f"Stage2_rule@{args.threshold_rule}", t1, f1, t2, f2))
        else:
            lines.append(f"(skip rule: not found {p})")

    if args.tfidf_csv:
        p = (base / args.tfidf_csv).resolve()
        if p.exists():
            mx = _max_score_in_scores_csv(p)
            if mx < args.threshold_tfidf:
                lines.append(
                    f"  WARNING: max score in {p.name} is {mx:.4f} < --threshold-tfidf "
                    f"{args.threshold_tfidf} → no positive predictions."
                )
            t1, f1, t2, f2 = eval_scores_csv(p, label_map, args.threshold_tfidf)
            lines.append(row(f"Stage2_tfidf@{args.threshold_tfidf}", t1, f1, t2, f2))
        else:
            lines.append(f"(skip tfidf: not found {p})")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
