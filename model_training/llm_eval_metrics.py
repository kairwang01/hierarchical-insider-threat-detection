#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Evaluation Metrics for Stage 2

Evaluates LLM predictions (JSONL from llm_evaluator.py) against ground truth
from suspicious_sequences_xgb.csv.

Large JSONL rows (full narrative per line) make json.loads very slow. This
script uses regex on each line when possible and falls back to json.loads only
if needed. Multiple JSONL lines for the same (user, date) are merged by taking
the maximum risk_score (same spirit as duplicate rows in score CSVs).
Use --progress-every N for heartbeat output while scanning huge files.

Usage (from model_training/):

    python llm_eval_metrics.py --predictions llm_predictions_xgb.jsonl --suspicious suspicious_sequences_xgb.csv --threshold 0.5
"""

import argparse
import json
import pathlib
import re
from typing import Any, Dict, Tuple

from label_utils import load_true_labels

# Narrative header inside the JSON string
_HEADER_RE = re.compile(
    r"Summary of suspicious behavior for user\s+(\S+)\s+on\s+(\d{4}-\d{2}-\d{2})"
)
# First numeric risk_score in the line (covers "parsed" and often raw_output JSON)
_RISK_SCORE_RE = re.compile(
    r'"risk_score"\s*:\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)'
)


def parse_header_user_date_from_narrative(narrative: str) -> Tuple[str, str] | None:
    head = "\n".join(narrative.splitlines()[:5])
    m = _HEADER_RE.search(head)
    if not m:
        return None
    return m.group(1), m.group(2)


def extract_key_and_risk_from_line(line: str) -> Tuple[Tuple[str, str] | None, float | None]:
    """
    Fast path: regex on raw JSONL line (avoids json.loads on multi-MB objects).
    Returns ( (user, date) or None, risk_score or None ).
    """
    m = _HEADER_RE.search(line)
    if not m:
        return None, None
    user_id, date_str = m.group(1), m.group(2)
    key: Tuple[str, str] = (user_id, date_str)

    rs: float | None = None
    for rm in _RISK_SCORE_RE.finditer(line):
        try:
            rs = float(rm.group(1))
        except ValueError:
            continue
    # Prefer last match: "parsed" usually follows "raw_output" in our dumps
    if rs is None:
        return key, None
    return key, rs


def load_llm_risk_by_key(
    path: pathlib.Path,
    progress_every: int = 0,
) -> Dict[Tuple[str, str], float]:
    """
    Scan JSONL once; for each (user, date) keep the maximum risk_score seen.
    Avoids double-counting duplicate lines and matches eval_window_scores (one prediction per window).
    """
    best: Dict[Tuple[str, str], float] = {}
    lines_done = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, risk = extract_key_and_risk_from_line(line)
            if key is None or risk is None:
                try:
                    rec: Any = json.loads(line)
                    narrative = (rec.get("narrative") or "") if isinstance(rec, dict) else ""
                    key2 = parse_header_user_date_from_narrative(narrative)
                    if key2 is not None:
                        key = key2
                    if risk is None and isinstance(rec, dict):
                        parsed = rec.get("parsed")
                        if isinstance(parsed, dict) and isinstance(parsed.get("risk_score"), (int, float)):
                            risk = float(parsed["risk_score"])
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
            if key is None or risk is None:
                lines_done += 1
                if progress_every and lines_done % progress_every == 0:
                    print(f"  ... scanned {lines_done} lines, {len(best):,} keys with scores", flush=True)
                continue
            prev = best.get(key)
            if prev is None or risk > prev:
                best[key] = risk
            lines_done += 1
            if progress_every and lines_done % progress_every == 0:
                print(f"  ... scanned {lines_done} lines, {len(best):,} keys with scores", flush=True)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLM predictions against ground truth.")
    parser.add_argument(
        "--predictions",
        type=str,
        default="llm_predictions_xgb_100.jsonl",
        help="JSONL from llm_evaluator.py",
    )
    parser.add_argument(
        "--suspicious",
        type=str,
        default="suspicious_sequences_xgb.csv",
        help="Suspicious sequences CSV with is_malicious",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold on risk_score (default: 0.5)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Print progress every N JSONL lines (0=disable, default: 200)",
    )

    args = parser.parse_args()

    base = pathlib.Path(__file__).parent
    pred_path = (base / args.predictions).resolve()
    susp_path = (base / args.suspicious).resolve()

    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")
    if not susp_path.exists():
        raise FileNotFoundError(f"Suspicious sequences file not found: {susp_path}")

    print(f"Loading ground-truth labels from {susp_path} ...", flush=True)
    label_map = load_true_labels(susp_path)
    print(f"  Built label map for {len(label_map):,} (user, date) windows", flush=True)

    tp = fp = tn = fn = 0

    print(
        f"Scanning predictions from {pred_path} (max risk per window; threshold={args.threshold}) ...",
        flush=True,
    )
    risks = load_llm_risk_by_key(pred_path, progress_every=args.progress_every)
    print(f"  Found scores for {len(risks):,} distinct (user, date) keys", flush=True)

    missing_score = 0
    for key, true_y in label_map.items():
        s = risks.get(key)
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

    print("\nConfusion matrix (one row per window in suspicious CSV):", flush=True)
    print(f"  TP: {tp}", flush=True)
    print(f"  FP: {fp}", flush=True)
    print(f"  TN: {tn}", flush=True)
    print(f"  FN: {fn}", flush=True)
    print("\nCoverage:", flush=True)
    print(
        f"  Windows with no usable LLM score (pred=0): {missing_score:,} / {len(label_map):,}",
        flush=True,
    )

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\nMetrics:", flush=True)
    print(f"  Precision: {precision:.4f}", flush=True)
    print(f"  Recall:    {recall:.4f}", flush=True)
    print(f"  F1-score:  {f1:.4f}", flush=True)


if __name__ == "__main__":
    main()
