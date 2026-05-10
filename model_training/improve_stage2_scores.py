#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fuse LLM risk_score with Stage 1 risk_score (from suspicious_sequences_*.csv) using
logistic regression on standardized features. Optionally add Rule and/or TF-IDF baseline scores
(same CSV format as stage2_baselines.py: file_user, window_date, score).

- Trains ONLY on train users/keys (from --split-json train list, or internal GroupShuffleSplit)
  so test metrics are not leaked into the calibrator.
- Writes file_user, window_date, score (calibrated P(malicious)) for every window in label_map.

Feature order: LLM, [Stage1_risk], [Rule], [TF-IDF] — Stage1 omitted if ``--no-stage1``
(for a **TF-IDF + LLM** collaborative scorer without Stage 1 risk in the stack).

Improves ranking vs raw LLM when signals complement; feed output to:

    python plot_stage2_comparison.py ... --fused-csv stage2_fused_llm_tfidf.csv \\
        --fused-label "Fused (LLM+TF-IDF)"

Usage:

    # Use same split as TF-IDF / fair LLM eval:
    python improve_stage2_scores.py --suspicious suspicious_sequences_xgb.csv \\
        --llm-jsonl llm_predictions_xgb_all.jsonl --rule-csv baseline_rule_scores.csv \\
        --split-json tfidf_eval_split.json --output stage2_fused_lr_scores.csv

    # Add TF-IDF scores (e.g. baseline_tfidf_full_scores.csv from --full-pool-fit):
    python improve_stage2_scores.py --suspicious suspicious_sequences_xgb.csv \\
        --llm-jsonl llm_predictions_xgb.jsonl --tfidf-csv baseline_tfidf_full_scores.csv \\
        --output stage2_fused_with_tfidf.csv --test-size 0.2 --random-state 42

    # No split file: random 80/20 by user (GroupShuffleSplit)
    python improve_stage2_scores.py --suspicious suspicious_sequences_xgb.csv \\
        --llm-jsonl llm_predictions_xgb_all.jsonl --output stage2_fused_lr_scores.csv \\
        --test-size 0.2 --random-state 42

    # Collaborative LLM + TF-IDF only (no Stage1 column in the LR):
    python improve_stage2_scores.py --suspicious suspicious_sequences_xgb.csv \\
        --llm-jsonl llm_predictions_xgb_all.jsonl --tfidf-csv baseline_tfidf_full_scores.csv \\
        --no-stage1 --split-json tfidf_eval_split.json --output stage2_fused_llm_tfidf.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from label_utils import file_date_to_date_only, load_true_labels
from llm_eval_metrics import load_llm_risk_by_key


def load_stage1_risk_max(path: pathlib.Path) -> Dict[Tuple[str, str], float]:
    """Max Stage1 risk_score per (user, YYYY-MM-DD)."""
    best: Dict[Tuple[str, str], float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "risk_score" not in {h.strip() for h in reader.fieldnames}:
            raise ValueError(f"{path} must contain column 'risk_score'")
        for row in reader:
            row = {k.strip() if isinstance(k, str) else k: v for k, v in row.items()}
            u = (row.get("file_user") or "").strip()
            d = file_date_to_date_only(row.get("file_date") or "")
            if not u or len(d) < 10:
                continue
            try:
                r = float(row.get("risk_score") or 0)
            except (TypeError, ValueError):
                r = 0.0
            k = (u, d)
            prev = best.get(k, float("-inf"))
            best[k] = max(prev, r)
    return best


def load_scores_csv_simple(path: pathlib.Path) -> Dict[Tuple[str, str], float]:
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
            k = (u, d)
            scores[k] = max(scores.get(k, float("-inf")), s)
    return scores


def load_split_keys(path: pathlib.Path, subset: str) -> List[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    key = "train_keys" if subset == "train" else "test_keys"
    raw = data.get(key) or []
    out: List[Tuple[str, str]] = []
    for row in raw:
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            out.append((str(row[0]).strip(), str(row[1]).strip()[:10]))
    return out


def build_xy(
    keys: List[Tuple[str, str]],
    label_map: Dict[Tuple[str, str], int],
    llm: Dict[Tuple[str, str], float],
    s1: Dict[Tuple[str, str], float] | None,
    rule: Dict[Tuple[str, str], float] | None,
    tfidf: Dict[Tuple[str, str], float] | None,
) -> Tuple[np.ndarray, np.ndarray]:
    n_feat = 1 + (1 if s1 is not None else 0) + (1 if rule is not None else 0) + (1 if tfidf is not None else 0)
    X = np.zeros((len(keys), n_feat), dtype=np.float64)
    y = np.zeros(len(keys), dtype=np.int32)
    for i, k in enumerate(keys):
        if k not in label_map:
            continue
        c = 0
        X[i, c] = float(llm.get(k, 0.0))
        c += 1
        if s1 is not None:
            X[i, c] = float(s1.get(k, 0.0))
            c += 1
        if rule is not None:
            X[i, c] = float(rule.get(k, 0.0))
            c += 1
        if tfidf is not None:
            X[i, c] = float(tfidf.get(k, 0.0))
        y[i] = int(label_map[k])
    return X, y


def best_f1_for_scores(
    label_map: Dict[Tuple[str, str], int],
    eval_keys: List[Tuple[str, str]],
    scores: Dict[Tuple[str, str], float],
    n_steps: int = 101,
) -> Tuple[float, float, float, float, float]:
    """Returns f1, p, r, t, ap."""
    keys = [k for k in eval_keys if k in label_map]
    y = np.array([label_map[k] for k in keys], dtype=int)
    s = np.array([float(scores.get(k, 0.0)) for k in keys], dtype=float)
    ap = (
        float(average_precision_score(y, s))
        if len(np.unique(y)) > 1
        else 0.0
    )
    best_f1 = -1.0
    best_t, best_p, best_r = 0.5, 0.0, 0.0
    for t in np.linspace(0.0, 1.0, n_steps):
        pred = (s >= t).astype(int)
        tp = int(np.sum((y == 1) & (pred == 1)))
        fp = int(np.sum((y == 0) & (pred == 1)))
        fn = int(np.sum((y == 1) & (pred == 0)))
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        if f1 > best_f1:
            best_f1, best_t, best_p, best_r = f1, t, p, r
    return best_f1, best_p, best_r, best_t, ap


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse LLM + Stage1 (+Rule +TF-IDF) with logistic regression.")
    parser.add_argument("--suspicious", type=str, default="suspicious_sequences_xgb.csv")
    parser.add_argument("--llm-jsonl", type=str, required=True)
    parser.add_argument("--rule-csv", type=str, default="", help="Optional Rule score column (stage2_baselines rule output CSV)")
    parser.add_argument(
        "--tfidf-csv",
        type=str,
        default="",
        help="Optional TF-IDF+LR score column (same CSV format as --rule-csv)",
    )
    parser.add_argument(
        "--split-json",
        type=str,
        default="",
        help="Use train_keys from this JSON for fitting (same file as stage2_baselines --write-split).",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="If no --split-json, GroupShuffleSplit test fraction")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", type=str, default="stage2_fused_lr_scores.csv")
    parser.add_argument(
        "--no-stage1",
        action="store_true",
        help="Do not use Stage1 risk_score as a feature (e.g. build LLM+TF-IDF-only fusion).",
    )
    args = parser.parse_args()

    base = pathlib.Path(__file__).parent
    susp_path = (base / args.suspicious).resolve()
    llm_path = (base / args.llm_jsonl).resolve()
    if not susp_path.exists():
        print(f"ERROR: {susp_path}", file=sys.stderr)
        sys.exit(1)
    if not llm_path.exists():
        print(f"ERROR: {llm_path}", file=sys.stderr)
        sys.exit(1)

    label_map = load_true_labels(susp_path)
    print(f"Labels: {len(label_map):,} windows", flush=True)

    s1 = load_stage1_risk_max(susp_path)
    print(f"Stage1 risk_score: {len(s1):,} keys", flush=True)

    llm = load_llm_risk_by_key(llm_path, progress_every=0)
    print(f"LLM scores: {len(llm):,} keys", flush=True)

    rule: Dict[Tuple[str, str], float] | None = None
    if args.rule_csv:
        rp = (base / args.rule_csv).resolve()
        if rp.exists():
            rule = load_scores_csv_simple(rp)
            print(f"Rule scores: {len(rule):,} keys", flush=True)
        else:
            print(f"WARN: rule CSV not found {rp}, continuing without rule feature", flush=True)

    tfidf: Dict[Tuple[str, str], float] | None = None
    if args.tfidf_csv:
        tp = (base / args.tfidf_csv).resolve()
        if tp.exists():
            tfidf = load_scores_csv_simple(tp)
            print(f"TF-IDF scores: {len(tfidf):,} keys", flush=True)
        else:
            print(f"WARN: TF-IDF CSV not found {tp}, continuing without TF-IDF feature", flush=True)

    s1_for_xy: Dict[Tuple[str, str], float] | None = s1 if not args.no_stage1 else None
    if args.no_stage1 and rule is None and tfidf is None:
        print(
            "ERROR: --no-stage1 requires at least one of --tfidf-csv or --rule-csv "
            "(need a second feature besides LLM).",
            file=sys.stderr,
        )
        sys.exit(1)

    keys_all = sorted(label_map.keys())

    if args.split_json:
        sp = (base / args.split_json).resolve()
        if not sp.exists():
            print(f"ERROR: {sp}", file=sys.stderr)
            sys.exit(1)
        train_keys = [k for k in load_split_keys(sp, "train") if k in label_map]
        test_keys = [k for k in load_split_keys(sp, "test") if k in label_map]
        print(f"Split JSON: {len(train_keys):,} train / {len(test_keys):,} test windows", flush=True)
    else:
        users = np.array([k[0] for k in keys_all])
        y_all = np.array([label_map[k] for k in keys_all])
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
        tr_idx, te_idx = next(gss.split(np.zeros(len(keys_all)), y_all, groups=users))
        train_keys = [keys_all[i] for i in tr_idx]
        test_keys = [keys_all[i] for i in te_idx]
        print(
            f"Internal GroupShuffleSplit: {len(train_keys):,} train / {len(test_keys):,} test windows",
            flush=True,
        )

    if len(train_keys) < 50:
        print("ERROR: too few train keys for stable LR.", file=sys.stderr)
        sys.exit(1)

    n_llm_tr = sum(1 for k in train_keys if k in llm)
    if n_llm_tr < max(10, len(train_keys) // 20):
        print(
            f"WARN: only {n_llm_tr}/{len(train_keys)} train windows appear in LLM JSONL — "
            "if you only ran LLM on test, fusion weights for LLM are unreliable. "
            "Prefer full-pool LLM JSONL + --split-json.",
            flush=True,
        )

    X_tr, y_tr = build_xy(train_keys, label_map, llm, s1_for_xy, rule, tfidf)
    if y_tr.min() == y_tr.max():
        print("ERROR: train set has only one class.", file=sys.stderr)
        sys.exit(1)

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=args.random_state,
                ),
            ),
        ]
    )
    clf.fit(X_tr, y_tr)
    feat_parts = ["LLM"]
    if s1_for_xy is not None:
        feat_parts.append("Stage1_risk")
    if rule is not None:
        feat_parts.append("Rule")
    if tfidf is not None:
        feat_parts.append("TF-IDF")
    print(f"Fitted LogisticRegression({', '.join(feat_parts)})", flush=True)

    # Baselines on test (same keys)
    baselines: List[Tuple[str, Dict[Tuple[str, str], float]]] = [("LLM_raw", llm)]
    if s1_for_xy is not None:
        baselines.append(("Stage1_only", s1))
    for name, score_dict in baselines:
        bf1, p, r, t, ap = best_f1_for_scores(label_map, test_keys, score_dict)
        print(f"  Test {name}: AP={ap:.4f}  bestF1={bf1:.4f} (thr~{t:.2f}) P={p:.4f} R={r:.4f}", flush=True)

    X_all, _ = build_xy(keys_all, label_map, llm, s1_for_xy, rule, tfidf)
    fused_proba = clf.predict_proba(X_all)[:, 1]
    fused_scores = {keys_all[i]: float(fused_proba[i]) for i in range(len(keys_all))}

    bf1, p, r, t, ap = best_f1_for_scores(label_map, test_keys, fused_scores)
    print(f"  Test Fused:  AP={ap:.4f}  bestF1={bf1:.4f} (thr~{t:.2f}) P={p:.4f} R={r:.4f}", flush=True)

    out_path = (base / args.output).resolve()
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_user", "window_date", "score"])
        for k in keys_all:
            w.writerow([k[0], k[1], f"{fused_scores[k]:.8f}"])
    print(f"Wrote {out_path} ({len(keys_all):,} rows)", flush=True)


if __name__ == "__main__":
    main()
