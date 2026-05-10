#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 non-LLM baselines for ablation / comparison.

1) rule: Heuristic score from narrative keywords (no training, no API).
2) tfidf_lr: TF-IDF + LogisticRegression on narrative text.
   - Default: GroupShuffleSplit by user (train 80% / test 20%) — rigorous, metrics on test users only.
   - --use-split-json: reuse train/test keys from an existing JSON (same schema as --write-split).
   - --honest-fusion-csv: after held-out fit, write train+test TF-IDF probs for improve_stage2_scores (fair vs test-only baseline).
   - --full-pool-fit: fit on all windows (optimistic / matches full-pool LLM eval protocol; may overfit).

Outputs CSV: file_user, window_date, score  (for eval_window_scores.py)

Example:

    python stage2_baselines.py --mode rule --narratives stage2_narratives_xgb.txt --suspicious suspicious_sequences_xgb.csv --output baseline_rule_scores.csv
    python eval_window_scores.py --scores baseline_rule_scores.csv --suspicious suspicious_sequences_xgb.csv --threshold 0.35

    python stage2_baselines.py --mode tfidf_lr --narratives stage2_narratives_xgb.txt --suspicious suspicious_sequences_xgb.csv --output baseline_tfidf_scores.csv --full-pool-fit
"""

import argparse
import csv
import json
import pathlib
import re
from collections import defaultdict
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from label_utils import load_true_labels
from llm_eval_metrics import parse_header_user_date_from_narrative


def load_blocks(path: pathlib.Path, max_samples: int | None = None) -> List[str]:
    """Same block format as llm_evaluator (separators of 80 '=')."""
    blocks: List[str] = []
    current_lines: List[str] = []
    line_no = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line_no += 1
            if line_no % 100000 == 0:
                print(f"  ... reading narrative file: {line_no:,} lines, {len(blocks):,} blocks", flush=True)
            if line.strip() == "=" * 80:
                current_lines.append(line)
                block = "".join(current_lines).strip()
                if block:
                    blocks.append(block)
                current_lines = []
                if max_samples is not None and len(blocks) >= max_samples:
                    break
            else:
                current_lines.append(line)
    if current_lines and (max_samples is None or len(blocks) < max_samples):
        block = "".join(current_lines).strip()
        if block:
            blocks.append(block)
    return blocks

# (regex, weight) — English narratives from stage2_narrative.py
_RULE_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (re.compile(r"cross[- ]department|mismatch|conflict|does not match", re.I), 0.14),
    (re.compile(r"usb|removable|device connect|device disconnect", re.I), 0.1),
    (re.compile(r"external (domain|email)|exfiltration|sensitive path|sensitive directory", re.I), 0.12),
    (re.compile(r"after[- ]hours|off[- ]hour|non[- ]?business", re.I), 0.08),
    (re.compile(r"unusual|first time|not seen before|first occurrence|spike|vs user avg", re.I), 0.1),
    (re.compile(r"abnormal attachment|large attachment|95th percentile", re.I), 0.08),
    (re.compile(r"\bhttp\b|url|web request|browse", re.I), 0.05),
    (re.compile(r"terminated|resign|departing", re.I), 0.06),
    (re.compile(r"encrypt|archive|\.7z|\.zip.*sensitive", re.I), 0.07),
]


def rule_score(text: str) -> float:
    s = 0.0
    for pat, w in _RULE_PATTERNS:
        if pat.search(text):
            s += w
    return float(min(1.0, s))


def narratives_to_labeled_rows(
    blocks: List[str],
    label_map: dict,
) -> Tuple[List[str], List[int], List[str], List[Tuple[str, str]]]:
    """One row per (user,date): longest block text if duplicates exist."""
    by_key: dict[Tuple[str, str], List[str]] = defaultdict(list)
    for block in blocks:
        key = parse_header_user_date_from_narrative(block)
        if key is None or key not in label_map:
            continue
        by_key[key].append(block)

    keys_sorted = sorted(by_key.keys())
    texts: List[str] = []
    labels: List[int] = []
    groups: List[str] = []
    keys: List[Tuple[str, str]] = []
    for key in keys_sorted:
        chunks = by_key[key]
        text = max(chunks, key=len)
        texts.append(text)
        labels.append(label_map[key])
        groups.append(key[0])
        keys.append(key)
    return texts, labels, groups, keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 baselines (rule / TF-IDF+LR)")
    parser.add_argument("--mode", type=str, choices=("rule", "tfidf_lr"), required=True)
    parser.add_argument("--narratives", type=str, default="stage2_narratives_xgb.txt")
    parser.add_argument("--suspicious", type=str, default="suspicious_sequences_xgb.csv")
    parser.add_argument("--output", type=str, default="baseline_scores.csv")
    parser.add_argument(
        "--full-pool-fit",
        action="store_true",
        help="TF-IDF+LR: train on all windows (for apple-to-apple with full-pool LLM eval; optimistic).",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="User held-out fraction for tfidf_lr")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--write-split",
        type=str,
        default="",
        help="tfidf_lr without --full-pool-fit: write JSON with train/test (user,date) keys for LLM aligned eval",
    )
    parser.add_argument(
        "--use-split-json",
        type=str,
        default="",
        help="tfidf_lr: load train/test keys from this JSON (same schema as --write-split) instead of a new GroupShuffleSplit.",
    )
    parser.add_argument(
        "--honest-fusion-csv",
        type=str,
        default="",
        help="tfidf_lr: after fit on train users, write TF-IDF probs for BOTH train and test windows to this path "
        "(for improve_stage2_scores; test scores are held-out, train scores are in-sample for TF-IDF).",
    )
    args = parser.parse_args()

    base = pathlib.Path(__file__).parent
    narr_path = (base / args.narratives).resolve()
    susp_path = (base / args.suspicious).resolve()
    out_path = (base / args.output).resolve()

    if not narr_path.exists():
        raise FileNotFoundError(narr_path)

    label_map = load_true_labels(susp_path)
    print(f"Loading narrative blocks from {narr_path.name} ...", flush=True)
    blocks = load_blocks(narr_path)
    texts, labels, groups, keys = narratives_to_labeled_rows(blocks, label_map)
    print(f"Loaded {len(texts)} unique (user,date) narratives with labels (from {len(blocks)} blocks).")

    if args.mode == "rule":
        with out_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file_user", "window_date", "score"])
            for text, key in zip(texts, keys):
                sc = rule_score(text)
                w.writerow([key[0], key[1], f"{sc:.6f}"])
        print(f"Wrote rule scores to {out_path}")
        print("Tune --threshold in eval_window_scores.py (typical max ~0.22; try 0.12–0.20).")
        return

    # tfidf_lr
    y = np.asarray(labels, dtype=int)
    group_arr = np.asarray(groups)

    clf = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=8000,
                    ngram_range=(1, 2),
                    min_df=2,
                    stop_words="english",
                ),
            ),
            (
                "lr",
                LogisticRegression(
                    max_iter=200,
                    class_weight="balanced",
                    random_state=args.random_state,
                ),
            ),
        ]
    )

    if args.full_pool_fit:
        clf.fit(texts, y)
        proba = clf.predict_proba(texts)[:, 1]
        with out_path.open("w", encoding="utf-8", newline="") as f:
            cw = csv.writer(f)
            cw.writerow(["file_user", "window_date", "score"])
            for key, p in zip(keys, proba):
                cw.writerow([key[0], key[1], f"{float(p):.8f}"])
        print(f"Wrote full-pool TF-IDF+LR scores to {out_path}")
        print("Note: scores are in-sample (optimistic). Compare to LLM on same eval_window_scores run.")
        return

    key_to_i = {keys[i]: i for i in range(len(keys))}

    if args.use_split_json:
        split_path = (base / args.use_split_json).resolve()
        if not split_path.exists():
            raise FileNotFoundError(split_path)
        with split_path.open("r", encoding="utf-8") as f:
            sp_data = json.load(f)
        tr_i = []
        for row in sp_data.get("train_keys") or []:
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                k = (str(row[0]).strip(), str(row[1]).strip()[:10])
                if k in key_to_i:
                    tr_i.append(key_to_i[k])
        te_i = []
        for row in sp_data.get("test_keys") or []:
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                k = (str(row[0]).strip(), str(row[1]).strip()[:10])
                if k in key_to_i:
                    te_i.append(key_to_i[k])
        tr_i = sorted(set(tr_i))
        te_i = sorted(set(te_i))
        n_miss_tr = len(sp_data.get("train_keys") or []) - len(tr_i)
        n_miss_te = len(sp_data.get("test_keys") or []) - len(te_i)
        if n_miss_tr or n_miss_te:
            print(
                f"WARN: --use-split-json: {n_miss_tr} train / {n_miss_te} test keys not found in narratives; skipped.",
                flush=True,
            )
        if len(tr_i) < 50:
            raise ValueError("Too few train windows after --use-split-json; check narratives vs split file.")
        print(f"Using split from {split_path.name}: {len(tr_i):,} train / {len(te_i):,} test indices.", flush=True)
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
        train_idx, test_idx = next(gss.split(texts, y, groups=group_arr))
        tr_i, te_i = sorted(train_idx.tolist()), sorted(test_idx.tolist())

    X_tr = [texts[i] for i in tr_i]
    X_te = [texts[i] for i in te_i]
    y_tr, y_te = y[tr_i], y[te_i]

    clf.fit(X_tr, y_tr)
    p_te = clf.predict_proba(X_te)[:, 1]
    pred = (p_te >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_te, pred, average="binary", zero_division=0
    )
    print(f"\nTF-IDF+LR user-held-out test ({len(te_i)} windows, ~{args.test_size:.0%} users):")
    print(f"  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}  (threshold=0.5 on prob)")

    if args.write_split and not args.use_split_json:
        split_path = (base / args.write_split).resolve()
        train_keys = [list(keys[i]) for i in tr_i]
        test_keys = [list(keys[i]) for i in te_i]
        payload = {
            "schema": "tfidf_user_group_shuffle_split_v1",
            "random_state": args.random_state,
            "test_size": args.test_size,
            "n_train_windows": len(train_keys),
            "n_test_windows": len(test_keys),
            "train_keys": train_keys,
            "test_keys": test_keys,
        }
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with split_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Wrote train/test key split to {split_path} (use with llm_evaluator.py --keys-json).", flush=True)

    if args.honest_fusion_csv:
        p_tr = clf.predict_proba(X_tr)[:, 1]
        fusion_path = (base / args.honest_fusion_csv).resolve()
        out_rows: List[Tuple[str, str, float]] = []
        for j, i in enumerate(tr_i):
            k = keys[i]
            out_rows.append((k[0], k[1], float(p_tr[j])))
        for j, i in enumerate(te_i):
            k = keys[i]
            out_rows.append((k[0], k[1], float(p_te[j])))
        out_rows.sort(key=lambda t: (t[0], t[1]))
        with fusion_path.open("w", encoding="utf-8", newline="") as f:
            cw = csv.writer(f)
            cw.writerow(["file_user", "window_date", "score"])
            for u, d, sc in out_rows:
                cw.writerow([u, d, f"{sc:.8f}"])
        print(
            f"Wrote honest fusion TF-IDF scores ({len(out_rows):,} rows = train+test) to {fusion_path}",
            flush=True,
        )

    with out_path.open("w", encoding="utf-8", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["file_user", "window_date", "score"])
        for i, p in zip(te_i, p_te):
            k = keys[i]
            cw.writerow([k[0], k[1], f"{float(p):.8f}"])
    print(f"Wrote TEST-ONLY scores ({len(te_i)} rows) to {out_path}")
    print("Eval: python eval_window_scores.py --scores <this file> --threshold 0.5")


if __name__ == "__main__":
    main()
