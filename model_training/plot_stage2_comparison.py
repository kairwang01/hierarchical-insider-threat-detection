#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual comparison and threshold sweep for Stage 2 methods (Rule, TF-IDF, LLM) vs Stage1 pool baseline.

Reads the same (user, date) labels as eval_window_scores / ablation_compare. Optionally restricts
evaluation to keys from stage2_baselines --write-split JSON (e.g. TF-IDF test users) so LLM and
TF-IDF are compared on the same windows.

Outputs (under --out-dir; append --out-suffix or auto-suffix when --eval-mode ≠ best_f1):
  - stage2_metrics_summary*.csv — per-method threshold + confusion counts
  - stage2_metrics_table*.png — metrics table
  - stage2_confusion_matrices*.png — 2×2 confusion matrix per method
  - stage2_pr_f1_bar*.png — grouped bars: Precision / Recall / F1
  - stage2_pr_curves*.png — PR curves (sklearn)

Threshold selection (--eval-mode):
  - best_f1 (default): maximize F1 over thresholds
  - fixed_recall + --constraint-value: recall ≥ target, then maximize precision (tie-break F1)
  - fixed_precision + --constraint-value: precision ≥ target, then maximize recall
  - fixed_threshold + --constraint-value: use that threshold for all score methods

Combine narrative ablation summaries: `plot_narrative_ablation_panel.py`

Usage (from model_training/):

    python plot_stage2_comparison.py --suspicious suspicious_sequences_xgb.csv \\
        --rule-csv baseline_rule_scores.csv --tfidf-csv baseline_tfidf_full_scores.csv \\
        --llm-jsonl llm_predictions_xgb.jsonl --out-dir figures

    # With logistic fused scores (improve_stage2_scores.py); use --fused-label to match features:
    python plot_stage2_comparison.py ... --fused-csv stage2_fused_with_tfidf.csv \\
        --fused-label "Fused (LLM+S1+TF-IDF)"

    # Same windows as TF-IDF user-held-out test:
    python plot_stage2_comparison.py --suspicious suspicious_sequences_xgb.csv \\
        --keys-json tfidf_eval_split.json --keys-subset test \\
        --tfidf-csv baseline_tfidf_test_scores.csv --llm-jsonl llm_test.jsonl --out-dir figures
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve

from label_utils import load_true_labels
from llm_eval_metrics import load_llm_risk_by_key


def load_scores_csv(path: pathlib.Path) -> Dict[Tuple[str, str], float]:
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
            key = (u, d)
            scores[key] = max(scores.get(key, float("-inf")), s)
    return scores


def load_eval_keys_from_split(path: pathlib.Path, subset: str) -> List[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    key = "test_keys" if subset == "test" else "train_keys"
    raw = data.get(key) or []
    out: List[Tuple[str, str]] = []
    for row in raw:
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            out.append((str(row[0]).strip(), str(row[1]).strip()[:10]))
    return out


def confusion(
    label_map: Dict[Tuple[str, str], int],
    eval_keys: List[Tuple[str, str]],
    scores: Optional[Dict[Tuple[str, str], float]],
    threshold: float,
    pred_all_one: bool,
    default_score: float = 0.0,
) -> Tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for k in eval_keys:
        if k not in label_map:
            continue
        y = label_map[k]
        if pred_all_one:
            pred = 1
        else:
            s = default_score if scores is None else scores.get(k)
            if s is None:
                s = default_score
            pred = 1 if float(s) >= threshold else 0
        if y == 1 and pred == 1:
            tp += 1
        elif y == 0 and pred == 1:
            fp += 1
        elif y == 0 and pred == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def prf1(tp: int, fp: int, tn: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def _pr_ap_for_scores(
    label_map: Dict[Tuple[str, str], int],
    keys: List[Tuple[str, str]],
    scores: Dict[Tuple[str, str], float],
) -> float:
    y = np.array([label_map[k] for k in keys], dtype=int)
    s = np.array([float(scores.get(k, 0.0)) for k in keys], dtype=float)
    return float(average_precision_score(y, s)) if y.max() > 0 and y.min() < 1 else 0.0


def best_f1_sweep(
    label_map: Dict[Tuple[str, str], int],
    eval_keys: List[Tuple[str, str]],
    scores: Dict[Tuple[str, str], float],
    n_steps: int = 101,
) -> Tuple[float, float, float, float, float, Tuple[int, int, int, int]]:
    """Returns best_f1, best_p, best_r, best_threshold, pr_auc_ap, (tp,fp,tn,fn)."""
    keys = [k for k in eval_keys if k in label_map]
    ap = _pr_ap_for_scores(label_map, keys, scores)

    best_f1 = -1.0
    best_t = 0.5
    best_counts = (0, 0, 0, 0)
    best_p = best_r = 0.0
    for t in np.linspace(0.0, 1.0, n_steps):
        tp, fp, tn, fn = confusion(label_map, keys, scores, t, pred_all_one=False, default_score=0.0)
        p, r, f1 = prf1(tp, fp, tn, fn)
        if f1 > best_f1:
            best_f1, best_t, best_p, best_r = f1, t, p, r
            best_counts = (tp, fp, tn, fn)
    return best_f1, best_p, best_r, best_t, ap, best_counts


def threshold_sweep_constrained(
    label_map: Dict[Tuple[str, str], int],
    eval_keys: List[Tuple[str, str]],
    scores: Dict[Tuple[str, str], float],
    mode: str,
    constraint_value: float,
    n_steps: int = 401,
) -> Tuple[float, float, float, float, float, Tuple[int, int, int, int]]:
    """
    Pick one threshold under a fixed constraint (same eval_keys as best_f1_sweep).
    Modes:
      fixed_recall — among thresholds with recall >= constraint_value, maximize precision (tie-break F1);
                   if none, pick threshold minimizing |recall - constraint_value|.
      fixed_precision — among thresholds with precision >= constraint_value, maximize recall (tie-break F1);
                       if none, pick threshold minimizing |precision - constraint_value|.
      fixed_threshold — use constraint_value as threshold directly.
    Returns f1, p, r, threshold, pr_auc_ap, (tp,fp,tn,fn).
    """
    keys = [k for k in eval_keys if k in label_map]
    ap = _pr_ap_for_scores(label_map, keys, scores)

    if mode == "fixed_threshold":
        t = float(constraint_value)
        tp, fp, tn, fn = confusion(label_map, keys, scores, t, pred_all_one=False, default_score=0.0)
        p, r, f1 = prf1(tp, fp, tn, fn)
        return f1, p, r, t, ap, (tp, fp, tn, fn)

    rows: List[Tuple[float, float, float, float, int, int, int, int]] = []
    for t in np.linspace(0.0, 1.0, n_steps):
        tp, fp, tn, fn = confusion(label_map, keys, scores, t, pred_all_one=False, default_score=0.0)
        p, r, f1 = prf1(tp, fp, tn, fn)
        rows.append((t, p, r, f1, tp, fp, tn, fn))

    target = float(constraint_value)
    if mode == "fixed_recall":
        feasible = [row for row in rows if row[2] + 1e-9 >= target]
        if feasible:
            # max precision, then max f1, then higher threshold (fewer positives)
            best = max(feasible, key=lambda x: (x[1], x[3], x[0]))
        else:
            best = min(rows, key=lambda x: abs(x[2] - target))
    elif mode == "fixed_precision":
        feasible = [row for row in rows if row[1] + 1e-9 >= target]
        if feasible:
            best = max(feasible, key=lambda x: (x[2], x[3], -x[0]))
        else:
            best = min(rows, key=lambda x: abs(x[1] - target))
    else:
        raise ValueError(f"unknown mode {mode}")

    t, p, r, f1, tp, fp, tn, fn = best
    return f1, p, r, t, ap, (tp, fp, tn, fn)


def choose_threshold_for_method(
    label_map: Dict[Tuple[str, str], int],
    eval_keys: List[Tuple[str, str]],
    scores: Dict[Tuple[str, str], float],
    eval_mode: str,
    constraint_value: float | None,
) -> Tuple[float, float, float, float, float, Tuple[int, int, int, int]]:
    if eval_mode == "best_f1":
        return best_f1_sweep(label_map, eval_keys, scores)
    if constraint_value is None:
        raise ValueError("constraint_value required for fixed_* eval modes")
    return threshold_sweep_constrained(
        label_map, eval_keys, scores, eval_mode, float(constraint_value)
    )


def plot_bars(rows: List[dict], out_path: pathlib.Path, *, subtitle: str) -> None:
    methods = [r["method"] for r in rows]
    x = np.arange(len(methods))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.2), 5))
    ax.bar(x - w, [r["precision"] for r in rows], w, label="Precision")
    ax.bar(x, [r["recall"] for r in rows], w, label="Recall")
    ax.bar(x + w, [r["f1"] for r in rows], w, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title(f"Stage 2 comparison ({subtitle})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _fmt_metric(v: object) -> str:
    if v == "N/A" or v is None:
        return "—"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.4f}"
    return str(v)


def plot_metrics_table(rows: List[dict], out_path: pathlib.Path, *, subtitle: str, thr_col: str = "Thr") -> None:
    """Render summary_rows as a matplotlib table (Method, threshold, P/R/F1, PR-AUC, counts)."""
    if not rows:
        return
    headers = ["Method", thr_col, "Precision", "Recall", "F1", "PR-AUC", "TP", "FP", "TN", "FN"]
    cell_text: List[List[str]] = []
    for r in rows:
        cell_text.append(
            [
                str(r.get("method", "")),
                _fmt_metric(r.get("threshold_best_f1")),
                _fmt_metric(r.get("precision")),
                _fmt_metric(r.get("recall")),
                _fmt_metric(r.get("f1")),
                _fmt_metric(r.get("pr_auc_ap")),
                _fmt_metric(r.get("tp")),
                _fmt_metric(r.get("fp")),
                _fmt_metric(r.get("tn")),
                _fmt_metric(r.get("fn")),
            ]
        )
    fig_w = min(18, 10 + 0.12 * max(len(str(c[0])) for c in cell_text))
    fig_h = max(2.5, 1.0 + 0.38 * len(rows))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(f"Stage 2 metrics ({subtitle})", fontsize=12, pad=12)
    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.05, 1.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(rows: List[dict], out_path: pathlib.Path, *, subtitle: str) -> None:
    """
    One 2×2 heatmap per method: rows = true label (0,1), cols = predicted (0,1),
    cells [[TN, FP], [FN, TP]].
    """
    if not rows:
        return
    n = len(rows)
    ncols = min(3, max(1, n))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols + 0.5, 3.4 * nrows + 0.8))
    if n == 1:
        axes_list = [axes]
    else:
        axes_list = np.atleast_1d(axes).ravel().tolist()

    for i, r in enumerate(rows):
        ax = axes_list[i]
        tp, fp, tn, fn = int(r["tp"]), int(r["fp"]), int(r["tn"]), int(r["fn"])
        cm = np.array([[tn, fp], [fn, tp]], dtype=float)
        local_max = max(float(cm.max()), 1.0)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=local_max)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0 (neg)", "Pred 1 (pos)"])
        ax.set_yticklabels(["True 0 (neg)", "True 1 (pos)"])
        for (yi, xi), val in np.ndenumerate(cm):
            txt_color = "white" if val > local_max * 0.55 and local_max > 0 else "black"
            ax.text(xi, yi, f"{int(val)}", ha="center", va="center", fontsize=11, color=txt_color)
        title = str(r.get("method", ""))
        if len(title) > 42:
            title = title[:39] + "..."
        ax.set_title(title, fontsize=9)

    for j in range(n, len(axes_list)):
        axes_list[j].set_visible(False)

    fig.suptitle(
        f"Confusion matrices — {subtitle}; Stage1_pool = predict all positive",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(
    series: List[Tuple[str, np.ndarray, np.ndarray]],
    out_path: pathlib.Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, y, s in series:
        if y.max() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y, s)
        a = auc(rec, prec)
        ax.plot(rec, prec, label=f"{name} (AUC={a:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall curves (same eval windows)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Stage 2 method comparison + PR curves.")
    parser.add_argument("--suspicious", type=str, default="suspicious_sequences_xgb.csv")
    parser.add_argument(
        "--keys-json",
        type=str,
        default="",
        help="Optional split JSON (stage2_baselines --write-split); limits eval windows.",
    )
    parser.add_argument("--keys-subset", choices=("test", "train"), default="test")
    parser.add_argument("--rule-csv", type=str, default="")
    parser.add_argument("--tfidf-csv", type=str, default="")
    parser.add_argument("--llm-jsonl", type=str, default="")
    parser.add_argument(
        "--fused-csv",
        type=str,
        default="",
        help="Optional: calibrated fused scores from improve_stage2_scores.py",
    )
    parser.add_argument(
        "--fused-label",
        type=str,
        default="LLM+S1+calib",
        help="Legend name for --fused-csv (describe which features were fused, e.g. Fused (LLM+S1+TF-IDF))",
    )
    parser.add_argument(
        "--fused-extra-csv",
        type=str,
        default="",
        help="Optional second fused CSV (e.g. LR without TF-IDF vs with TF-IDF on same figure)",
    )
    parser.add_argument(
        "--fused-extra-label",
        type=str,
        default="Fused+TF-IDF",
        help="Legend name for --fused-extra-csv",
    )
    parser.add_argument("--out-dir", type=str, default="figures")
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=("best_f1", "fixed_recall", "fixed_precision", "fixed_threshold"),
        default="best_f1",
        help="How to pick the threshold for score-based methods (Rule/TF-IDF/LLM/Fused).",
    )
    parser.add_argument(
        "--constraint-value",
        type=float,
        default=None,
        help="Target for fixed_recall / fixed_precision, or threshold for fixed_threshold (required unless best_f1).",
    )
    parser.add_argument(
        "--out-suffix",
        type=str,
        default="",
        help="Append to output basenames (e.g. _fixed_r075). If empty and eval-mode≠best_f1, auto-suffix from mode+constraint.",
    )
    args = parser.parse_args()

    if args.eval_mode != "best_f1" and args.constraint_value is None:
        print("ERROR: --constraint-value is required when --eval-mode is not best_f1.", file=sys.stderr)
        sys.exit(1)

    out_suffix = args.out_suffix
    if not out_suffix and args.eval_mode != "best_f1":
        cv = str(args.constraint_value).replace(".", "p")
        out_suffix = f"_{args.eval_mode}_{cv}"

    if args.eval_mode == "best_f1":
        eval_subtitle = "score methods: threshold at best F1 on eval set"
        thr_col = "Thr@F1"
    elif args.eval_mode == "fixed_recall":
        eval_subtitle = f"score methods: fixed recall ≥ {args.constraint_value} (then max precision)"
        thr_col = "Thr"
    elif args.eval_mode == "fixed_precision":
        eval_subtitle = f"score methods: fixed precision ≥ {args.constraint_value} (then max recall)"
        thr_col = "Thr"
    else:
        eval_subtitle = f"score methods: fixed threshold = {args.constraint_value}"
        thr_col = "Thr"

    base = pathlib.Path(__file__).parent
    susp_path = (base / args.suspicious).resolve()
    if not susp_path.exists():
        print(f"ERROR: not found {susp_path}", file=sys.stderr)
        sys.exit(1)

    label_map = load_true_labels(susp_path)
    if args.keys_json:
        split_path = (base / args.keys_json).resolve()
        if not split_path.exists():
            print(f"ERROR: not found {split_path}", file=sys.stderr)
            sys.exit(1)
        eval_keys = load_eval_keys_from_split(split_path, args.keys_subset)
        eval_keys = [k for k in eval_keys if k in label_map]
        print(f"Evaluating on {len(eval_keys):,} windows from keys-json ({args.keys_subset}).", flush=True)
    else:
        eval_keys = sorted(label_map.keys())
        print(f"Evaluating on all {len(eval_keys):,} windows in suspicious CSV.", flush=True)

    if not eval_keys:
        print("ERROR: no evaluation windows after key filter.", file=sys.stderr)
        sys.exit(1)

    out_dir = (base / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[dict] = []
    pr_series: List[Tuple[str, np.ndarray, np.ndarray]] = []

    csv_fields = [
        "method",
        "threshold_best_f1",
        "precision",
        "recall",
        "f1",
        "pr_auc_ap",
        "tp",
        "fp",
        "tn",
        "fn",
    ]

    # Stage1 pool all-1
    tp, fp, tn, fn = confusion(label_map, eval_keys, None, 0.0, pred_all_one=True)
    p, r, f1 = prf1(tp, fp, tn, fn)
    summary_rows.append(
        {
            "method": "Stage1_pool_all_1",
            "threshold_best_f1": "N/A",
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "pr_auc_ap": "N/A",
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
    )

    keys_list = [k for k in eval_keys if k in label_map]
    y_arr = np.array([label_map[k] for k in keys_list], dtype=int)

    score_sources: List[Tuple[str, str, str]] = [
        ("Rule", args.rule_csv, "csv"),
        ("TF-IDF", args.tfidf_csv, "csv"),
        ("LLM", args.llm_jsonl, "jsonl"),
    ]
    if args.fused_csv:
        score_sources.append((args.fused_label, args.fused_csv, "csv"))
    if args.fused_extra_csv:
        score_sources.append((args.fused_extra_label, args.fused_extra_csv, "csv"))

    for name, path_arg, loader in score_sources:
        if not path_arg:
            continue
        pth = (base / path_arg).resolve()
        if not pth.exists():
            print(f"SKIP {name}: not found {pth}", flush=True)
            continue
        if loader == "csv":
            scores = load_scores_csv(pth)
        else:
            scores = load_llm_risk_by_key(pth, progress_every=0)
        bf1, bp, br, bt, ap, (tp, fp, tn, fn) = choose_threshold_for_method(
            label_map,
            eval_keys,
            scores,
            args.eval_mode,
            args.constraint_value,
        )
        summary_rows.append(
            {
                "method": name,
                "threshold_best_f1": round(float(bt), 4),
                "precision": round(bp, 4),
                "recall": round(br, 4),
                "f1": round(bf1, 4),
                "pr_auc_ap": round(float(ap), 4),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )
        s_arr = np.array([float(scores.get(k, 0.0)) for k in keys_list], dtype=float)
        pr_series.append((name, y_arr, s_arr))

    csv_path = out_dir / f"stage2_metrics_summary{out_suffix}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        if summary_rows:
            w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(summary_rows)
    print(f"Wrote {csv_path}", flush=True)

    plot_metrics_table(
        summary_rows,
        out_dir / f"stage2_metrics_table{out_suffix}.png",
        subtitle=eval_subtitle,
        thr_col=thr_col,
    )
    print(f"Wrote {out_dir / f'stage2_metrics_table{out_suffix}.png'}", flush=True)
    plot_confusion_matrices(
        summary_rows,
        out_dir / f"stage2_confusion_matrices{out_suffix}.png",
        subtitle=eval_subtitle,
    )
    print(f"Wrote {out_dir / f'stage2_confusion_matrices{out_suffix}.png'}", flush=True)

    # Bar chart: normalize keys for Stage1 row
    bar_rows = []
    for r in summary_rows:
        bar_rows.append(
            {
                "method": r["method"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
            }
        )
    plot_bars(bar_rows, out_dir / f"stage2_pr_f1_bar{out_suffix}.png", subtitle=eval_subtitle)
    print(f"Wrote {out_dir / f'stage2_pr_f1_bar{out_suffix}.png'}", flush=True)

    if pr_series:
        plot_pr_curves(pr_series, out_dir / f"stage2_pr_curves{out_suffix}.png")
        print(f"Wrote {out_dir / f'stage2_pr_curves{out_suffix}.png'}", flush=True)
    else:
        print("No score files provided — skipped PR curves.", flush=True)


if __name__ == "__main__":
    main()
