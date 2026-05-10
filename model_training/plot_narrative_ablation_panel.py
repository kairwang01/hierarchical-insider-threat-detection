#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine multiple `stage2_metrics_summary*.csv` files (from plot_stage2_comparison.py)
into one figure: one subplot per narrative / ablation setting, same methods compared.

Typical workflow (model_training/):

  1) Full narrative → plot → figures_full/stage2_metrics_summary.csv
  2) stage2_narrative.py --no-ldap → … → plot → figures_no_ldap/...
  3) python plot_narrative_ablation_panel.py \\
       --run "Full=figures_full/stage2_metrics_summary.csv" \\
       --run "No LDAP=figures_no_ldap/stage2_metrics_summary.csv" \\
       --run "No history=figures_no_hist/stage2_metrics_summary.csv" \\
       --output figures/ablation_narrative_panel.png

Methods are matched by exact `method` column first, then prefix match (e.g. "Fused" matches
"Fused (LLM+TF-IDF)").
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_summary_rows(path: pathlib.Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def find_metric_row(rows: List[Dict[str, str]], method_query: str) -> Optional[Dict[str, str]]:
    for r in rows:
        if r.get("method", "") == method_query:
            return r
    for r in rows:
        m = r.get("method", "")
        if m.startswith(method_query) or method_query in m:
            return r
    return None


def parse_run_arg(s: str) -> Tuple[str, pathlib.Path]:
    if "=" not in s:
        raise ValueError(f"Expected LABEL=path, got: {s}")
    label, _, path = s.partition("=")
    label, path = label.strip(), path.strip()
    if not label or not path:
        raise ValueError(f"Expected LABEL=path, got: {s}")
    return label, pathlib.Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Panel plot: narrative ablation Stage-2 metrics.")
    parser.add_argument(
        "--run",
        action="append",
        metavar="LABEL=CSV",
        required=True,
        help="Repeat per variant; CSV is stage2_metrics_summary*.csv from plot_stage2_comparison.py",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["Rule", "TF-IDF", "LLM", "Fused"],
        help="Match `method` column (exact or prefix / substring for Fused*)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/ablation_narrative_panel.png",
        help="Output PNG path (relative to cwd or absolute).",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default="",
        help="Optional: write long-form metrics CSV for reporting",
    )
    args = parser.parse_args()

    runs: List[Tuple[str, pathlib.Path]] = []
    for raw in args.run:
        label, pth = parse_run_arg(raw)
        if not pth.is_absolute():
            pth = pathlib.Path.cwd() / pth
        if not pth.exists():
            print(f"ERROR: not found {pth}", file=sys.stderr)
            sys.exit(1)
        runs.append((label, pth.resolve()))

    base = pathlib.Path(__file__).parent
    out_png = pathlib.Path(args.output)
    if not out_png.is_absolute():
        out_png = (base / out_png).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)

    n = len(runs)
    fig_w = max(10, 3.8 * n)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.8), sharey=True)
    if n == 1:
        axes_list = [axes]
    else:
        axes_list = list(np.atleast_1d(axes).ravel())

    long_rows: List[Dict[str, object]] = []

    for ax, (variant_label, csv_path) in zip(axes_list, runs):
        rows = load_summary_rows(csv_path)
        labels_m: List[str] = []
        prec_l: List[float] = []
        rec_l: List[float] = []
        f1_l: List[float] = []

        for mq in args.methods:
            hit = find_metric_row(rows, mq)
            if hit is None:
                continue
            try:
                p = float(hit["precision"])
                r_ = float(hit["recall"])
                f1 = float(hit["f1"])
            except (KeyError, TypeError, ValueError):
                continue
            disp = hit.get("method", mq)
            if len(disp) > 22:
                disp = disp[:19] + "…"
            labels_m.append(disp)
            prec_l.append(p)
            rec_l.append(r_)
            f1_l.append(f1)
            long_rows.append(
                {
                    "variant": variant_label,
                    "method": hit.get("method", mq),
                    "precision": p,
                    "recall": r_,
                    "f1": f1,
                    "source_csv": str(csv_path),
                }
            )

        if not labels_m:
            ax.set_visible(False)
            continue

        x = np.arange(len(labels_m))
        w = 0.25
        ax.bar(x - w, prec_l, w, label="Precision")
        ax.bar(x, rec_l, w, label="Recall")
        ax.bar(x + w, f1_l, w, label="F1")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_m, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(variant_label, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "Narrative ablation — Stage 2 metrics (same eval protocol per CSV)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}", flush=True)

    if args.csv_out:
        csv_path = pathlib.Path(args.csv_out)
        if not csv_path.is_absolute():
            csv_path = (base / csv_path).resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["variant", "method", "precision", "recall", "f1", "source_csv"]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in long_rows:
                w.writerow(row)
        print(f"Wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
