#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared ground-truth loading for Stage 2 evaluation scripts."""

import csv
import pathlib
from typing import Any, Dict, Tuple


def file_date_to_date_only(raw: str) -> str:
    s = (raw or "").strip()
    if not s or s.lower() == "nan":
        return ""
    for sep in (" ", "T"):
        if sep in s:
            s = s.split(sep, 1)[0]
            break
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    return ""


def parse_binary_label(raw: Any) -> int:
    if raw is None:
        return 0
    s = str(raw).strip().lower()
    if s in ("", "nan", "none"):
        return 0
    if s in ("1", "true", "t", "yes"):
        return 1
    if s in ("0", "false", "f", "no"):
        return 0
    try:
        return 1 if int(float(s)) != 0 else 0
    except (TypeError, ValueError):
        return 0


def load_true_labels(suspicious_path: pathlib.Path) -> Dict[Tuple[str, str], int]:
    """
    (file_user, YYYY-MM-DD) -> 0/1. Malicious if any row in that window is malicious.
    """
    required_cols = {"file_user", "file_date", "is_malicious"}
    label_map: Dict[Tuple[str, str], int] = {}

    with suspicious_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{suspicious_path} has no header row")
        headers = {h.strip() for h in reader.fieldnames if h}
        if not required_cols.issubset(headers):
            raise ValueError(
                f"{suspicious_path} must contain columns {required_cols}, found {headers}"
            )

        for row in reader:
            row = {k.strip() if isinstance(k, str) else k: v for k, v in row.items()}
            user = (row.get("file_user") or "").strip()
            date_only = file_date_to_date_only(row.get("file_date") or "")
            if not user or not date_only:
                continue
            mal = parse_binary_label(row.get("is_malicious"))
            key = (user, date_only)
            label_map[key] = max(label_map.get(key, 0), mal)

    return label_map
