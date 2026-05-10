#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified data cleaning for the main pipeline.
All log/table processing should run cleaning first, then aggregate or label.
- Standardize timestamps to datetime (with invalid -> NaT for dropping).
- Drop invalid/missing critical fields, drop duplicates, sort by time.
"""

import re
import pandas as pd
import numpy as np
from datetime import datetime


# Formats tried in order (same as label_extraction for consistency)
_TIMESTAMP_FORMATS = [
    '%m/%d/%Y %H:%M:%S',
    '%d/%m/%Y %H:%M:%S',
    '%m/%d/%Y %H:%M',
    '%d/%m/%Y %H:%M',
    '%Y-%m-%d %H:%M:%S',
    '%Y-%m-%d %H:%M',
    '%m/%d/%Y',
    '%d/%m/%Y',
]


def parse_timestamp_safe(date_str):
    """
    Parse timestamp string to datetime. Returns pd.NaT on failure (for batch cleaning).
    Supports multiple formats; strips whitespace and handles missing month (e.g. /21/2011).
    """
    if pd.isna(date_str) or date_str is None or str(date_str).strip() == '':
        return pd.NaT
    s = str(date_str).strip()
    if s.startswith('/'):
        parts = s.split('/')
        if len(parts) >= 3 and not parts[0]:
            s = '01' + s
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return pd.NaT


def parse_timestamp_strict(date_str):
    """
    Parse timestamp string to datetime. Raises ValueError on failure.
    Use for insiders / small tables where we want to fail fast.
    """
    result = parse_timestamp_safe(date_str)
    if pd.isna(result):
        raise ValueError(f"Unable to parse date string: {date_str!r}")
    return result


def parse_timestamp_series(series):
    """
    Vectorized parse of a date column: fast pandas path first, then row-wise
    fallback for stubborn strings (same formats as parse_timestamp_safe).
    """
    s = series.copy()
    if not isinstance(s, pd.Series):
        s = pd.Series(s, dtype=object)
    s = s.astype(str).str.strip()
    s = s.replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA, '<NA>': pd.NA})
    dt = pd.to_datetime(s, errors='coerce', utc=False)
    need = dt.isna() & s.notna()
    if need.any():
        dt = dt.copy()
        slow = s.loc[need].apply(parse_timestamp_safe)
        dt.loc[need] = pd.to_datetime(slow, errors='coerce')
    return dt


def normalize_user_id(u):
    """Single user id: strip, collapse whitespace; None if missing/invalid."""
    if u is None:
        return None
    try:
        if isinstance(u, float) and np.isnan(u):
            return None
    except (TypeError, ValueError):
        pass
    try:
        if pd.isna(u):
            return None
    except (ValueError, TypeError):
        pass
    s = str(u).strip()
    s = re.sub(r'\s+', ' ', s)
    if not s or s.lower() in ('nan', 'none', '<na>'):
        return None
    return s


def _normalize_user_series(series):
    """Strip, collapse internal whitespace, drop string 'nan'/empty."""
    u = series.astype(str).str.strip()
    u = u.str.replace(r'\s+', ' ', regex=True)
    u = u.replace({'nan': pd.NA, 'None': pd.NA, '': pd.NA})
    return u


def clean_dataframe(df, date_col, user_col, datetime_col='datetime',
                    drop_invalid_date=True, drop_missing_user=True,
                    drop_duplicates=True, sort_by_date=True):
    """
    Unified cleaning: parse date, drop invalid/missing, drop dupes, sort.
    Ensures data consistency before any aggregation or labeling.

    Parameters
    ----------
    df : pandas.DataFrame
    date_col : str
        Column name for raw date string (e.g. 'date', 'file_date').
    user_col : str
        Column name for user id (e.g. 'user', 'file_user').
    datetime_col : str
        Name of the parsed datetime column to add (default 'datetime').
    drop_invalid_date : bool
        Drop rows where date cannot be parsed.
    drop_missing_user : bool
        Drop rows where user_col is null/empty.
    drop_duplicates : bool
        Drop duplicate rows.
    sort_by_date : bool
        Sort by datetime ascending.

    Returns
    -------
    pandas.DataFrame
        Copy of df with cleaning applied; includes new column datetime_col.
    """
    df = df.copy()
    original_len = len(df)

    if date_col not in df.columns:
        raise ValueError(f"date_col '{date_col}' not in DataFrame: {list(df.columns)}")
    if user_col not in df.columns:
        raise ValueError(f"user_col '{user_col}' not in DataFrame: {list(df.columns)}")

    # Parse timestamps (invalid -> NaT); vectorized + rare fallback
    df[datetime_col] = parse_timestamp_series(df[date_col])

    # Normalize user ids (trim, collapse spaces) for cleaner joins
    df[user_col] = _normalize_user_series(df[user_col])

    if drop_invalid_date:
        before = len(df)
        df = df[df[datetime_col].notna()].copy()
        dropped_dt = before - len(df)
        if dropped_dt > 0:
            print(f"  [clean] Dropped {dropped_dt} rows with invalid/unparseable date")

    if drop_missing_user:
        before = len(df)
        bad_u = df[user_col].isna() | (df[user_col].astype(str).str.strip() == '')
        df = df[~bad_u].copy()
        dropped_user = before - len(df)
        if dropped_user > 0:
            print(f"  [clean] Dropped {dropped_user} rows with missing user")

    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates().copy()
        dropped_dup = before - len(df)
        if dropped_dup > 0:
            print(f"  [clean] Dropped {dropped_dup} duplicate rows")

    if sort_by_date:
        df = df.sort_values(datetime_col).reset_index(drop=True)

    if original_len > len(df):
        print(f"  [clean] Total: {original_len:,} -> {len(df):,} records")
    return df


def clean_integrated_logs_df(df, date_col='file_date', user_col='file_user',
                             logon_date_col='logon_date'):
    """
    Cleaning for integrated_logs-style DataFrame (has file_* and optionally logon_*).
    Converts file_date (and logon_date if present) to datetime, drops invalid/missing, dupes, sorts.
    """
    df = df.copy()
    original_len = len(df)

    # Standardize main date (vectorized + same fallback as raw logs when still string-like)
    if date_col in df.columns:
        if df[date_col].dtype == object or str(df[date_col].dtype) == 'string':
            df[date_col] = parse_timestamp_series(df[date_col])
        else:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        invalid = df[date_col].isna().sum()
        if invalid > 0:
            print(f"  [clean] Dropping {invalid} rows with invalid {date_col}")
            df = df[df[date_col].notna()].copy()
    if user_col in df.columns:
        df[user_col] = _normalize_user_series(df[user_col])
        bad = df[user_col].isna() | (df[user_col].astype(str).str.strip() == '')
        dropped = bad.sum()
        if dropped > 0:
            print(f"  [clean] Dropping {dropped} rows with missing/invalid {user_col}")
            df = df[~bad].copy()
    if logon_date_col and logon_date_col in df.columns:
        if df[logon_date_col].dtype == object or str(df[logon_date_col].dtype) == 'string':
            df[logon_date_col] = parse_timestamp_series(df[logon_date_col])
        else:
            df[logon_date_col] = pd.to_datetime(df[logon_date_col], errors='coerce')

    dupes = df.duplicated().sum()
    if dupes > 0:
        print(f"  [clean] Dropping {dupes} duplicate rows")
        df = df.drop_duplicates().copy()
    df = df.sort_values(date_col).reset_index(drop=True)
    if original_len > len(df):
        print(f"  [clean] Total: {original_len:,} -> {len(df):,} records")
    return df
