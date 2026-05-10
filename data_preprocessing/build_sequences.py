#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build behavior sequences: aggregate all logs by (user, time_window).
Multi-source fusion: logon, file, device, email.
Output: one row per (user, window) with event counts and sequence-level label.
Pipeline: load & clean 4 logs -> aggregate by window -> label sequences -> save.
"""

from __future__ import annotations

import argparse
import os
import sys

# Pandas can take 30s–several minutes to import on some Windows setups; say so immediately.
print("build_sequences: importing pandas/numpy (wait if this line stays alone for a while)...", flush=True)
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_cleaning import clean_dataframe


_LOGON_COLS = ('id', 'date', 'user', 'pc', 'activity')


def _read_csv_fast(path, usecols=None):
    """Read CSV with C engine and stable typing (faster on large files)."""
    kw = dict(low_memory=False, engine='c')
    if usecols is not None:
        try:
            return pd.read_csv(path, usecols=list(usecols), **kw)
        except (ValueError, TypeError, OSError):
            pass
    return pd.read_csv(path, **kw)


def load_and_clean_logon(path):
    """Load logon.csv, clean, keep Logon only, add date key."""
    print(f"  reading CSV: {path}", flush=True)
    df = _read_csv_fast(path, usecols=_LOGON_COLS)
    print(f"  rows after read: {len(df):,}", flush=True)
    df = clean_dataframe(df, date_col='date', user_col='user')
    df = df[df['activity'] == 'Logon'].copy()
    df['date_key'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    return df


_FILE_COLS = ('id', 'date', 'user', 'pc', 'filename', 'content')


def load_and_clean_file(path):
    """Load file.csv, clean, add date key, sensitive extension and sensitive-path heuristic."""
    print(f"  reading CSV: {path}", flush=True)
    df = _read_csv_fast(path, usecols=_FILE_COLS)
    print(f"  rows after read: {len(df):,}", flush=True)
    df = clean_dataframe(df, date_col='date', user_col='user')
    df['date_key'] = df['datetime'].dt.date
    # Sensitive extension (doc, pdf, xls, ppt, etc.)
    ext = df['filename'].astype(str).str.extract(r'\.(\w+)$', expand=False)
    df['is_sensitive_ext'] = ext.str.lower().isin(['doc', 'pdf', 'xls', 'ppt', 'docx', 'xlsx', 'pptx']).astype(int)
    # Sensitive directory heuristic: C:\, backup, path-like (filename may contain path in some datasets)
    fn = df['filename'].astype(str).str.lower()
    df['is_sensitive_path'] = (
        fn.str.contains(r'backup', na=False) |
        fn.str.contains(r'c:\\|\\\\c:\\\\|/c/|^c:', regex=True, na=False) |
        fn.str.contains(r'\\\\', na=False)  # path separator
    ).astype(int)
    return df


_DEVICE_COLS = ('id', 'date', 'user', 'pc', 'activity')


def load_and_clean_device(path):
    """Load device.csv, clean, add date key."""
    print(f"  reading CSV: {path}", flush=True)
    df = _read_csv_fast(path, usecols=_DEVICE_COLS)
    print(f"  rows after read: {len(df):,}", flush=True)
    df = clean_dataframe(df, date_col='date', user_col='user')
    df['date_key'] = df['datetime'].dt.date
    return df


def load_and_clean_email(path, usecols=None, internal_domain_substr='dtaa'):
    """Load email.csv (without content by default), clean, add date key."""
    if usecols is None:
        # r4.2 email.csv: id,date,user,pc,to,cc,bcc,from,size,attachments,content
        usecols = ['id', 'date', 'user', 'pc', 'to', 'cc', 'bcc', 'from', 'size', 'attachments']
    # Only read columns that exist
    print(f"  reading CSV: {path}", flush=True)
    try:
        df = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False, engine='c')
    except Exception:
        df = pd.read_csv(path, nrows=0, low_memory=False, engine='c')
        valid = [c for c in usecols if c in df.columns]
        df = pd.read_csv(path, usecols=valid if valid else None, low_memory=False, engine='c')
    print(f"  rows after read: {len(df):,}", flush=True)
    if 'date' not in df.columns or 'user' not in df.columns:
        raise ValueError("email.csv must have 'date' and 'user' columns")
    df = clean_dataframe(df, date_col='date', user_col='user')
    df['date_key'] = df['datetime'].dt.date
    # Heuristic: external = to contains @ and not internal domain substring (e.g. dtaa)
    sub = (internal_domain_substr or '').lower()
    to_str = df['to'].astype(str) if 'to' in df.columns else pd.Series('', index=df.index)
    if sub:
        df['has_external'] = to_str.str.contains('@', na=False) & ~to_str.str.lower().str.contains(sub, na=False)
    else:
        df['has_external'] = to_str.str.contains('@', na=False)
    return df


def to_week_start(dt_series):
    """Return date of Monday of the week for each datetime."""
    return (pd.to_datetime(dt_series).dt.to_period('W').dt.start_time.dt.date)


def aggregate_by_window(logon_df, file_df, device_df, email_df, window='day'):
    """
    Aggregate events by (user, window_key). window in ('day', 'week').
    Returns DataFrame with one row per (user, window_key) and count columns.
    """
    _GB_KEYS = ['user', 'window_key']

    def agg_logon(df):
        if df.empty:
            return pd.DataFrame(columns=['user', 'window_key', 'n_logon', 'n_logon_after_hours'])
        df = df.copy()
        if window == 'week':
            df['window_key'] = to_week_start(df['datetime'])
        else:
            df['window_key'] = df['date_key']
        df['after_hrs'] = ((df['hour'] < 6) | (df['hour'] > 20)).astype(int)
        out = df.groupby(_GB_KEYS, sort=False, as_index=False).agg(
            n_logon=('after_hrs', 'count'),
            n_logon_after_hours=('after_hrs', 'sum'),
        )
        return out

    def agg_file(df):
        if df.empty:
            return pd.DataFrame(
                columns=['user', 'window_key', 'n_file', 'n_file_sensitive', 'n_file_sensitive_dir']
            )
        df = df.copy()
        if window == 'week':
            df['window_key'] = to_week_start(df['datetime'])
        else:
            df['window_key'] = df['date_key']
        agg_kw = {'n_file': ('user', 'count')}
        if 'is_sensitive_ext' in df.columns:
            agg_kw['n_file_sensitive'] = ('is_sensitive_ext', 'sum')
        if 'is_sensitive_path' in df.columns:
            agg_kw['n_file_sensitive_dir'] = ('is_sensitive_path', 'sum')
        out = df.groupby(_GB_KEYS, sort=False, as_index=False).agg(**agg_kw)
        if 'n_file_sensitive' not in out.columns:
            out['n_file_sensitive'] = 0
        if 'n_file_sensitive_dir' not in out.columns:
            out['n_file_sensitive_dir'] = 0
        out['n_file_sensitive'] = out['n_file_sensitive'].fillna(0)
        out['n_file_sensitive_dir'] = out['n_file_sensitive_dir'].fillna(0)
        return out

    def agg_device(df):
        if df.empty:
            return pd.DataFrame(
                columns=['user', 'window_key', 'n_device_connect', 'n_device_disconnect', 'n_device_total']
            )
        df = df.copy()
        if window == 'week':
            df['window_key'] = to_week_start(df['datetime'])
        else:
            df['window_key'] = df['date_key']
        df['n_connect'] = (df['activity'] == 'Connect').astype(int)
        df['n_disconnect'] = (df['activity'] == 'Disconnect').astype(int)
        out = df.groupby(_GB_KEYS, sort=False, as_index=False).agg(
            n_device_connect=('n_connect', 'sum'),
            n_device_disconnect=('n_disconnect', 'sum'),
        )
        out['n_device_total'] = out['n_device_connect'] + out['n_device_disconnect']
        return out

    def agg_email(df):
        if df.empty:
            return pd.DataFrame(
                columns=['user', 'window_key', 'n_email', 'n_email_external', 'n_email_abnormal_attachment']
            )
        df = df.copy()
        if window == 'week':
            df['window_key'] = to_week_start(df['datetime'])
        else:
            df['window_key'] = df['date_key']
        agg_kw = {'n_email': ('user', 'count')}
        if 'has_external' in df.columns:
            agg_kw['n_email_external'] = ('has_external', 'sum')
        if 'is_abnormal_attachment' in df.columns:
            agg_kw['n_email_abnormal_attachment'] = ('is_abnormal_attachment', 'sum')
        out = df.groupby(_GB_KEYS, sort=False, as_index=False).agg(**agg_kw)
        if 'n_email_external' not in out.columns:
            out['n_email_external'] = 0
        if 'n_email_abnormal_attachment' not in out.columns:
            out['n_email_abnormal_attachment'] = 0
        out['n_email_external'] = out['n_email_external'].fillna(0)
        out['n_email_abnormal_attachment'] = out['n_email_abnormal_attachment'].fillna(0)
        return out

    # Aggregate each source
    L = agg_logon(logon_df)
    F = agg_file(file_df)
    D = agg_device(device_df)
    E = agg_email(email_df)

    # Merge on (user, window_key) outer to keep all (user, window) pairs
    out = L.merge(F, on=['user', 'window_key'], how='outer')
    out = out.merge(D, on=['user', 'window_key'], how='outer')
    out = out.merge(E, on=['user', 'window_key'], how='outer')
    out = out.fillna(0)
    # Integer counts
    int_cols = ['n_logon', 'n_logon_after_hours', 'n_file', 'n_file_sensitive', 'n_file_sensitive_dir',
                'n_device_connect', 'n_device_disconnect', 'n_device_total',
                'n_email', 'n_email_external', 'n_email_abnormal_attachment']
    for c in int_cols:
        if c in out.columns:
            out[c] = out[c].astype(int)
    # Derived: 窗口内非工作时间登录占比
    out['logon_after_hours_ratio'] = np.where(
        out['n_logon'] > 0,
        out['n_logon_after_hours'].astype(float) / out['n_logon'],
        0.0
    )
    # 突发性特征：当前窗口与用户全局均值的比值，降低对单一强特征（如 is_terminated）的依赖
    for col in ['n_logon', 'n_file', 'n_device_total', 'n_email']:
        if col not in out.columns:
            continue
        user_mean = out.groupby('user')[col].transform('mean')
        out[f'{col}_vs_user_avg'] = np.where(user_mean > 0, out[col].astype(float) / (user_mean + 1e-6), 0.0)
    return out


def main():
    parser = argparse.ArgumentParser(
        description='Build behavior sequences: (user, time_window) with multi-source fusion (logon, file, device, email)'
    )
    parser.add_argument('--data-dir', '-d', type=str, default='../r4.2',
                       help='Directory containing logon.csv, file.csv, device.csv, email.csv')
    parser.add_argument('--window', '-w', type=str, default='day', choices=['day', 'week'],
                       help='Time window: day or week')
    parser.add_argument('--insiders', '-i', type=str, default='../answers/insiders.csv',
                       help='Path to insiders.csv for sequence-level labeling')
    parser.add_argument('--dataset', type=str, default='4.2',
                       help='Dataset filter for insiders (e.g. 4.2)')
    parser.add_argument('--output', '-o', type=str, default='../integrated_sequences_labeled.csv',
                       help='Output path for sequence table with labels')
    parser.add_argument('--output-features', type=str, default=None,
                       help='If set, also write features.csv for stage1 (one row per sequence)')
    parser.add_argument('--internal-domain', type=str, default='dtaa',
                       help='Lowercase substring for internal email domain (excluded from external count); empty disables')
    args = parser.parse_args()

    data_dir = os.path.normpath(args.data_dir)
    logon_path = os.path.join(data_dir, 'logon.csv')
    file_path = os.path.join(data_dir, 'file.csv')
    device_path = os.path.join(data_dir, 'device.csv')
    email_path = os.path.join(data_dir, 'email.csv')

    for p, name in [(logon_path, 'logon'), (file_path, 'file'), (device_path, 'device'), (email_path, 'email')]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {name} at {p}")

    print("Loading and cleaning logon...")
    logon_df = load_and_clean_logon(logon_path)
    print("Loading and cleaning file...")
    file_df = load_and_clean_file(file_path)
    print("Loading and cleaning device...")
    device_df = load_and_clean_device(device_path)
    print("Loading and cleaning email (no content column)...")
    email_df = load_and_clean_email(
        email_path,
        internal_domain_substr=args.internal_domain.strip().lower() if args.internal_domain else '',
    )
    # 附件/大小异常：size 或 attachments 超过全局 95 分位数则记为异常
    if 'size' in email_df.columns or 'attachments' in email_df.columns:
        sz = pd.to_numeric(email_df['size'], errors='coerce') if 'size' in email_df.columns else pd.Series(0, index=email_df.index)
        att = pd.to_numeric(email_df['attachments'], errors='coerce') if 'attachments' in email_df.columns else pd.Series(0, index=email_df.index)
        p95_s = sz.quantile(0.95) if sz.notna().any() and sz.max() > 0 else 0
        p95_a = att.quantile(0.95) if att.notna().any() and att.max() > 0 else 0
        email_df['is_abnormal_attachment'] = ((sz.fillna(0) > p95_s) | (att.fillna(0) > p95_a)).astype(int)
    else:
        email_df['is_abnormal_attachment'] = 0

    print(f"\nAggregating by (user, {args.window})...")
    seq = aggregate_by_window(logon_df, file_df, device_df, email_df, window=args.window)

    # Rename for pipeline compatibility: user -> file_user, window_key -> file_date
    seq = seq.rename(columns={'user': 'file_user', 'window_key': 'file_date'})
    seq['file_date'] = pd.to_datetime(seq['file_date'])

    print("Applying sequence-level labeling (user+window overlap with insiders)...")
    from label_extraction import label_sequence_table
    seq = label_sequence_table(seq, args.insiders, args.dataset,
                               user_col='file_user', window_date_col='file_date')

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    seq.to_csv(args.output, index=False)
    print(f"Saved {len(seq):,} sequences to {args.output}")
    print(f"  Malicious sequences: {seq['is_malicious'].sum():,}")

    if args.output_features:
        # features.csv for stage1: same columns; stage1 expects file_user, file_date, is_malicious, malicious_scenario + feature cols
        feat_cols = [c for c in seq.columns if c not in ('file_user', 'file_date', 'is_malicious', 'malicious_scenario')]
        out_f = seq[['file_user', 'file_date'] + feat_cols + ['is_malicious', 'malicious_scenario']].copy()
        out_f.to_csv(args.output_features, index=False)
        print(f"Saved features to {args.output_features} ({len(feat_cols)} feature columns)")


if __name__ == '__main__':
    main()
