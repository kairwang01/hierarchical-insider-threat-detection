#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2: Graph-Augmented Narrative Builder

Given:
  - Suspicious sequences from Stage 1 (e.g. suspicious_sequences_xgb.csv)
  - Integrated logs with labels (integrated_logs_labeled.csv)
  - LDAP data (for role/department)

This script builds a KG-augmented narrative for each suspicious user+day window:
  - Turn structured logs into a background-rich narrative text
  - Inject department/role information from LDAP
  - Highlight conflicts/anomalies based on recent history, e.g.:
      * User's department does not match the resource type being accessed
      * No similar access observed in the last 30 days
"""

import os
import sys
import argparse
import time
from datetime import datetime, timedelta
from itertools import islice

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from ldap_helper import LDAPProcessor  # type: ignore


def parse_date_safe(s):
    try:
        return pd.to_datetime(s)
    except Exception:
        return pd.NaT


def _index_logs_by_user_and_day(logs_df, user_col, need_users):
    """
    One-time groupby to avoid O(n_windows * n_rows) repeated boolean masks.
    Returns (by_user, by_user_date) dicts mapping to DataFrame views/groups.
    """
    if logs_df.empty or user_col not in logs_df.columns or 'date_only' not in logs_df.columns:
        return {}, {}
    sub = logs_df[logs_df[user_col].isin(need_users)]
    if sub.empty:
        return {}, {}
    by_user = {u: g for u, g in sub.groupby(user_col, sort=False)}
    by_ud = {}
    for (u, d), g in sub.groupby([user_col, 'date_only'], sort=False):
        by_ud[(u, d)] = g
    return by_user, by_ud


def detect_resource_type(filename: str) -> str:
    """Heuristic: infer coarse resource type from filename/path."""
    name = str(filename).lower()
    if any(k in name for k in ['salary', 'salaries', 'payroll', 'compensation', 'bonus']):
        return 'financial'
    if any(k in name for k in ['hr', 'personnel', 'employee_records']):
        return 'hr'
    if any(k in name for k in ['secret', 'confidential', 'classified']):
        return 'highly_sensitive'
    if any(k in name for k in ['source', '.cs', '.py', '.java', 'repo', 'git']):
        return 'code'
    return 'general'


def is_cross_department_access(user_dept: str, resource_type: str) -> bool:
    """Simple conflict rule between user's department and resource type."""
    if not user_dept or not resource_type or resource_type == 'general':
        return False
    dept = user_dept.lower()
    if resource_type == 'financial' and 'financ' not in dept:
        return True
    if resource_type == 'hr' and 'human' not in dept and 'hr' not in dept:
        return True
    return False


def build_narrative_for_window(
    user_id,
    window_date,
    window_logs,
    ldap_proc,
    full_logs_30d,
    device_logs_day=None,
    email_logs_day=None,
    http_logs_day=None,
    *,
    use_ldap=True,
    use_history=True,
    use_cross_source=True,
):
    """
    Build narrative text for a single (user, date) window.

    window_logs: logs for this user on that day (subset of integrated_logs_labeled)
    full_logs_30d: all logs for this user over the last 30 days (excluding this day)
    device_logs_day/email_logs_day/http_logs_day: same-day logs from device/email/http sources
    use_ldap: if False, omit LDAP department/role and cross-department conflict notes from LDAP.
    use_history: if False, omit the 30-day historical comparison section.
    use_cross_source: if False, omit device/email/http timeline section.
    """
    lines = []
    dt = parse_date_safe(window_date)
    date_str = dt.strftime('%Y-%m-%d') if not pd.isna(dt) else str(window_date)

    role = None
    dept = None
    if use_ldap and ldap_proc is not None:
        user_info = None
        try:
            user_info = ldap_proc.get_user_info(user_id, target_date=dt if not pd.isna(dt) else None)
        except Exception:
            user_info = None
        role = user_info['role'] if user_info and 'role' in user_info else None
        dept = user_info['department'] if user_info and 'department' in user_info else None

    header = f"Summary of suspicious behavior for user {user_id} on {date_str}:"
    context_bits = []
    if use_ldap:
        if dept:
            context_bits.append(f"department: {dept}")
        if role:
            context_bits.append(f"role: {role}")
        if context_bits:
            header += " (" + ", ".join(context_bits) + ")"
    lines.append(header)

    # Key file/logon events on that day (limit to a few)
    if window_logs.empty:
        lines.append("No concrete log events were found for this day in the integrated logs.")
        return "\n".join(lines)

    window_logs = window_logs.sort_values('file_date')

    lines.append("")
    lines.append("Key events:")

    max_events = 10
    resource_conflicts = []

    for idx, row in window_logs.head(max_events).iterrows():
        ts = parse_date_safe(row.get('file_date'))
        ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(ts) else str(row.get('file_date'))
        pc = row.get('file_pc') or row.get('logon_pc') or 'UNKNOWN_HOST'
        filename = row.get('file_filename') or row.get('file_content') or 'UNKNOWN_FILE'
        activity = row.get('logon_activity') or 'FileOp'

        resource_type = detect_resource_type(filename)
        conflict_flags = []
        if use_ldap and is_cross_department_access(dept or '', resource_type):
            conflict_flags.append("user department and resource type are inconsistent")

        line = f"- At {ts_str}, user on host {pc} performed {activity} and accessed \"{filename}\"."
        if conflict_flags:
            line += " [NOTE: " + "; ".join(conflict_flags) + "]"
            resource_conflicts.append((filename, resource_type))
        lines.append(line)

    # Check if similar patterns exist over the last 30 days
    if not pd.isna(dt):
        start_30 = dt - timedelta(days=30)
        history = full_logs_30d[
            (full_logs_30d['file_user'] == user_id)
            & (full_logs_30d['file_date'] >= start_30)
            & (full_logs_30d['file_date'] < dt)
        ].copy()
    else:
        history = full_logs_30d[full_logs_30d['file_user'] == user_id].copy()

    if use_history:
        lines.append("")
        lines.append("Historical behavior comparison:")
        if history.empty:
            lines.append(f"- In the previous 30 days, no file access activity was observed for this user.")
        else:
            lines.append(f"- In the last 30 days, this user had {len(history):,} file-operation events.")
            if resource_conflicts:
                for fname, rtype in resource_conflicts:
                    hist_mask = history['file_filename'].astype(str).str.lower().str.contains(fname.split('.')[0].lower(), na=False)
                    if not hist_mask.any():
                        lines.append(f"- [NOTE] In the last 30 days, no prior access to resources similar to \"{fname}\" ({rtype}) was observed; this access is anomalous relative to the user's history.")
                    else:
                        lines.append(f"- For resources similar to \"{fname}\" ({rtype}), there were {hist_mask.sum()} accesses in the last 30 days; this access appears more like a continuation of an existing pattern.")

    # Cross-source chronological narrative: device, email, http
    timeline_events = []

    if use_cross_source and device_logs_day is not None and not device_logs_day.empty:
        for _, row in device_logs_day.iterrows():
            ts = parse_date_safe(row.get('date'))
            ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(ts) else str(row.get('date'))
            pc = row.get('pc') or 'UNKNOWN_PC'
            activity = row.get('activity') or 'device_event'
            timeline_events.append(
                (ts, f"{ts_str}: user {activity.lower()} a USB device on {pc}.")
            )

    if use_cross_source and email_logs_day is not None and not email_logs_day.empty:
        for _, row in email_logs_day.iterrows():
            ts = parse_date_safe(row.get('date'))
            ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(ts) else str(row.get('date'))
            pc = row.get('pc') or 'UNKNOWN_PC'
            to = row.get('to') or ''
            size = row.get('size')
            attachments = row.get('attachments')
            extra_bits = []
            if size is not None:
                extra_bits.append(f"size={size}")
            if attachments is not None:
                extra_bits.append(f"attachments={attachments}")
            extra_str = f" ({', '.join(extra_bits)})" if extra_bits else ""
            timeline_events.append(
                (ts, f"{ts_str}: user sent an email from {pc} to {to}{extra_str}.")
            )

    if use_cross_source and http_logs_day is not None and not http_logs_day.empty:
        for _, row in http_logs_day.iterrows():
            ts = parse_date_safe(row.get('date'))
            ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(ts) else str(row.get('date'))
            pc = row.get('pc') or 'UNKNOWN_PC'
            url = row.get('url') or 'UNKNOWN_URL'
            timeline_events.append(
                (ts, f"{ts_str}: user browsed URL {url} from {pc}.")
            )

    if timeline_events:
        # Keep events with valid timestamps first, then others
        timeline_events.sort(key=lambda x: (pd.isna(x[0]), x[0]))
        lines.append("")
        lines.append("Cross-source chronological narrative:")
        for _, text in timeline_events:
            lines.append(f"- {text}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Graph-Augmented Narrative Builder")
    parser.add_argument('--suspicious', type=str, default='suspicious_sequences_xgb.csv',
                        help='Suspicious sequences CSV from Stage1 (default: suspicious_sequences_xgb.csv)')
    parser.add_argument('--logs', type=str, default='../integrated_logs_labeled.csv',
                        help='Integrated logs with labels (default: ../integrated_logs_labeled.csv)')
    parser.add_argument('--ldap-dir', type=str, default='../r4.2/LDAP',
                        help='LDAP directory (default: ../r4.2/LDAP)')
    parser.add_argument('--raw-data-dir', type=str, default='../r4.2',
                        help='Directory containing raw log files (device.csv, email.csv, http.csv)')
    parser.add_argument('--output', type=str, default='stage2_narratives_xgb.txt',
                        help='Output narrative text file (default: stage2_narratives_xgb.txt)')
    parser.add_argument('--max-users', type=int, default=None,
                        help='Optionally limit number of users/sequences for quick testing')
    parser.add_argument(
        '--progress-every',
        type=int,
        default=100,
        help='Print progress every N user-date windows (0 disables; default: 50)',
    )
    parser.add_argument(
        '--no-ldap',
        action='store_true',
        help='Ablation: omit LDAP department/role and LDAP-based cross-department conflict notes.',
    )
    parser.add_argument(
        '--no-history',
        action='store_true',
        help='Ablation: omit the 30-day historical behavior comparison section.',
    )
    parser.add_argument(
        '--no-cross-source',
        action='store_true',
        help='Ablation: omit device/email/http timeline; skips loading raw device.csv, email.csv, http.csv (much faster).',
    )

    args = parser.parse_args()

    if not os.path.exists(args.suspicious):
        raise FileNotFoundError(f"Suspicious file not found: {args.suspicious}")
    if not os.path.exists(args.logs):
        raise FileNotFoundError(f"Integrated logs file not found: {args.logs}")

    print(f"Loading suspicious sequences from {args.suspicious}...")
    sus_df = pd.read_csv(args.suspicious, low_memory=False, engine='c')
    print(f"  Loaded {len(sus_df):,} suspicious sequences")

    print(f"Loading integrated logs from {args.logs}...")
    logs_df = pd.read_csv(args.logs, low_memory=False, engine='c')
    print(f"  Loaded {len(logs_df):,} log records", flush=True)

    # Ensure file_date is datetime (can take a few seconds on ~450k rows)
    print("Parsing integrated log timestamps (file_date)...", flush=True)
    logs_df['file_date'] = pd.to_datetime(logs_df['file_date'], errors='coerce')
    print("  file_date parsed.", flush=True)

    raw_dir = os.path.normpath(args.raw_data_dir)
    device_df = pd.DataFrame()
    email_df = pd.DataFrame()
    http_df = pd.DataFrame()

    if args.no_cross_source:
        print("Skipping raw device/email/http load (--no-cross-source).", flush=True)
    else:
        device_path = os.path.join(raw_dir, 'device.csv')
        email_path = os.path.join(raw_dir, 'email.csv')
        http_path = os.path.join(raw_dir, 'http.csv')

        if os.path.exists(device_path):
            print(f"Loading {device_path} (device timeline)...", flush=True)
            _dev_cols = {'id', 'date', 'user', 'pc', 'activity'}
            try:
                device_df = pd.read_csv(
                    device_path,
                    usecols=lambda c: c in _dev_cols,
                    low_memory=False,
                    engine='c',
                )
            except (ValueError, TypeError, OSError):
                device_df = pd.read_csv(device_path, low_memory=False, engine='c')
            print(f"  device rows: {len(device_df):,}", flush=True)

        if os.path.exists(email_path):
            print(f"Loading {email_path} (email timeline; often the slowest file)...", flush=True)
            _email_cols = {'id', 'date', 'user', 'pc', 'to', 'size', 'attachments'}
            try:
                email_df = pd.read_csv(
                    email_path,
                    usecols=lambda c: c in _email_cols,
                    low_memory=False,
                    engine='c',
                )
            except (ValueError, TypeError, OSError):
                email_df = pd.read_csv(email_path, low_memory=False, engine='c')
            print(f"  email rows: {len(email_df):,}", flush=True)

        if os.path.exists(http_path):
            print(f"Loading {http_path} (http timeline)...", flush=True)
            _http_cols = {'id', 'date', 'user', 'pc', 'url'}
            try:
                http_df = pd.read_csv(
                    http_path,
                    usecols=lambda c: c in _http_cols,
                    low_memory=False,
                    engine='c',
                )
            except (ValueError, TypeError, OSError):
                http_df = pd.read_csv(http_path, low_memory=False, engine='c')
            print(f"  http rows: {len(http_df):,}", flush=True)

        for df_raw, name in [(device_df, 'device'), (email_df, 'email'), (http_df, 'http')]:
            if not df_raw.empty and 'date' in df_raw.columns:
                df_raw['date'] = pd.to_datetime(df_raw['date'], errors='coerce')
            else:
                if df_raw.empty:
                    print(f"Warning: {name}.csv not found or empty in {raw_dir}")
                else:
                    print(f"Warning: {name}.csv missing 'date' column; skipping this source in timeline.")

    ldap_proc = None
    if args.no_ldap:
        print("Skipping LDAP (--no-ldap).", flush=True)
    else:
        print(f"Loading LDAP data from {args.ldap_dir}...", flush=True)
        ldap_proc = LDAPProcessor(args.ldap_dir)
        print("  LDAP ready.", flush=True)

    # Group by (user, date)
    sus_df['file_date'] = pd.to_datetime(sus_df['file_date'], errors='coerce')
    sus_df['date_only'] = sus_df['file_date'].dt.date
    gb = sus_df.groupby(['file_user', 'date_only'], sort=False)
    n_groups_total = gb.ngroups
    if args.max_users:
        n_groups = min(args.max_users, n_groups_total)
        groups_iter = islice(gb, args.max_users)
    else:
        n_groups = n_groups_total
        groups_iter = gb

    need_users = set(sus_df['file_user'].dropna().unique())

    # Pre-index logs and raw sources: per-window full-table scans were ~O(n_windows * n_rows).
    logs_df['date_only'] = logs_df['file_date'].dt.date
    t_idx0 = time.perf_counter()
    print(
        f"Indexing integrated logs for {len(need_users):,} users ({len(logs_df):,} rows)...",
        flush=True,
    )
    logs_by_user, logs_by_ud = _index_logs_by_user_and_day(
        logs_df, 'file_user', need_users
    )
    print(f"  log index done in {time.perf_counter() - t_idx0:.1f}s", flush=True)

    device_by_ud = {}
    email_by_ud = {}
    http_by_ud = {}
    for df_raw, name, out_dict in [
        (device_df, 'device', device_by_ud),
        (email_df, 'email', email_by_ud),
        (http_df, 'http', http_by_ud),
    ]:
        if df_raw.empty or 'user' not in df_raw.columns or 'date' not in df_raw.columns:
            continue
        df_raw['date_only'] = df_raw['date'].dt.date
        t0_src = time.perf_counter()
        print(f"Indexing {name}.csv for timeline lookups...", flush=True)
        _, by_ud = _index_logs_by_user_and_day(df_raw, 'user', need_users)
        out_dict.update(by_ud)
        print(f"  {name} index done in {time.perf_counter() - t0_src:.1f}s", flush=True)

    print(f"Building narratives for {n_groups} user-date windows...", flush=True)
    if args.progress_every:
        print(
            f"  Progress: every {args.progress_every} windows (use --progress-every 0 to silence)",
            flush=True,
        )
    out_lines = []
    t0 = time.perf_counter()
    for idx, ((user_id, date_only), group) in enumerate(groups_iter, start=1):
        window_date = datetime.combine(date_only, datetime.min.time())
        window_logs = logs_by_ud.get((user_id, date_only), pd.DataFrame())
        full_logs_30d = logs_by_user.get(user_id, pd.DataFrame())

        if not device_df.empty:
            device_logs_day = device_by_ud.get((user_id, date_only), pd.DataFrame())
        else:
            device_logs_day = None

        if not email_df.empty:
            email_logs_day = email_by_ud.get((user_id, date_only), pd.DataFrame())
        else:
            email_logs_day = None

        if not http_df.empty:
            http_logs_day = http_by_ud.get((user_id, date_only), pd.DataFrame())
        else:
            http_logs_day = None

        narrative = build_narrative_for_window(
            user_id=user_id,
            window_date=window_date,
            window_logs=window_logs,
            ldap_proc=ldap_proc,
            full_logs_30d=full_logs_30d,
            device_logs_day=device_logs_day,
            email_logs_day=email_logs_day,
            http_logs_day=http_logs_day,
            use_ldap=not args.no_ldap,
            use_history=not args.no_history,
            use_cross_source=not args.no_cross_source,
        )
        out_lines.append(narrative)
        out_lines.append("\n" + "=" * 80 + "\n")

        if args.progress_every and idx % args.progress_every == 0:
            elapsed = time.perf_counter() - t0
            pct = 100.0 * idx / n_groups if n_groups else 100.0
            rate = idx / elapsed if elapsed > 0 else 0.0
            eta = (n_groups - idx) / rate if rate > 0 else 0.0
            print(
                f"  ... {idx}/{n_groups} ({pct:.1f}%) | {elapsed:.1f}s elapsed | "
                f"~{rate:.2f} windows/s | ETA ~{eta:.0f}s | last: {user_id} @ {date_only}",
                flush=True,
            )

    print(f"Built {n_groups} narratives in {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"Writing narratives to {args.output}...", flush=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("\n".join(out_lines))
    print("Done.", flush=True)


if __name__ == '__main__':
    main()

