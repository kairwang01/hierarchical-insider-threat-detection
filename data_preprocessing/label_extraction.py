#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Label Extraction Script
Marks user sequences as malicious based on ground-truth data from answers folder
"""

import numpy as np
import pandas as pd
import os
import argparse
import sys
sys.path.insert(0, os.path.dirname(__file__))
from data_cleaning import (
    parse_timestamp_safe,
    parse_timestamp_strict,
    clean_dataframe,
    normalize_user_id,
    _normalize_user_series,
)


def parse_timestamp(date_str):
    """Parse timestamp string to datetime (strict, for insiders). Raises on failure."""
    return parse_timestamp_strict(date_str)


def load_insiders(insiders_file='answers/insiders.csv', dataset_filter=None):
    """
    Load insiders.csv and create a mapping of malicious users with time ranges
    
    Parameters:
        insiders_file: Path to insiders.csv
        dataset_filter: Filter by dataset (e.g., '4.2'), or None for all
    
    Returns:
        dict: {user_id: [(start_time, end_time, scenario, dataset), ...]}
    """
    print(f"Loading insiders from {insiders_file}...")
    
    if not os.path.exists(insiders_file):
        raise FileNotFoundError(f"Insiders file not found: {insiders_file}")
    
    df = pd.read_csv(insiders_file)
    
    # Filter by dataset if specified
    if dataset_filter:
        # `insiders.csv` 的 dataset 列可能被 pandas 读成 float（例如 4.2），
        # 但 dataset_filter 通常是字符串（例如 "4.2"），直接比较会得到 0 记录。
        df = df.copy()
        df['dataset'] = df['dataset'].astype(str).str.strip()
        ds = str(dataset_filter).strip()
        df = df[df['dataset'] == ds].copy()
        print(f"Filtered to dataset {ds}: {len(df)} records")
    
    malicious_users = {}
    
    for _, row in df.iterrows():
        user_id = normalize_user_id(row['user'])
        if user_id is None:
            continue
        start_str = row['start']
        end_str = row['end']
        scenario = row['scenario']
        dataset = row['dataset']

        try:
            start_time = parse_timestamp(start_str)
            end_time = parse_timestamp(end_str)
        except ValueError as e:
            print(f"Warning: Could not parse dates for user {user_id}: {e}")
            continue

        if user_id not in malicious_users:
            malicious_users[user_id] = []

        malicious_users[user_id].append({
            'start': start_time,
            'end': end_time,
            'scenario': scenario,
            'dataset': str(dataset).strip() if dataset is not None else dataset
        })
    
    print(f"Loaded {len(malicious_users)} malicious users")
    return malicious_users


def is_malicious_record(user_id, record_time, malicious_users):
    """
    Check if a record is malicious based on user ID and timestamp
    
    Parameters:
        user_id: User ID to check
        record_time: Timestamp of the record (datetime object)
        malicious_users: Dictionary from load_insiders()
    
    Returns:
        tuple: (is_malicious: bool, scenario: int or None, dataset: str or None)
    """
    if user_id not in malicious_users:
        return False, None, None
    
    # Check if record time falls within any malicious period for this user
    for period in malicious_users[user_id]:
        if period['start'] <= record_time <= period['end']:
            return True, period['scenario'], period['dataset']
    
    return False, None, None


def is_malicious_sequence(user_id, window_start, window_end, malicious_users):
    """
    Check if a (user, time window) overlaps any malicious period.
    Used for sequence-level labeling: 若行为序列的时间段内包含恶意记录则标 1.

    Parameters
    ----------
    user_id : str
    window_start, window_end : datetime
        Window [window_start, window_end] (e.g. one day 00:00 to 23:59:59).
    malicious_users : dict
        From load_insiders().

    Returns
    -------
    (bool, scenario or None)
    """
    if user_id not in malicious_users:
        return False, None
    for period in malicious_users[user_id]:
        if window_start <= period['end'] and window_end >= period['start']:
            return True, period['scenario']
    return False, None


def label_sequence_table(seq_df, insiders_path, dataset_filter='4.2',
                         user_col='file_user', window_date_col='file_date'):
    """
    序列级打标：若某条序列（用户+时间窗口）的时间段与 answers 中该用户的恶意时间段有重叠，则标 1。

    Parameters
    ----------
    seq_df : pandas.DataFrame
        序列表，必须含 user_col 与 window_date_col（每行一个用户一个窗口）。
    insiders_path : str
        answers/insiders.csv 路径。
    dataset_filter : str
        如 '4.2'。
    user_col : str
        用户名列，默认 'file_user'。
    window_date_col : str
        窗口日期列（日期或 datetime），默认 'file_date'。

    Returns
    -------
    pandas.DataFrame
        带 is_malicious, malicious_scenario 的副本。
    """
    seq_df = seq_df.copy()
    if user_col not in seq_df.columns or window_date_col not in seq_df.columns:
        raise ValueError(f"Sequence table must have columns '{user_col}' and '{window_date_col}'.")
    seq_df[user_col] = _normalize_user_series(seq_df[user_col])
    bad_u = seq_df[user_col].isna()
    if bad_u.any():
        print(f"  Dropping {bad_u.sum()} sequence rows with invalid {user_col}")
        seq_df = seq_df[~bad_u].copy()

    malicious_users = load_insiders(insiders_path, dataset_filter)
    seq_df['is_malicious'] = False
    seq_df['malicious_scenario'] = None
    if not malicious_users:
        return seq_df

    wd = pd.to_datetime(seq_df[window_date_col], errors='coerce')
    invalid_w = wd.isna()
    if invalid_w.any():
        print(f"  Dropping {invalid_w.sum()} sequence rows with invalid {window_date_col}")
        seq_df = seq_df.loc[~invalid_w].copy()
        wd = pd.to_datetime(seq_df[window_date_col], errors='coerce')

    window_start = wd.dt.normalize()
    window_end = window_start + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    is_mal = np.zeros(len(seq_df), dtype=bool)
    scen = np.empty(len(seq_df), dtype=object)
    scen[:] = None
    users = seq_df[user_col].to_numpy()
    ws = window_start.to_numpy()
    we = window_end.to_numpy()
    pos = np.arange(len(seq_df))

    for user_id, periods in malicious_users.items():
        um = users == user_id
        if not um.any():
            continue
        idx_local = pos[um]
        ws_u = ws[um]
        we_u = we[um]
        already = is_mal[um].copy()
        for period in periods:
            ps = np.datetime64(pd.Timestamp(period['start']))
            pe = np.datetime64(pd.Timestamp(period['end']))
            overlap = (ws_u <= pe) & (we_u >= ps) & ~already
            if np.any(overlap):
                hit = idx_local[overlap]
                is_mal[hit] = True
                scen[hit] = period['scenario']
                already |= overlap

    seq_df['is_malicious'] = is_mal
    seq_df['malicious_scenario'] = scen
    return seq_df


def _label_arrays_for_records(df, user_col, datetime_col, malicious_users):
    """
    Vectorized record-level labels: first matching insider period wins per row.
    Returns (is_malicious bool array, scenario array, dataset array).
    """
    n = len(df)
    is_mal = np.zeros(n, dtype=bool)
    scen = np.empty(n, dtype=object)
    dst = np.empty(n, dtype=object)
    scen[:] = None
    dst[:] = None

    users = df[user_col].to_numpy()
    times = pd.to_datetime(df[datetime_col]).to_numpy()

    for user_id, periods in malicious_users.items():
        um = users == user_id
        if not um.any():
            continue
        pos = np.flatnonzero(um)
        ti = times[pos]
        already = is_mal[pos].copy()
        for p in periods:
            ps = np.datetime64(pd.Timestamp(p['start']))
            pe = np.datetime64(pd.Timestamp(p['end']))
            in_p = (ti >= ps) & (ti <= pe) & ~already
            if np.any(in_p):
                hit = pos[in_p]
                is_mal[hit] = True
                scen[hit] = p['scenario']
                dst[hit] = p['dataset']
                already |= in_p

    return is_mal, scen, dst


def label_data_file(data_file, malicious_users, output_file=None, 
                    user_col=None, date_col=None):
    """
    Label a data file (logon.csv, file.csv, etc.) with malicious labels
    
    Parameters:
        data_file: Path to data CSV file
        malicious_users: Dictionary from load_insiders()
        output_file: Output file path (optional)
        user_col: Name of user column (auto-detect if None)
        date_col: Name of date column (auto-detect if None)
    """
    print(f"\nLabeling {data_file}...")
    
    if not os.path.exists(data_file):
        print(f"Error: File not found {data_file}")
        return None
    
    # Read data file
    df = pd.read_csv(data_file)
    print(f"  Loaded {len(df)} records")
    print(f"  Available columns: {list(df.columns)}")
    
    # Auto-detect column names if not provided
    if user_col is None:
        # Try common user column names (prioritize file_user for integrated_logs.csv)
        for col in ['file_user', 'logon_user', 'user']:
            if col in df.columns:
                user_col = col
                break
        if user_col is None:
            raise ValueError("Could not find user column. Available columns: " + str(list(df.columns)))
    
    if date_col is None:
        # If we found file_user, prioritize file_date; if logon_user, prioritize logon_date
        if user_col == 'file_user':
            date_col = 'file_date' if 'file_date' in df.columns else None
        elif user_col == 'logon_user':
            date_col = 'logon_date' if 'logon_date' in df.columns else None
        
        # If still not found, try common date column names
        if date_col is None:
            for col in ['file_date', 'logon_date', 'date']:
                if col in df.columns:
                    date_col = col
                    break
        
        if date_col is None:
            raise ValueError("Could not find date column. Available columns: " + str(list(df.columns)))
    
    # Verify columns exist
    if user_col not in df.columns:
        print(f"  Available columns: {list(df.columns)}")
        raise ValueError(f"User column '{user_col}' not found in DataFrame")
    if date_col not in df.columns:
        print(f"  Available columns: {list(df.columns)}")
        raise ValueError(f"Date column '{date_col}' not found in DataFrame")
    
    print(f"  Using columns: user='{user_col}', date='{date_col}'")

    # --- Unified cleaning first (then label) ---
    print("  Cleaning data (parse time, drop invalid/missing, dedup, sort)...")
    df = clean_dataframe(df, date_col=date_col, user_col=user_col)

    print("  Labeling records (vectorized)...")
    is_mal, scen, dst = _label_arrays_for_records(df, user_col, 'datetime', malicious_users)
    df['is_malicious'] = is_mal
    df['malicious_scenario'] = scen
    df['malicious_dataset'] = dst
    malicious_count = int(is_mal.sum())
    print(f"  Found {malicious_count} malicious records ({malicious_count/len(df)*100:.2f}%)")
    
    # Remove datetime column before saving (keep original date column)
    df_output = df.drop('datetime', axis=1)
    
    # Save labeled data
    if output_file:
        print(f"  Saving to {output_file}...")
        df_output.to_csv(output_file, index=False)
        print("  Save completed!")
    else:
        # Generate default output filename
        base_name = os.path.splitext(os.path.basename(data_file))[0]
        dir_name = os.path.dirname(data_file)
        if dir_name:
            default_output = os.path.join(dir_name, f"{base_name}_labeled.csv")
        else:
            default_output = f"{base_name}_labeled.csv"
        print(f"  Saving to {default_output}...")
        df_output.to_csv(default_output, index=False)
        print("  Save completed!")
        return default_output
    
    return output_file


def create_user_sequence_labels(malicious_users, output_file='user_sequence_labels.csv'):
    """
    Create a summary file of all malicious user sequences
    
    Parameters:
        malicious_users: Dictionary from load_insiders()
        output_file: Output file path
    """
    print("\nCreating user sequence labels summary...")
    
    records = []
    for user_id, periods in malicious_users.items():
        for period in periods:
            records.append({
                'user_id': user_id,
                'start': period['start'].strftime('%Y-%m-%d %H:%M:%S'),
                'end': period['end'].strftime('%Y-%m-%d %H:%M:%S'),
                'scenario': period['scenario'],
                'dataset': period['dataset']
            })
    
    df = pd.DataFrame(records)
    df = df.sort_values(['user_id', 'start'])
    
    print(f"  Total malicious sequences: {len(df)}")
    print(f"  Unique malicious users: {df['user_id'].nunique()}")
    
    print(f"  Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    print("  Save completed!")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Extract labels for malicious user sequences from ground-truth data'
    )
    parser.add_argument('--insiders', '-i', type=str, default='../answers/insiders.csv',
                       help='Path to insiders.csv file (default: ../answers/insiders.csv)')
    parser.add_argument('--dataset', '-d', type=str, default=None,
                       help='Filter by dataset (e.g., "4.2"), or None for all')
    parser.add_argument('--logon', '-l', type=str, default=None,
                       help='Path to logon.csv file to label')
    parser.add_argument('--file', '-f', type=str, default=None,
                       help='Path to file.csv file to label')
    parser.add_argument('--device', type=str, default=None,
                       help='Path to device.csv file to label')
    parser.add_argument('--http', type=str, default=None,
                       help='Path to http.csv file to label')
    parser.add_argument('--email', type=str, default=None,
                       help='Path to email.csv file to label')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='Output directory for labeled files (default: same as input)')
    parser.add_argument('--summary', '-s', action='store_true',
                       help='Generate summary of malicious user sequences')
    parser.add_argument('--sequences', type=str, default=None,
                       help='Path to sequence CSV (user+window per row) for sequence-level labeling')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for labeled sequence CSV (use with --sequences)')
    parser.add_argument('--user-col', type=str, default='file_user',
                       help='User column name in sequence table (default: file_user)')
    parser.add_argument('--date-col', type=str, default='file_date',
                       help='Window date column in sequence table (default: file_date)')
    
    args = parser.parse_args()
    
    # 序列级打标：对“用户+时间窗口”序列表打标（若窗口与恶意时间段重叠则标 1）
    if args.sequences:
        if not os.path.exists(args.sequences):
            print(f"Error: File not found {args.sequences}")
            return
        out_path = args.output or args.sequences.replace('.csv', '_labeled.csv')
        print(f"Loading sequences from {args.sequences}...")
        seq_df = pd.read_csv(args.sequences)
        print(f"  Loaded {len(seq_df):,} sequences")
        print("Applying sequence-level labeling (window overlap with insiders)...")
        seq_df = label_sequence_table(seq_df, args.insiders, args.dataset or '4.2',
                                      user_col=args.user_col, window_date_col=args.date_col)
        seq_df.to_csv(out_path, index=False)
        print(f"Saved {out_path} (malicious: {seq_df['is_malicious'].sum():,})")
        return
    
    # Load malicious users for record-level labeling
    try:
        malicious_users = load_insiders(args.insiders, args.dataset)
    except Exception as e:
        print(f"Error loading insiders: {e}")
        return
    
    if not malicious_users:
        print("No malicious users found!")
        return
    
    # Generate summary if requested
    if args.summary:
        create_user_sequence_labels(malicious_users)
    
    # Label data files
    files_to_label = []
    if args.logon:
        files_to_label.append(('logon', args.logon, None, None))  # Auto-detect columns
    if args.file:
        files_to_label.append(('file', args.file, None, None))  # Auto-detect columns
    if args.device:
        files_to_label.append(('device', args.device, None, None))  # Auto-detect columns
    if args.http:
        files_to_label.append(('http', args.http, None, None))  # Auto-detect columns
    if args.email:
        files_to_label.append(('email', args.email, None, None))  # Auto-detect columns
    
    # If no files specified, use defaults for dataset 4.2
    if not files_to_label and args.dataset == '4.2':
        default_dir = '../r4.2'
        if os.path.exists(default_dir):
            if os.path.exists(os.path.join(default_dir, 'logon.csv')):
                files_to_label.append(('logon', os.path.join(default_dir, 'logon.csv'), None, None))
            if os.path.exists(os.path.join(default_dir, 'file.csv')):
                files_to_label.append(('file', os.path.join(default_dir, 'file.csv'), None, None))
            if os.path.exists(os.path.join(default_dir, 'device.csv')):
                files_to_label.append(('device', os.path.join(default_dir, 'device.csv'), None, None))
            if os.path.exists(os.path.join(default_dir, 'http.csv')):
                files_to_label.append(('http', os.path.join(default_dir, 'http.csv'), None, None))
            if os.path.exists(os.path.join(default_dir, 'email.csv')):
                files_to_label.append(('email', os.path.join(default_dir, 'email.csv'), None, None))
    
    if not files_to_label:
        print("\nNo files to label. Use --logon, --file, etc. to specify files.")
        print("Or use --dataset 4.2 to automatically label files in r4.2/ directory.")
        return
    
    for file_type, file_path, user_col, date_col in files_to_label:
        if args.output_dir:
            base_name = os.path.basename(file_path)
            output_file = os.path.join(args.output_dir, f"{os.path.splitext(base_name)[0]}_labeled.csv")
        else:
            output_file = None
        
        try:
            # Pass column names (None means auto-detect)
            label_data_file(file_path, malicious_users, output_file, user_col, date_col)
        except Exception as e:
            print(f"Error labeling {file_path}: {e}")


if __name__ == '__main__':
    main()
