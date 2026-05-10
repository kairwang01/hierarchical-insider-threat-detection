#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Log Integration Script
Matches logon events from logon.csv with file operations from file.csv
by user_id and timestamp. Uses unified cleaning first (see data_cleaning.py).

Matching rule (same as before): for each file event, use the latest logon at or
before the file time that falls within time_window_hours — implemented with
pd.merge_asof (backward) per user for speed and stable results.
"""

import pandas as pd
import argparse
import os

from data_cleaning import clean_dataframe


def integrate_logs(logon_file, file_file, output_file=None, time_window_hours=24):
    """
    Integrate logs: Match records from logon.csv and file.csv by user_id and timestamp

    Parameters:
        logon_file: Path to logon.csv file
        file_file: Path to file.csv file
        output_file: Output file path (optional)
        time_window_hours: Time window (hours) for matching logons and file operations
    """
    print(f"Reading {logon_file}...")
    logon_df = pd.read_csv(logon_file, low_memory=False, engine='c')
    print(f"Reading {file_file}...")
    file_df = pd.read_csv(file_file, low_memory=False, engine='c')

    print("Cleaning logon data (parse time, drop invalid/missing, dedup, sort)...")
    logon_df = clean_dataframe(logon_df, date_col='date', user_col='user')
    logon_df = logon_df[logon_df['activity'] == 'Logon'].copy()

    print("Cleaning file data (parse time, drop invalid/missing, dedup, sort)...")
    file_df = clean_dataframe(file_df, date_col='date', user_col='user')

    print(f"Number of logon records: {len(logon_df)}")
    print(f"Number of file operation records: {len(file_df)}")

    # Rename so merge_asof does not clobber columns; align on temporary user key
    f = file_df.rename(columns={
        'id': 'file_id',
        'date': 'file_date',
        'user': 'user',
        'pc': 'file_pc',
        'filename': 'file_filename',
        'content': 'file_content',
    })
    l = logon_df.rename(columns={
        'id': 'logon_id',
        'date': 'logon_date',
        'user': 'user',
        'pc': 'logon_pc',
        'activity': 'logon_activity',
    })
    l = l.rename(columns={'datetime': 'logon_datetime'})
    f = f.rename(columns={'datetime': 'file_datetime'})

    # merge_asof requires the left/right *on* columns to be globally non-decreasing.
    # Sort by time first, then user (stable), not user-first — otherwise timestamps
    # drop backward when switching users and pandas raises "left keys must be sorted".
    f = f.sort_values(['file_datetime', 'user'], kind='mergesort')
    l = l.sort_values(['logon_datetime', 'user'], kind='mergesort')

    print("Matching records (merge_asof per user)...")
    merged = pd.merge_asof(
        f,
        l,
        left_on='file_datetime',
        right_on='logon_datetime',
        by='user',
        direction='backward',
    )

    td = (merged['file_datetime'] - merged['logon_datetime']).dt.total_seconds() / 3600.0
    merged['_time_diff_hours'] = td
    valid = merged['logon_datetime'].notna() & (merged['_time_diff_hours'] <= time_window_hours)
    result_df = merged.loc[valid].copy()
    result_df['time_diff_hours'] = result_df['_time_diff_hours']
    result_df['same_pc'] = result_df['file_pc'] == result_df['logon_pc']
    result_df = result_df.rename(columns={'user': 'file_user'})
    drop_cols = ['file_datetime', 'logon_datetime', '_time_diff_hours']
    result_df = result_df.drop(columns=[c for c in drop_cols if c in result_df.columns])

    print(f"\nMatching completed!")
    print(f"Number of successfully matched records: {len(result_df)}")
    if len(file_df) > 0:
        print(f"Match rate: {len(result_df)/len(file_df)*100:.2f}%")

    if len(result_df) > 0:
        print(f"\nStatistics:")
        print(f"  Average time difference: {result_df['time_diff_hours'].mean():.2f} hours")
        print(f"  Median time difference: {result_df['time_diff_hours'].median():.2f} hours")
        print(f"  Operations on same PC: {result_df['same_pc'].sum()} ({result_df['same_pc'].sum()/len(result_df)*100:.2f}%)")

    if output_file:
        print(f"\nSaving results to {output_file}...")
        result_df.to_csv(output_file, index=False)
        print("Save completed!")
    else:
        default_output = '../integrated_logs.csv'
        print(f"\nSaving results to {default_output}...")
        result_df.to_csv(default_output, index=False)
        print("Save completed!")
        return default_output

    return output_file


def main():
    parser = argparse.ArgumentParser(description='Integrate logs: Match records from logon.csv and file.csv')
    parser.add_argument('--logon', '-l', type=str, default='../r4.2/logon.csv',
                        help='Path to logon.csv file (default: ../r4.2/logon.csv)')
    parser.add_argument('--file', '-f', type=str, default='../r4.2/file.csv',
                        help='Path to file.csv file (default: ../r4.2/file.csv)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path (default: integrated_logs.csv)')
    parser.add_argument('--time-window', '-t', type=float, default=24.0,
                        help='Time window (hours) for matching logons and file operations (default: 24.0)')

    args = parser.parse_args()

    if not os.path.exists(args.logon):
        print(f"Error: File not found {args.logon}")
        return

    if not os.path.exists(args.file):
        print(f"Error: File not found {args.file}")
        return

    integrate_logs(args.logon, args.file, args.output, args.time_window)


if __name__ == '__main__':
    main()
