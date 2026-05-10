#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Engineering Script
Extract features for insider threat detection
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from data_cleaning import clean_integrated_logs_df
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from ldap_helper import LDAPProcessor


def extract_temporal_features(df):
    """Extract temporal features"""
    print("Extracting temporal features...")
    
    df = df.copy()
    df['file_date'] = pd.to_datetime(df['file_date'])
    
    # Time features
    df['hour'] = df['file_date'].dt.hour
    df['day_of_week'] = df['file_date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_month'] = df['file_date'].dt.day
    df['month'] = df['file_date'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_after_hours'] = ((df['hour'] < 8) | (df['hour'] > 18)).astype(int)
    
    return df


def extract_user_behavior_features(df):
    """Extract user behavior features"""
    print("Extracting user behavior features...")
    
    # User-level statistics
    user_stats = df.groupby('file_user').agg({
        'file_id': 'count',
        'file_pc': 'nunique',
        'file_filename': 'nunique',
        'time_diff_hours': ['mean', 'std', 'min', 'max'],
        'same_pc': 'mean'
    }).reset_index()
    
    # Flatten column names
    user_stats.columns = ['file_user', 'user_total_ops', 'user_unique_pcs', 
                          'user_unique_files', 'user_avg_time_diff', 
                          'user_std_time_diff', 'user_min_time_diff', 
                          'user_max_time_diff', 'user_same_pc_rate']
    
    # Merge back
    df = df.merge(user_stats, on='file_user', how='left')
    
    # File extension features
    df['file_ext'] = df['file_filename'].str.extract(r'\.(\w+)$')[0]
    df['is_sensitive_ext'] = df['file_ext'].isin(['doc', 'pdf', 'xls', 'ppt']).astype(int)
    
    return df


def extract_session_features(df):
    """Extract session features"""
    print("Extracting session features...")
    
    # Session features (based on time_diff_hours)
    df['is_new_session'] = (df['time_diff_hours'] > 4).astype(int)
    df['session_duration'] = df['time_diff_hours']
    
    return df


def extract_ldap_features(df, ldap_processor):
    """Extract LDAP-based features"""
    print("Extracting LDAP features...")
    
    if ldap_processor is None:
        return df
    
    unique_users = df['file_user'].unique()
    rows = []
    for user_id in unique_users:
        info = ldap_processor.get_user_info(user_id)
        is_term, _ = ldap_processor.is_terminated(user_id)
        if info:
            rows.append({
                'file_user': user_id,
                'user_role': info['role'],
                'user_department': info['department'],
                'user_business_unit': info['business_unit'],
                'is_terminated': bool(is_term),
            })
        else:
            rows.append({
                'file_user': user_id,
                'user_role': None,
                'user_department': None,
                'user_business_unit': None,
                'is_terminated': bool(is_term),
            })
    user_info_df = pd.DataFrame(rows)
    df = df.merge(user_info_df, on='file_user', how='left')
    df['is_terminated'] = df['is_terminated'].fillna(False).astype(int)
    
    return df


def extract_aggregated_features(df):
    """Extract aggregated features over time windows"""
    print("Extracting aggregated features...")
    
    df = df.copy()
    df['file_date'] = pd.to_datetime(df['file_date'])
    df = df.sort_values('file_date')
    
    # Daily activity per user
    df['date'] = df['file_date'].dt.date
    daily_user_activity = (
        df.groupby(['file_user', 'date'], sort=False).size().reset_index(name='daily_ops')
    )
    daily_user_activity = daily_user_activity.sort_values(['file_user', 'date'])
    g = daily_user_activity.groupby('file_user', sort=False)['daily_ops']
    daily_user_activity['rolling_mean_7d'] = g.transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    daily_user_activity['rolling_std_7d'] = g.transform(
        lambda x: x.rolling(window=7, min_periods=1).std()
    )

    daily_stats_df = daily_user_activity.copy()
    daily_stats_df['date'] = pd.to_datetime(daily_stats_df['date'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.merge(
        daily_stats_df[
            ['file_user', 'date', 'daily_ops', 'rolling_mean_7d', 'rolling_std_7d']
        ],
        on=['file_user', 'date'],
        how='left',
    )
    df['ops_deviation'] = (df['daily_ops'] - df['rolling_mean_7d']) / (df['rolling_std_7d'] + 1e-6)
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Feature engineering for insider threat detection')
    parser.add_argument('--input', '-i', type=str, default='../integrated_logs_labeled.csv',
                       help='Input CSV file')
    parser.add_argument('--output', '-o', type=str, default='../features.csv',
                       help='Output CSV file with features')
    parser.add_argument('--ldap-dir', type=str, default='../r4.2/LDAP',
                       help='LDAP directory')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample size (for testing)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input, low_memory=False, engine='c')

    # --- Unified cleaning first (then extract features) ---
    print("Cleaning data (standardize dates, drop invalid/missing, dedup, sort)...")
    df = clean_integrated_logs_df(df, date_col='file_date', user_col='file_user',
                                   logon_date_col='logon_date' if 'logon_date' in df.columns else None)

    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42)
        print(f"  Sampled {len(df)} records for testing")
    
    print(f"  Loaded {len(df):,} records")
    
    # Load LDAP processor
    ldap_processor = None
    try:
        ldap_processor = LDAPProcessor(args.ldap_dir)
        print("LDAP processor loaded")
    except Exception as e:
        print(f"Warning: Could not load LDAP processor: {e}")
    
    # Extract features
    print("\n=== Feature Extraction ===")
    df = extract_temporal_features(df)
    df = extract_user_behavior_features(df)
    df = extract_session_features(df)
    
    if ldap_processor:
        df = extract_ldap_features(df, ldap_processor)
    
    # Extract aggregated features (optional, can be slow)
    # df = extract_aggregated_features(df)
    
    # Select feature columns
    feature_cols = [
        # Temporal
        'hour', 'day_of_week', 'day_of_month', 'month', 
        'is_weekend', 'is_after_hours',
        # User behavior
        'user_total_ops', 'user_unique_pcs', 'user_unique_files',
        'user_avg_time_diff', 'user_std_time_diff', 
        'user_min_time_diff', 'user_max_time_diff', 'user_same_pc_rate',
        # Session
        'is_new_session', 'session_duration', 'time_diff_hours', 'same_pc',
        # File
        'is_sensitive_ext',
        # LDAP (if available)
        'is_terminated'
    ]
    
    # Add label if available
    if 'is_malicious' in df.columns:
        feature_cols.append('is_malicious')
        feature_cols.append('malicious_scenario')
    
    # Select available columns
    available_cols = [col for col in feature_cols if col in df.columns]
    output_df = df[available_cols + ['file_user', 'file_date']].copy()
    
    # Save
    print(f"\nSaving features to {args.output}...")
    output_df.to_csv(args.output, index=False)
    print(f"  Saved {len(output_df):,} records with {len(available_cols)} features")
    
    # Print feature summary
    print("\n=== Feature Summary ===")
    print(f"Total features: {len(available_cols)}")
    print("Features:")
    for col in available_cols:
        print(f"  - {col}")
    
    if 'is_malicious' in output_df.columns:
        print(f"\nLabel distribution:")
        print(output_df['is_malicious'].value_counts())
        print(f"Malicious rate: {output_df['is_malicious'].mean()*100:.2f}%")


if __name__ == '__main__':
    main()
