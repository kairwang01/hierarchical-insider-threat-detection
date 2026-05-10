#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis (EDA) Script
Data cleaning and preliminary exploration for insider threat detection
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import argparse
from collections import Counter


def load_integrated_logs(file_path='../integrated_logs_labeled.csv'):
    """Load integrated logs with labels"""
    print(f"Loading integrated logs from {file_path}...")
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Trying integrated_logs.csv...")
        file_path = 'integrated_logs.csv'
        if not os.path.exists(file_path):
            raise FileNotFoundError("No integrated logs file found")
    
    df = pd.read_csv(file_path)
    print(f"  Loaded {len(df):,} records")
    return df


def load_ldap_summary(file_path='../ldap_user_summary.csv'):
    """Load LDAP user summary"""
    print(f"Loading LDAP summary from {file_path}...")
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Skipping LDAP data...")
        return None
    
    df = pd.read_csv(file_path)
    print(f"  Loaded {len(df):,} users")
    return df


def clean_data(df):
    """Clean and prepare data for analysis"""
    print("\n=== Data Cleaning ===")
    original_len = len(df)
    
    # Convert date columns to datetime
    print("Converting date columns to datetime...")
    date_cols = ['file_date', 'logon_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Remove records with invalid dates
    if 'file_date' in df.columns:
        invalid_dates = df['file_date'].isna().sum()
        if invalid_dates > 0:
            print(f"  Removing {invalid_dates} records with invalid file_date")
            df = df[df['file_date'].notna()].copy()
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"  Found {duplicates} duplicate records")
        df = df.drop_duplicates().copy()
    
    # Remove records with missing critical fields
    critical_fields = ['file_user', 'file_date']
    for field in critical_fields:
        if field in df.columns:
            missing = df[field].isna().sum()
            if missing > 0:
                print(f"  Removing {missing} records with missing {field}")
                df = df[df[field].notna()].copy()
    
    print(f"  Cleaned data: {original_len:,} -> {len(df):,} records")
    return df


def basic_statistics(df):
    """Generate basic statistics"""
    print("\n=== Basic Statistics ===")
    
    print(f"\nTotal Records: {len(df):,}")
    print(f"Unique Users: {df['file_user'].nunique():,}")
    print(f"Unique PCs: {df['file_pc'].nunique():,}")
    
    if 'is_malicious' in df.columns:
        malicious_count = df['is_malicious'].sum()
        malicious_rate = malicious_count / len(df) * 100
        print(f"\nMalicious Records: {malicious_count:,} ({malicious_rate:.2f}%)")
        print(f"Malicious Users: {df[df['is_malicious']]['file_user'].nunique():,}")
    
    # Time range
    if 'file_date' in df.columns:
        print(f"\nTime Range:")
        print(f"  Start: {df['file_date'].min()}")
        print(f"  End: {df['file_date'].max()}")
        print(f"  Duration: {(df['file_date'].max() - df['file_date'].min()).days} days")
    
    # File operations statistics
    if 'file_filename' in df.columns:
        print(f"\nFile Operations:")
        print(f"  Unique Files: {df['file_filename'].nunique():,}")
        print(f"  Most Common Extensions:")
        extensions = df['file_filename'].str.extract(r'\.(\w+)$')[0].value_counts().head(10)
        for ext, count in extensions.items():
            print(f"    .{ext}: {count:,}")


def analyze_malicious_behavior(df):
    """Analyze malicious behavior patterns"""
    if 'is_malicious' not in df.columns:
        print("\n=== Malicious Behavior Analysis ===")
        print("No malicious labels found. Skipping...")
        return
    
    print("\n=== Malicious Behavior Analysis ===")
    
    malicious = df[df['is_malicious'] == True].copy()
    
    if len(malicious) == 0:
        print("No malicious records found.")
        return
    
    # Scenario distribution
    print("\nScenario Distribution:")
    scenario_counts = malicious['malicious_scenario'].value_counts().sort_index()
    for scenario, count in scenario_counts.items():
        print(f"  Scenario {scenario}: {count:,} records ({count/len(malicious)*100:.1f}%)")
    
    # Top malicious users
    print("\nTop 10 Malicious Users:")
    top_users = malicious['file_user'].value_counts().head(10)
    for user, count in top_users.items():
        print(f"  {user}: {count:,} malicious records")
    
    # Time patterns
    if 'file_date' in malicious.columns:
        malicious['hour'] = malicious['file_date'].dt.hour
        malicious['day_of_week'] = malicious['file_date'].dt.day_name()
        malicious['month'] = malicious['file_date'].dt.month
        
        print("\nTemporal Patterns:")
        print("  Hour Distribution (Top 5):")
        hour_dist = malicious['hour'].value_counts().head(5)
        for hour, count in hour_dist.items():
            print(f"    {hour:02d}:00 - {count:,} records")
        
        print("  Day of Week Distribution:")
        dow_dist = malicious['day_of_week'].value_counts()
        for day, count in dow_dist.items():
            print(f"    {day}: {count:,} records")
    
    # PC usage
    if 'file_pc' in malicious.columns:
        print("\nPC Usage:")
        print(f"  Unique PCs used: {malicious['file_pc'].nunique()}")
        print("  Top 5 PCs:")
        pc_counts = malicious['file_pc'].value_counts().head(5)
        for pc, count in pc_counts.items():
            print(f"    {pc}: {count:,} operations")


def analyze_user_behavior(df):
    """Analyze user behavior patterns"""
    print("\n=== User Behavior Analysis ===")
    
    # User activity statistics
    user_stats = df.groupby('file_user').agg({
        'file_id': 'count',
        'file_pc': 'nunique',
        'file_filename': 'nunique'
    }).rename(columns={
        'file_id': 'total_operations',
        'file_pc': 'unique_pcs',
        'file_filename': 'unique_files'
    })
    
    if 'is_malicious' in df.columns:
        malicious_counts = df[df['is_malicious']].groupby('file_user').size()
        user_stats['malicious_operations'] = malicious_counts
        user_stats['malicious_operations'] = user_stats['malicious_operations'].fillna(0).astype(int)
        user_stats['malicious_rate'] = (user_stats['malicious_operations'] / user_stats['total_operations'] * 100).round(2)
    
    if 'time_diff_hours' in df.columns:
        user_stats['avg_time_diff'] = df.groupby('file_user')['time_diff_hours'].mean().round(2)
        user_stats['same_pc_rate'] = (df.groupby('file_user')['same_pc'].mean() * 100).round(2)
    
    print("\nUser Activity Summary:")
    print(user_stats.describe())
    
    # Most active users
    print("\nTop 10 Most Active Users:")
    top_active = user_stats.nlargest(10, 'total_operations')
    for user, row in top_active.iterrows():
        print(f"  {user}: {row['total_operations']:,} operations, "
              f"{row['unique_pcs']} PCs, {row['unique_files']} files")
    
    return user_stats


def analyze_temporal_patterns(df):
    """Analyze temporal patterns"""
    print("\n=== Temporal Pattern Analysis ===")
    
    if 'file_date' not in df.columns:
        print("No date information available.")
        return
    
    df['hour'] = df['file_date'].dt.hour
    df['day_of_week'] = df['file_date'].dt.day_name()
    df['month'] = df['file_date'].dt.month
    df['date'] = df['file_date'].dt.date
    
    # Daily activity
    daily_activity = df.groupby('date').size()
    print(f"\nDaily Activity:")
    print(f"  Average: {daily_activity.mean():.1f} operations/day")
    print(f"  Max: {daily_activity.max():,} operations/day")
    print(f"  Min: {daily_activity.min():,} operations/day")
    
    # Hourly distribution
    print("\nHourly Distribution (Top 5):")
    hourly_dist = df['hour'].value_counts().head(5)
    for hour, count in hourly_dist.items():
        print(f"  {hour:02d}:00 - {count:,} operations")
    
    # Day of week distribution
    print("\nDay of Week Distribution:")
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_dist = df['day_of_week'].value_counts()
    for day in dow_order:
        if day in dow_dist.index:
            count = dow_dist[day]
            pct = count / len(df) * 100
            print(f"  {day}: {count:,} ({pct:.1f}%)")


def merge_with_ldap(df, ldap_df):
    """Merge data with LDAP information"""
    if ldap_df is None:
        return df
    
    print("\n=== Merging with LDAP Data ===")
    
    # Merge on user_id
    merged = df.merge(
        ldap_df[['user_id', 'role', 'department', 'is_terminated']],
        left_on='file_user',
        right_on='user_id',
        how='left'
    )
    
    print(f"  Merged records: {len(merged):,}")
    print(f"  Records with LDAP info: {merged['role'].notna().sum():,}")
    
    # Analyze by role
    if 'role' in merged.columns:
        print("\nOperations by Role (Top 10):")
        role_counts = merged['role'].value_counts().head(10)
        for role, count in role_counts.items():
            print(f"  {role}: {count:,}")
    
    # Analyze by department
    if 'department' in merged.columns:
        print("\nOperations by Department (Top 10):")
        dept_counts = merged['department'].value_counts().head(10)
        for dept, count in dept_counts.items():
            print(f"  {dept}: {count:,}")
    
    return merged


def generate_summary_report(df, user_stats, output_file='eda_summary.txt'):
    """Generate summary report"""
    print(f"\n=== Generating Summary Report ===")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Exploratory Data Analysis Summary Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total Records: {len(df):,}\n")
        f.write(f"Unique Users: {df['file_user'].nunique():,}\n")
        f.write(f"Unique PCs: {df['file_pc'].nunique():,}\n\n")
        
        if 'is_malicious' in df.columns:
            malicious_count = df['is_malicious'].sum()
            f.write(f"Malicious Records: {malicious_count:,} ({malicious_count/len(df)*100:.2f}%)\n")
            f.write(f"Malicious Users: {df[df['is_malicious']]['file_user'].nunique():,}\n\n")
        
        if 'file_date' in df.columns:
            f.write(f"Time Range: {df['file_date'].min()} to {df['file_date'].max()}\n")
            f.write(f"Duration: {(df['file_date'].max() - df['file_date'].min()).days} days\n\n")
        
        if user_stats is not None:
            f.write("User Statistics:\n")
            f.write(str(user_stats.describe()))
            f.write("\n\n")
    
    print(f"  Report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Exploratory Data Analysis for insider threat detection')
    parser.add_argument('--integrated-logs', type=str, default='../integrated_logs_labeled.csv',
                       help='Path to integrated logs file')
    parser.add_argument('--ldap-summary', type=str, default='../ldap_user_summary.csv',
                       help='Path to LDAP summary file')
    parser.add_argument('--output', '-o', type=str, default='eda_summary.txt',
                       help='Output summary report file')
    parser.add_argument('--save-stats', action='store_true',
                       help='Save user statistics to CSV')
    
    args = parser.parse_args()
    
    # Load data
    df = load_integrated_logs(args.integrated_logs)
    ldap_df = load_ldap_summary(args.ldap_summary)
    
    # Clean data
    df = clean_data(df)
    
    # Basic statistics
    basic_statistics(df)
    
    # Analyze malicious behavior
    analyze_malicious_behavior(df)
    
    # Analyze user behavior
    user_stats = analyze_user_behavior(df)
    
    # Analyze temporal patterns
    analyze_temporal_patterns(df)
    
    # Merge with LDAP
    if ldap_df is not None:
        df = merge_with_ldap(df, ldap_df)
    
    # Generate summary report
    generate_summary_report(df, user_stats, args.output)
    
    # Save user statistics
    if args.save_stats and user_stats is not None:
        user_stats.to_csv('user_behavior_stats.csv')
        print("\nUser statistics saved to user_behavior_stats.csv")
    
    print("\n=== EDA Analysis Complete ===")


if __name__ == '__main__':
    main()
