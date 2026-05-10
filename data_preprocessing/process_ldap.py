#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LDAP Data Processing Script
Processes monthly LDAP CSV files to determine user roles, departments,
and termination status within specific time periods
"""

import pandas as pd
import os
import glob
from datetime import datetime
from collections import defaultdict
import argparse


def parse_month_from_filename(filename):
    """Extract year and month from filename (e.g., '2010-01.csv' -> (2010, 1))"""
    basename = os.path.basename(filename)
    # Remove .csv extension
    date_str = basename.replace('.csv', '')
    try:
        year, month = map(int, date_str.split('-'))
        return year, month
    except ValueError:
        return None, None


def load_ldap_data(ldap_dir):
    """
    Load all LDAP CSV files and organize by month
    
    Returns:
        dict: {year: {month: DataFrame}} - LDAP data organized by year and month
        list: List of (year, month, filename) tuples sorted chronologically
    """
    print(f"Loading LDAP files from {ldap_dir}...")
    
    # Find all CSV files
    pattern = os.path.join(ldap_dir, '*.csv')
    files = glob.glob(pattern)
    
    ldap_data = {}
    file_list = []
    
    for filename in files:
        year, month = parse_month_from_filename(filename)
        if year is None or month is None:
            print(f"Warning: Could not parse date from {filename}, skipping...")
            continue
        
        try:
            df = pd.read_csv(filename)
            if year not in ldap_data:
                ldap_data[year] = {}
            ldap_data[year][month] = df
            file_list.append((year, month, filename))
            print(f"  Loaded {os.path.basename(filename)}: {len(df)} records")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    # Sort file list chronologically
    file_list.sort()
    
    print(f"Loaded {len(file_list)} LDAP files")
    return ldap_data, file_list


def build_user_timeline(ldap_data, file_list):
    """
    Build a timeline for each user showing their presence in each month
    
    Returns:
        dict: {user_id: [(year, month, record), ...]} - User timeline
    """
    print("Building user timeline...")
    
    user_timeline = defaultdict(list)
    
    for year, month, filename in file_list:
        if year in ldap_data and month in ldap_data[year]:
            df = ldap_data[year][month]
            for _, row in df.iterrows():
                user_id = row['user_id']
                user_timeline[user_id].append((year, month, row))
    
    # Sort timeline for each user
    for user_id in user_timeline:
        user_timeline[user_id].sort()
    
    print(f"Built timeline for {len(user_timeline)} users")
    return user_timeline


def get_user_info_at_time(user_timeline, user_id, target_year, target_month):
    """
    Get user information at a specific time point
    
    Returns:
        dict: User information at the specified time, or None if user doesn't exist
    """
    if user_id not in user_timeline:
        return None
    
    # Find the most recent record before or at the target time
    user_records = user_timeline[user_id]
    
    # Filter records up to the target time
    valid_records = [(y, m, r) for y, m, r in user_records 
                     if (y < target_year) or (y == target_year and m <= target_month)]
    
    if not valid_records:
        return None
    
    # Get the most recent record
    _, _, record = valid_records[-1]
    
    return {
        'user_id': user_id,
        'employee_name': record['employee_name'],
        'email': record['email'],
        'role': record['role'],
        'business_unit': record['business_unit'],
        'functional_unit': record['functional_unit'],
        'department': record['department'],
        'team': record['team'],
        'supervisor': record['supervisor']
    }


def is_user_terminated(user_timeline, user_id, check_year, check_month):
    """
    Check if a user was terminated by a specific time
    
    Returns:
        bool: True if user was terminated before or during the check period
        tuple: (termination_year, termination_month) if terminated, None otherwise
    """
    if user_id not in user_timeline:
        return False, None
    
    user_records = user_timeline[user_id]
    
    if not user_records:
        return False, None
    
    # Get the last record for this user
    last_year, last_month, _ = user_records[-1]
    
    # Check if the last record is before the check period
    if (last_year < check_year) or (last_year == check_year and last_month < check_month):
        return True, (last_year, last_month)
    
    # Check if user exists in the check period
    exists_in_period = any(
        (y == check_year and m == check_month) 
        for y, m, _ in user_records
    )
    
    if not exists_in_period:
        # User's last record is after check period, but not in check period
        # This might indicate termination, but we need to check more carefully
        # For now, if last record is before check period, consider terminated
        if (last_year < check_year) or (last_year == check_year and last_month < check_month):
            return True, (last_year, last_month)
    
    return False, None


def get_user_termination_date(user_timeline, user_id):
    """
    Get the termination date for a user (last month they appeared in LDAP)
    
    Returns:
        tuple: (year, month) of last appearance, or None if user still active
    """
    if user_id not in user_timeline:
        return None
    
    user_records = user_timeline[user_id]
    if not user_records:
        return None
    
    last_year, last_month, _ = user_records[-1]
    
    # Get the last file in the dataset to determine if this is truly termination
    # For now, return the last appearance date
    return (last_year, last_month)


def create_user_summary(ldap_data, file_list, output_file=None):
    """
    Create a comprehensive summary of all users with their roles, departments,
    and termination status
    """
    print("Creating user summary...")
    
    user_timeline = build_user_timeline(ldap_data, file_list)
    
    # Get the last month in the dataset
    if not file_list:
        print("No LDAP files found!")
        return
    
    last_year, last_month, _ = file_list[-1]
    
    summary_records = []
    
    for user_id, records in user_timeline.items():
        if not records:
            continue
        
        # Get first and last appearance
        first_year, first_month, first_record = records[0]
        last_app_year, last_app_month, last_record = records[-1]
        
        # Determine if terminated (not in the last month)
        is_term = (last_app_year < last_year) or \
                 (last_app_year == last_year and last_app_month < last_month)
        
        summary_record = {
            'user_id': user_id,
            'employee_name': last_record['employee_name'],
            'email': last_record['email'],
            'role': last_record['role'],
            'business_unit': last_record['business_unit'],
            'functional_unit': last_record['functional_unit'],
            'department': last_record['department'],
            'team': last_record['team'],
            'supervisor': last_record['supervisor'],
            'first_appearance': f"{first_year}-{first_month:02d}",
            'last_appearance': f"{last_app_year}-{last_app_month:02d}",
            'is_terminated': is_term,
            'termination_date': f"{last_app_year}-{last_app_month:02d}" if is_term else None
        }
        summary_records.append(summary_record)
    
    summary_df = pd.DataFrame(summary_records)
    
    print(f"\nSummary Statistics:")
    print(f"  Total users: {len(summary_df)}")
    print(f"  Terminated users: {summary_df['is_terminated'].sum()}")
    print(f"  Active users: {(~summary_df['is_terminated']).sum()}")
    
    if output_file:
        print(f"\nSaving summary to {output_file}...")
        summary_df.to_csv(output_file, index=False)
        print("Save completed!")
    else:
        default_output = 'ldap_user_summary.csv'
        print(f"\nSaving summary to {default_output}...")
        summary_df.to_csv(default_output, index=False)
        print("Save completed!")
        return default_output
    
    return output_file


def query_user_info(ldap_data, file_list, user_id, target_date=None):
    """
    Query information about a specific user at a specific time
    
    Parameters:
        user_id: User ID to query
        target_date: Date string in format 'YYYY-MM' or 'YYYY-MM-DD', 
                     or None for current/latest info
    """
    user_timeline = build_user_timeline(ldap_data, file_list)
    
    if target_date:
        # Parse target date
        try:
            if len(target_date) == 7:  # YYYY-MM
                target_year, target_month = map(int, target_date.split('-'))
            elif len(target_date) == 10:  # YYYY-MM-DD
                target_year, target_month = map(int, target_date.split('-')[:2])
            else:
                print(f"Invalid date format: {target_date}. Use YYYY-MM or YYYY-MM-DD")
                return None
        except ValueError:
            print(f"Invalid date format: {target_date}")
            return None
    else:
        # Use the last month in the dataset
        if not file_list:
            print("No LDAP files found!")
            return None
        target_year, target_month, _ = file_list[-1]
    
    info = get_user_info_at_time(user_timeline, user_id, target_year, target_month)
    
    if info:
        print(f"\nUser Information for {user_id} at {target_year}-{target_month:02d}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Check termination status
        is_term, term_date = is_user_terminated(user_timeline, user_id, target_year, target_month)
        if is_term:
            print(f"  Termination Status: Terminated (last seen: {term_date[0]}-{term_date[1]:02d})")
        else:
            print(f"  Termination Status: Active")
    else:
        print(f"User {user_id} not found at {target_year}-{target_month:02d}")
    
    return info


def main():
    parser = argparse.ArgumentParser(
        description='Process LDAP data to determine user roles, departments, and termination status'
    )
    parser.add_argument('--ldap-dir', '-d', type=str, default='../r4.2/LDAP',
                       help='Directory containing LDAP CSV files (default: ../r4.2/LDAP)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file for user summary (default: ldap_user_summary.csv)')
    parser.add_argument('--user', '-u', type=str, default=None,
                       help='Query specific user ID')
    parser.add_argument('--date', '-t', type=str, default=None,
                       help='Target date for query (format: YYYY-MM or YYYY-MM-DD)')
    parser.add_argument('--summary', '-s', action='store_true',
                       help='Generate user summary CSV')
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.exists(args.ldap_dir):
        print(f"Error: Directory not found {args.ldap_dir}")
        return
    
    # Load LDAP data
    ldap_data, file_list = load_ldap_data(args.ldap_dir)
    
    if not file_list:
        print("No LDAP files found!")
        return
    
    # Query specific user
    if args.user:
        query_user_info(ldap_data, file_list, args.user, args.date)
    
    # Generate summary
    if args.summary or (not args.user):
        create_user_summary(ldap_data, file_list, args.output)


if __name__ == '__main__':
    main()
