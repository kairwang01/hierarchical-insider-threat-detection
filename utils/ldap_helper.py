#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LDAP Helper Module
Provides convenient functions to query LDAP data for user roles, departments,
and termination status
"""

import pandas as pd
import os
import glob
from datetime import datetime
from collections import defaultdict


class LDAPProcessor:
    """Class to process and query LDAP data"""
    
    def __init__(self, ldap_dir='../r4.2/LDAP'):
        """
        Initialize LDAP processor
        
        Parameters:
            ldap_dir: Directory containing LDAP CSV files
        """
        self.ldap_dir = ldap_dir
        self.ldap_data = {}
        self.file_list = []
        self.user_timeline = {}
        self._load_data()
    
    def _parse_month_from_filename(self, filename):
        """Extract year and month from filename"""
        basename = os.path.basename(filename)
        date_str = basename.replace('.csv', '')
        try:
            year, month = map(int, date_str.split('-'))
            return year, month
        except ValueError:
            return None, None
    
    def _load_data(self):
        """Load all LDAP CSV files"""
        if not os.path.exists(self.ldap_dir):
            raise FileNotFoundError(f"LDAP directory not found: {self.ldap_dir}")
        
        pattern = os.path.join(self.ldap_dir, '*.csv')
        files = glob.glob(pattern)
        
        for filename in files:
            year, month = self._parse_month_from_filename(filename)
            if year is None or month is None:
                continue
            
            try:
                df = pd.read_csv(filename)
                if year not in self.ldap_data:
                    self.ldap_data[year] = {}
                self.ldap_data[year][month] = df
                self.file_list.append((year, month, filename))
            except Exception as e:
                print(f"Warning: Error loading {filename}: {e}")
        
        self.file_list.sort()
        self._build_timeline()
    
    def _build_timeline(self):
        """Build user timeline from loaded data"""
        self.user_timeline = defaultdict(list)
        
        for year, month, _ in self.file_list:
            if year in self.ldap_data and month in self.ldap_data[year]:
                df = self.ldap_data[year][month]
                for _, row in df.iterrows():
                    user_id = row['user_id']
                    self.user_timeline[user_id].append((year, month, row))
        
        # Sort timeline for each user
        for user_id in self.user_timeline:
            self.user_timeline[user_id].sort()
    
    def get_user_info(self, user_id, target_date=None):
        """
        Get user information at a specific date
        
        Parameters:
            user_id: User ID to query
            target_date: Date string in format 'YYYY-MM' or datetime object,
                        or None for latest info
        
        Returns:
            dict: User information or None if not found
        """
        if target_date is None:
            # Use the last month in the dataset
            if not self.file_list:
                return None
            target_year, target_month, _ = self.file_list[-1]
        else:
            # Parse target date
            if isinstance(target_date, str):
                if len(target_date) == 7:  # YYYY-MM
                    target_year, target_month = map(int, target_date.split('-'))
                elif len(target_date) == 10:  # YYYY-MM-DD
                    target_year, target_month = map(int, target_date.split('-')[:2])
                else:
                    raise ValueError(f"Invalid date format: {target_date}")
            elif isinstance(target_date, datetime):
                target_year = target_date.year
                target_month = target_date.month
            else:
                raise ValueError(f"Invalid date type: {type(target_date)}")
        
        if user_id not in self.user_timeline:
            return None
        
        # Find the most recent record before or at the target time
        user_records = self.user_timeline[user_id]
        valid_records = [
            (y, m, r) for y, m, r in user_records
            if (y < target_year) or (y == target_year and m <= target_month)
        ]
        
        if not valid_records:
            return None
        
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
    
    def get_user_role(self, user_id, target_date=None):
        """Get user's role at a specific date"""
        info = self.get_user_info(user_id, target_date)
        return info['role'] if info else None
    
    def get_user_department(self, user_id, target_date=None):
        """Get user's department at a specific date"""
        info = self.get_user_info(user_id, target_date)
        return info['department'] if info else None
    
    def is_terminated(self, user_id, check_date=None):
        """
        Check if user was terminated by a specific date
        
        Parameters:
            user_id: User ID to check
            check_date: Date to check (format: 'YYYY-MM' or datetime),
                       or None for latest
        
        Returns:
            tuple: (is_terminated: bool, termination_date: (year, month) or None)
        """
        if check_date is None:
            if not self.file_list:
                return False, None
            check_year, check_month, _ = self.file_list[-1]
        else:
            if isinstance(check_date, str):
                if len(check_date) == 7:  # YYYY-MM
                    check_year, check_month = map(int, check_date.split('-'))
                elif len(check_date) == 10:  # YYYY-MM-DD
                    check_year, check_month = map(int, check_date.split('-')[:2])
                else:
                    raise ValueError(f"Invalid date format: {check_date}")
            elif isinstance(check_date, datetime):
                check_year = check_date.year
                check_month = check_date.month
            else:
                raise ValueError(f"Invalid date type: {type(check_date)}")
        
        if user_id not in self.user_timeline:
            return False, None
        
        user_records = self.user_timeline[user_id]
        if not user_records:
            return False, None
        
        last_year, last_month, _ = user_records[-1]
        
        # Check if last record is before check period
        if (last_year < check_year) or (last_year == check_year and last_month < check_month):
            return True, (last_year, last_month)
        
        # Check if user exists in the check period
        exists_in_period = any(
            (y == check_year and m == check_month)
            for y, m, _ in user_records
        )
        
        if not exists_in_period:
            if (last_year < check_year) or (last_year == check_year and last_month < check_month):
                return True, (last_year, last_month)
        
        return False, None
    
    def get_termination_date(self, user_id):
        """
        Get the termination date for a user (last month they appeared)
        
        Returns:
            tuple: (year, month) of last appearance, or None if still active
        """
        if user_id not in self.user_timeline:
            return None
        
        user_records = self.user_timeline[user_id]
        if not user_records:
            return None
        
        last_year, last_month, _ = user_records[-1]
        
        # Check if this is truly termination (not just the last month in dataset)
        if not self.file_list:
            return None
        
        last_file_year, last_file_month, _ = self.file_list[-1]
        
        # If last appearance is before the last file, user is terminated
        if (last_year < last_file_year) or \
           (last_year == last_file_year and last_month < last_file_month):
            return (last_year, last_month)
        
        return None  # User is still active
    
    def get_all_users(self, target_date=None):
        """
        Get all users active at a specific date
        
        Returns:
            list: List of user IDs
        """
        if target_date is None:
            if not self.file_list:
                return []
            target_year, target_month, _ = self.file_list[-1]
        else:
            if isinstance(target_date, str):
                if len(target_date) == 7:
                    target_year, target_month = map(int, target_date.split('-'))
                elif len(target_date) == 10:
                    target_year, target_month = map(int, target_date.split('-')[:2])
                else:
                    raise ValueError(f"Invalid date format: {target_date}")
            elif isinstance(target_date, datetime):
                target_year = target_date.year
                target_month = target_date.month
            else:
                raise ValueError(f"Invalid date type: {type(target_date)}")
        
        active_users = []
        for user_id, records in self.user_timeline.items():
            valid_records = [
                (y, m, r) for y, m, r in records
                if (y < target_year) or (y == target_year and m <= target_month)
            ]
            if valid_records:
                # Check if user was still active at target date
                last_year, last_month, _ = valid_records[-1]
                if (last_year == target_year and last_month == target_month) or \
                   (last_year > target_year) or \
                   (last_year == target_year and last_month > target_month):
                    active_users.append(user_id)
        
        return active_users


# Convenience functions for quick access
_global_processor = None


def get_ldap_processor(ldap_dir='../r4.2/LDAP'):
    """Get or create global LDAP processor instance"""
    global _global_processor
    if _global_processor is None or _global_processor.ldap_dir != ldap_dir:
        _global_processor = LDAPProcessor(ldap_dir)
    return _global_processor


def get_user_role(user_id, target_date=None, ldap_dir='../r4.2/LDAP'):
    """Quick function to get user role"""
    processor = get_ldap_processor(ldap_dir)
    return processor.get_user_role(user_id, target_date)


def get_user_department(user_id, target_date=None, ldap_dir='../r4.2/LDAP'):
    """Quick function to get user department"""
    processor = get_ldap_processor(ldap_dir)
    return processor.get_user_department(user_id, target_date)


def is_user_terminated(user_id, check_date=None, ldap_dir='../r4.2/LDAP'):
    """Quick function to check if user is terminated"""
    processor = get_ldap_processor(ldap_dir)
    return processor.is_terminated(user_id, check_date)
