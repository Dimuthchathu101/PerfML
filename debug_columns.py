#!/usr/bin/env python3
"""
Debug script to check columns in the data pipeline
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collection.data_pipeline import DataStorage
from data_collection.feature_engineering import PerformanceFeatureEngineer
import pandas as pd

def debug_columns():
    """Debug the column issue"""
    print("🔍 Debugging Column Issue")
    print("=" * 40)
    
    # Initialize components
    db = DataStorage("debug_performance_data.db")
    feature_engineer = PerformanceFeatureEngineer()
    
    # Create sample data
    sample_data = [
        {
            'test_id': 'test_001',
            'test_type': 'api',
            'timestamp': '2025-08-16T10:30:00',
            'total_requests': 1000,
            'total_failures': 5,
            'error_rate': 0.005,
            'avg_response_time': 150.5,
            'max_response_time': 500.0,
            'min_response_time': 50.0,
            'requests_per_sec': 100.0,
            'concurrent_users': 50,
            'spawn_rate': 10.0,
            'duration': 300,
            'network_condition': 'wifi',
            'browser': 'chrome',
            'target_url': 'http://localhost:5000/api/health',
            'performance_score': 0.95,
            'accessibility_score': 0.98,
            'best_practices_score': 0.92,
            'seo_score': 0.89,
            'status': 'success'
        }
    ]
    
    # Store in database
    db.store_results_batch(sample_data)
    
    # Get raw data
    raw_data = db.get_results()
    print(f"1. Raw data columns ({len(raw_data.columns)}):")
    print(f"   {list(raw_data.columns)}")
    print(f"   test_id in columns: {'test_id' in raw_data.columns}")
    print()
    
    # Engineer features
    engineered_data = feature_engineer.engineer_features(raw_data)
    print(f"2. Engineered data columns ({len(engineered_data.columns)}):")
    print(f"   {list(engineered_data.columns)}")
    print(f"   test_id in columns: {'test_id' in engineered_data.columns}")
    print()
    
    # Create target variables
    data_with_targets = feature_engineer.create_target_variables(engineered_data)
    print(f"3. Data with targets columns ({len(data_with_targets.columns)}):")
    print(f"   {list(data_with_targets.columns)}")
    print(f"   test_id in columns: {'test_id' in data_with_targets.columns}")
    print()
    
    # Normalize features
    normalized_data = feature_engineer.normalize_features(data_with_targets)
    print(f"4. Normalized data columns ({len(normalized_data.columns)}):")
    print(f"   {list(normalized_data.columns)}")
    print(f"   test_id in columns: {'test_id' in normalized_data.columns}")
    print()
    
    # Check if test_id is in any of the dataframes
    print("5. Summary:")
    print(f"   Raw data has test_id: {'test_id' in raw_data.columns}")
    print(f"   Engineered data has test_id: {'test_id' in engineered_data.columns}")
    print(f"   Data with targets has test_id: {'test_id' in data_with_targets.columns}")
    print(f"   Normalized data has test_id: {'test_id' in normalized_data.columns}")

if __name__ == "__main__":
    debug_columns()
