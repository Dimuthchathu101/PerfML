#!/usr/bin/env python3
"""
Test script to demonstrate SQLite functionality in the Performance ML System
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collection.data_pipeline import DataStorage
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sqlite_functionality():
    """Test the SQLite database functionality"""
    print("🧪 Testing SQLite Database Functionality")
    print("=" * 50)
    
    # Initialize database
    print("1. Initializing SQLite database...")
    db = DataStorage("test_performance_data.db")
    
    # Create sample test data
    print("2. Creating sample test data...")
    sample_data = [
        {
            'test_id': 'test_001',
            'test_type': 'api',
            'timestamp': datetime.now().isoformat(),
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
        },
        {
            'test_id': 'test_002',
            'test_type': 'frontend',
            'timestamp': datetime.now().isoformat(),
            'total_requests': 500,
            'total_failures': 2,
            'error_rate': 0.004,
            'avg_response_time': 200.0,
            'max_response_time': 800.0,
            'min_response_time': 80.0,
            'requests_per_sec': 50.0,
            'concurrent_users': 25,
            'spawn_rate': 5.0,
            'duration': 600,
            'network_condition': '4g',
            'browser': 'firefox',
            'target_url': 'http://localhost:5000/',
            'performance_score': 0.88,
            'accessibility_score': 0.95,
            'best_practices_score': 0.90,
            'seo_score': 0.85,
            'status': 'success'
        },
        {
            'test_id': 'test_003',
            'test_type': 'load',
            'timestamp': datetime.now().isoformat(),
            'total_requests': 2000,
            'total_failures': 50,
            'error_rate': 0.025,
            'avg_response_time': 800.0,
            'max_response_time': 2000.0,
            'min_response_time': 200.0,
            'requests_per_sec': 200.0,
            'concurrent_users': 200,
            'spawn_rate': 20.0,
            'duration': 900,
            'network_condition': '3g',
            'browser': 'safari',
            'target_url': 'http://localhost:5000/api/data/test',
            'performance_score': 0.75,
            'accessibility_score': 0.82,
            'best_practices_score': 0.78,
            'seo_score': 0.80,
            'status': 'success'
        }
    ]
    
    # Store data
    print("3. Storing test data in SQLite...")
    db.store_results_batch(sample_data)
    
    # Retrieve all data
    print("4. Retrieving all test results...")
    all_results = db.get_results()
    print(f"   Retrieved {len(all_results)} test results")
    
    # Get recent results
    print("5. Retrieving recent results (last 24 hours)...")
    recent_results = db.get_recent_results(hours=24)
    print(f"   Retrieved {len(recent_results)} recent results")
    
    # Get test summary
    print("6. Generating test summary...")
    summary = db.get_test_summary()
    print(f"   Total tests: {summary['total_tests']}")
    print(f"   Successful tests: {summary['successful_tests']}")
    print(f"   Failed tests: {summary['failed_tests']}")
    print(f"   Success rate: {summary['success_rate']:.2%}")
    print(f"   Average latency: {summary['avg_latency']:.2f}ms")
    print(f"   Average error rate: {summary['avg_error_rate']:.2%}")
    print(f"   Average throughput: {summary['avg_throughput']:.2f} req/s")
    
    # Filter results
    print("7. Filtering results by test type...")
    api_results = db.get_results(filters={'test_type': 'api'})
    print(f"   API tests: {len(api_results)}")
    
    frontend_results = db.get_results(filters={'test_type': 'frontend'})
    print(f"   Frontend tests: {len(frontend_results)}")
    
    # Export to CSV
    print("8. Exporting results to CSV...")
    csv_file = db.export_to_csv("test_results_export.csv")
    print(f"   Exported to: {csv_file}")
    
    # Create backup
    print("9. Creating database backup...")
    backup_file = db.backup_database()
    print(f"   Backup created: {backup_file}")
    
    # Display sample data
    print("\n10. Sample data from database:")
    print("-" * 30)
    print(all_results[['test_id', 'test_type', 'avg_response_time', 'error_rate', 'status']].head())
    
    print("\n✅ SQLite functionality test completed successfully!")
    print(f"📊 Database file: {db.db_path}")
    print(f"📁 Backup file: {backup_file}")
    print(f"📄 CSV export: {csv_file}")
    
    return True

if __name__ == "__main__":
    try:
        test_sqlite_functionality()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
