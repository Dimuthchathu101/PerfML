"""
Data Pipeline for Performance ML System
Phase 1: Data Collection & Preprocessing - Step 2
"""

import pandas as pd
import sqlite3
import json
import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .parameterized_tests import TestConfiguration, TestType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceTestRunner:
    """Runs performance tests and collects results"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.results = []
    
    def run_locust_test(self, config: TestConfiguration) -> Dict[str, Any]:
        """Run a Locust load test"""
        try:
            # Create Locust script dynamically
            script_content = self._generate_locust_script(config)
            script_path = f"temp_locust_{config.test_id}.py"
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Run Locust
            cmd = [
                "locust",
                "-f", script_path,
                "--headless",
                "--users", str(config.concurrent_users),
                "--spawn-rate", str(config.spawn_rate),
                "--run-time", f"{config.duration}s",
                "--html", f"results_{config.test_id}.html",
                "--csv", f"results_{config.test_id}"
            ]
            
            logger.info(f"Running Locust test: {config.test_id}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=config.duration + 60)
            
            # Parse results
            test_results = self._parse_locust_results(config, f"results_{config.test_id}_stats.csv")
            
            # Cleanup
            os.remove(script_path)
            os.remove(f"results_{config.test_id}.html")
            os.remove(f"results_{config.test_id}_stats.csv")
            os.remove(f"results_{config.test_id}_failures.csv")
            
            return test_results
            
        except Exception as e:
            logger.error(f"Error running Locust test {config.test_id}: {e}")
            return self._create_error_result(config, str(e))
    
    def run_lighthouse_test(self, config: TestConfiguration) -> Dict[str, Any]:
        """Run a Lighthouse frontend test"""
        try:
            if not config.target_url:
                raise ValueError("Target URL required for Lighthouse test")
            
            # Run Lighthouse CI
            cmd = [
                "lhci", "autorun",
                "--config", self._generate_lighthouse_config(config),
                "--url", config.target_url
            ]
            
            logger.info(f"Running Lighthouse test: {config.test_id}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Parse Lighthouse results
            test_results = self._parse_lighthouse_results(config, result.stdout)
            
            return test_results
            
        except Exception as e:
            logger.error(f"Error running Lighthouse test {config.test_id}: {e}")
            return self._create_error_result(config, str(e))
    
    def _generate_locust_script(self, config: TestConfiguration) -> str:
        """Generate Locust script for the test configuration"""
        script = f"""
from locust import HttpUser, task, between
import json
import random

class PerformanceTestUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        self.client.base_url = "{self.base_url}"
    
    @task(1)
    def health_check(self):
        self.client.get("/api/health")
    
    @task(2)
    def get_data(self):
        keys = ["test", "heavy_data", "medium_data", "light_data"]
        key = random.choice(keys)
        self.client.get(f"/api/data/{{key}}")
    
    @task(1)
    def search(self):
        queries = ["performance", "test", "load", "api"]
        query = random.choice(queries)
        self.client.get(f"/api/search?q={{query}}")
    
    @task(1)
    def get_users(self):
        page = random.randint(1, 10)
        limit = random.randint(5, 20)
        self.client.get(f"/api/users?page={{page}}&limit={{limit}}")
    
    @task(1)
    def upload_file(self):
        data = {{"key": "test", "value": {{"data": "test_data"}}}}
        self.client.post("/api/data", json=data)
"""
        return script
    
    def _generate_lighthouse_config(self, config: TestConfiguration) -> str:
        """Generate Lighthouse CI configuration"""
        config_content = f"""
module.exports = {{
  ci: {{
    collect: {{
      url: ['{config.target_url}'],
      numberOfRuns: 3,
      settings: {{
        chromeFlags: '--no-sandbox --disable-dev-shm-usage',
        preset: '{config.network_condition.value}'
      }}
    }},
    assert: {{
      assertions: {{
        'categories:performance': ['warn', {{minScore: 0.8}}],
        'categories:accessibility': ['warn', {{minScore: 0.8}}],
        'categories:best-practices': ['warn', {{minScore: 0.8}}],
        'categories:seo': ['warn', {{minScore: 0.8}}]
      }}
    }},
    upload: {{
      target: 'temporary-public-storage'
    }}
  }}
}};
"""
        config_path = f"lighthouse_config_{config.test_id}.js"
        with open(config_path, 'w') as f:
            f.write(config_content)
        return config_path
    
    def _parse_locust_results(self, config: TestConfiguration, csv_path: str) -> Dict[str, Any]:
        """Parse Locust CSV results"""
        try:
            df = pd.read_csv(csv_path)
            
            # Calculate metrics
            total_requests = df['request_count'].sum()
            total_failures = df['failure_count'].sum()
            avg_response_time = df['avg_response_time'].mean()
            max_response_time = df['max_response_time'].max()
            min_response_time = df['min_response_time'].min()
            requests_per_sec = df['requests_per_sec'].mean()
            
            return {
                'test_id': config.test_id,
                'test_type': config.test_type.value,
                'timestamp': datetime.now().isoformat(),
                'total_requests': int(total_requests),
                'total_failures': int(total_failures),
                'error_rate': float(total_failures / total_requests) if total_requests > 0 else 0.0,
                'avg_response_time': float(avg_response_time),
                'max_response_time': float(max_response_time),
                'min_response_time': float(min_response_time),
                'requests_per_sec': float(requests_per_sec),
                'concurrent_users': config.concurrent_users,
                'spawn_rate': config.spawn_rate,
                'duration': config.duration,
                'network_condition': config.network_condition.value,
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"Error parsing Locust results: {e}")
            return self._create_error_result(config, f"Parse error: {e}")
    
    def _parse_lighthouse_results(self, config: TestConfiguration, output: str) -> Dict[str, Any]:
        """Parse Lighthouse results"""
        try:
            # Extract scores from output (simplified parsing)
            lines = output.split('\n')
            scores = {}
            
            for line in lines:
                if 'Performance' in line and 'score' in line.lower():
                    scores['performance'] = self._extract_score(line)
                elif 'Accessibility' in line and 'score' in line.lower():
                    scores['accessibility'] = self._extract_score(line)
                elif 'Best Practices' in line and 'score' in line.lower():
                    scores['best_practices'] = self._extract_score(line)
                elif 'SEO' in line and 'score' in line.lower():
                    scores['seo'] = self._extract_score(line)
            
            return {
                'test_id': config.test_id,
                'test_type': config.test_type.value,
                'timestamp': datetime.now().isoformat(),
                'performance_score': scores.get('performance', 0.0),
                'accessibility_score': scores.get('accessibility', 0.0),
                'best_practices_score': scores.get('best_practices', 0.0),
                'seo_score': scores.get('seo', 0.0),
                'network_condition': config.network_condition.value,
                'browser': config.browser.value if config.browser else None,
                'target_url': config.target_url,
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"Error parsing Lighthouse results: {e}")
            return self._create_error_result(config, f"Parse error: {e}")
    
    def _extract_score(self, line: str) -> float:
        """Extract score from Lighthouse output line"""
        try:
            # Simple regex-like extraction
            if 'score' in line.lower():
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.lower() == 'score':
                        if i + 1 < len(parts):
                            return float(parts[i + 1])
            return 0.0
        except:
            return 0.0
    
    def _create_error_result(self, config: TestConfiguration, error: str) -> Dict[str, Any]:
        """Create error result when test fails"""
        return {
            'test_id': config.test_id,
            'test_type': config.test_type.value,
            'timestamp': datetime.now().isoformat(),
            'error': error,
            'status': 'error',
            'concurrent_users': config.concurrent_users,
            'spawn_rate': config.spawn_rate,
            'duration': config.duration,
            'network_condition': config.network_condition.value
        }

class DataStorage:
    """Handles data storage in SQLite database with enhanced features"""
    
    def __init__(self, db_path: str = "performance_data.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with enhanced schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create main test_results table with enhanced schema
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    test_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    total_requests INTEGER,
                    total_failures INTEGER,
                    error_rate REAL,
                    avg_response_time REAL,
                    max_response_time REAL,
                    min_response_time REAL,
                    requests_per_sec REAL,
                    concurrent_users INTEGER,
                    spawn_rate REAL,
                    duration INTEGER,
                    network_condition TEXT,
                    browser TEXT,
                    target_url TEXT,
                    performance_score REAL,
                    accessibility_score REAL,
                    best_practices_score REAL,
                    seo_score REAL,
                    status TEXT,
                    error TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better query performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_test_id ON test_results(test_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON test_results(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_test_type ON test_results(test_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON test_results(status)')
            
            # Create metadata table for system information
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE NOT NULL,
                    value TEXT,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create model_results table for storing ML model results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    test_id TEXT NOT NULL,
                    prediction REAL,
                    actual_value REAL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"SQLite database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def store_result(self, result: Dict[str, Any]):
        """Store a single test result with validation"""
        try:
            # Validate required fields
            required_fields = ['test_id', 'test_type', 'timestamp']
            for field in required_fields:
                if field not in result or result[field] is None:
                    raise ValueError(f"Missing required field: {field}")
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert result to database format with metadata
            metadata = json.dumps({k: v for k, v in result.items() 
                                 if k not in ['test_id', 'test_type', 'timestamp', 'status', 'error']})
            
            cursor.execute('''
                INSERT INTO test_results (
                    test_id, test_type, timestamp, total_requests, total_failures,
                    error_rate, avg_response_time, max_response_time, min_response_time,
                    requests_per_sec, concurrent_users, spawn_rate, duration,
                    network_condition, browser, target_url, performance_score,
                    accessibility_score, best_practices_score, seo_score, status, error, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.get('test_id'),
                result.get('test_type'),
                result.get('timestamp'),
                result.get('total_requests'),
                result.get('total_failures'),
                result.get('error_rate'),
                result.get('avg_response_time'),
                result.get('max_response_time'),
                result.get('min_response_time'),
                result.get('requests_per_sec'),
                result.get('concurrent_users'),
                result.get('spawn_rate'),
                result.get('duration'),
                result.get('network_condition'),
                result.get('browser'),
                result.get('target_url'),
                result.get('performance_score'),
                result.get('accessibility_score'),
                result.get('best_practices_score'),
                result.get('seo_score'),
                result.get('status'),
                result.get('error'),
                metadata
            ))
            
            conn.commit()
            conn.close()
            logger.debug(f"Stored test result: {result.get('test_id')}")
            
        except Exception as e:
            logger.error(f"Error storing result: {e}")
            raise
    
    def store_results_batch(self, results: List[Dict[str, Any]]):
        """Store multiple test results efficiently using batch insert"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Prepare batch data
            batch_data = []
            for result in results:
                # Validate required fields
                required_fields = ['test_id', 'test_type', 'timestamp']
                for field in required_fields:
                    if field not in result or result[field] is None:
                        logger.warning(f"Skipping result with missing field {field}: {result}")
                        continue
                
                metadata = json.dumps({k: v for k, v in result.items() 
                                     if k not in ['test_id', 'test_type', 'timestamp', 'status', 'error']})
                
                batch_data.append((
                    result.get('test_id'),
                    result.get('test_type'),
                    result.get('timestamp'),
                    result.get('total_requests'),
                    result.get('total_failures'),
                    result.get('error_rate'),
                    result.get('avg_response_time'),
                    result.get('max_response_time'),
                    result.get('min_response_time'),
                    result.get('requests_per_sec'),
                    result.get('concurrent_users'),
                    result.get('spawn_rate'),
                    result.get('duration'),
                    result.get('network_condition'),
                    result.get('browser'),
                    result.get('target_url'),
                    result.get('performance_score'),
                    result.get('accessibility_score'),
                    result.get('best_practices_score'),
                    result.get('seo_score'),
                    result.get('status'),
                    result.get('error'),
                    metadata
                ))
            
            # Batch insert
            cursor.executemany('''
                INSERT INTO test_results (
                    test_id, test_type, timestamp, total_requests, total_failures,
                    error_rate, avg_response_time, max_response_time, min_response_time,
                    requests_per_sec, concurrent_users, spawn_rate, duration,
                    network_condition, browser, target_url, performance_score,
                    accessibility_score, best_practices_score, seo_score, status, error, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', batch_data)
            
            conn.commit()
            conn.close()
            logger.info(f"Stored {len(batch_data)} test results in batch")
            
        except Exception as e:
            logger.error(f"Error storing batch results: {e}")
            raise
    
    def get_results(self, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """Retrieve test results with optional filtering and limit"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT * FROM test_results"
            params = []
            
            if filters:
                conditions = []
                for key, value in filters.items():
                    if isinstance(value, (list, tuple)):
                        placeholders = ','.join(['?' for _ in value])
                        conditions.append(f"{key} IN ({placeholders})")
                        params.extend(value)
                    else:
                        conditions.append(f"{key} = ?")
                        params.append(value)
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            logger.debug(f"Retrieved {len(df)} results from database")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving results: {e}")
            raise
    
    def get_recent_results(self, hours: int = 24) -> pd.DataFrame:
        """Get results from the last N hours"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Calculate timestamp for N hours ago
            from datetime import datetime, timedelta
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            query = "SELECT * FROM test_results WHERE timestamp >= ? ORDER BY timestamp DESC"
            df = pd.read_sql_query(query, conn, params=[cutoff_time])
            conn.close()
            
            logger.info(f"Retrieved {len(df)} results from last {hours} hours")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving recent results: {e}")
            raise
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all test results"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get basic counts
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM test_results")
            total_tests = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM test_results WHERE status = 'success'")
            successful_tests = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM test_results WHERE status = 'error'")
            failed_tests = cursor.fetchone()[0]
            
            # Get performance metrics
            cursor.execute("""
                SELECT 
                    AVG(avg_response_time) as avg_latency,
                    MAX(avg_response_time) as max_latency,
                    MIN(avg_response_time) as min_latency,
                    AVG(error_rate) as avg_error_rate,
                    AVG(requests_per_sec) as avg_throughput
                FROM test_results 
                WHERE status = 'success'
            """)
            metrics = cursor.fetchone()
            
            conn.close()
            
            summary = {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': failed_tests,
                'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
                'avg_latency': metrics[0],
                'max_latency': metrics[1],
                'min_latency': metrics[2],
                'avg_error_rate': metrics[3],
                'avg_throughput': metrics[4]
            }
            
            logger.info(f"Generated test summary: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating test summary: {e}")
            raise
    
    def export_to_csv(self, filename: str, filters: Optional[Dict[str, Any]] = None):
        """Export results to CSV with enhanced error handling"""
        try:
            df = self.get_results(filters)
            df.to_csv(filename, index=False)
            logger.info(f"Exported {len(df)} results to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            raise
    
    def backup_database(self, backup_path: str = None):
        """Create a backup of the database"""
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"performance_data_backup_{timestamp}.db"
            
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Error backing up database: {e}")
            raise
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old test results"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate cutoff date
            from datetime import datetime, timedelta
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Delete old records
            cursor.execute("DELETE FROM test_results WHERE timestamp < ?", [cutoff_date])
            deleted_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cleaned up {deleted_count} old test results (older than {days} days)")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            raise

class AutomatedTestRunner:
    """Orchestrates automated test runs"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.test_runner = PerformanceTestRunner(base_url)
        self.storage = DataStorage()
        self.base_url = base_url
    
    def run_test_suite(self, configs: List[TestConfiguration], max_workers: int = 4):
        """Run a suite of tests with parallel execution"""
        logger.info(f"Starting test suite with {len(configs)} configurations")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tests
            future_to_config = {}
            for config in configs:
                if config.test_type == TestType.FRONTEND:
                    future = executor.submit(self.test_runner.run_lighthouse_test, config)
                else:
                    future = executor.submit(self.test_runner.run_locust_test, config)
                future_to_config[future] = config
            
            # Collect results
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed test: {config.test_id}")
                except Exception as e:
                    logger.error(f"Test {config.test_id} failed: {e}")
                    error_result = self.test_runner._create_error_result(config, str(e))
                    results.append(error_result)
        
        # Store results
        self.storage.store_results_batch(results)
        
        logger.info(f"Test suite completed. {len(results)} results stored.")
        return results
    
    def run_scheduled_tests(self, config_file: str, schedule_interval: int = 3600):
        """Run tests on a schedule"""
        logger.info(f"Starting scheduled test runner (interval: {schedule_interval}s)")
        
        while True:
            try:
                # Load configurations
                from .parameterized_tests import TestParameterizer
                parameterizer = TestParameterizer(self.base_url)
                configs = parameterizer.load_configs(config_file)
                
                # Run tests
                self.run_test_suite(configs)
                
                # Wait for next run
                time.sleep(schedule_interval)
                
            except KeyboardInterrupt:
                logger.info("Scheduled test runner stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in scheduled test runner: {e}")
                time.sleep(60)  # Wait before retrying

def main():
    """Example usage of the data pipeline"""
    # Create test configurations
    from .parameterized_tests import TestParameterizer
    parameterizer = TestParameterizer()
    configs = parameterizer.generate_all_configs()
    
    # Save configurations
    parameterizer.save_configs(configs, "test_configurations.json")
    
    # Run automated tests
    runner = AutomatedTestRunner()
    
    # Run a subset of tests for demonstration
    demo_configs = configs[:5]  # First 5 tests
    results = runner.run_test_suite(demo_configs)
    
    # Export results
    runner.storage.export_to_csv("demo_results.csv")
    
    print(f"Demo completed. {len(results)} tests run.")

if __name__ == "__main__":
    main()
