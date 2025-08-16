"""
Feature Engineering for Performance ML System
Phase 1: Data Collection & Preprocessing - Step 3
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PerformanceFeatureEngineer:
    """Engineers features from raw performance test data"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_names = []
        self.derived_features = {}
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main method to engineer all features"""
        logger.info("Starting feature engineering process")
        
        # Create a copy to avoid modifying original data
        df_engineered = df.copy()
        
        # Preserve important metadata columns
        metadata_cols = ['test_id', 'timestamp', 'test_type']
        preserved_cols = [col for col in metadata_cols if col in df.columns]
        
        # Basic derived features
        df_engineered = self._create_basic_features(df_engineered)
        
        # Load-related features
        df_engineered = self._create_load_features(df_engineered)
        
        # Error and reliability features
        df_engineered = self._create_reliability_features(df_engineered)
        
        # Performance ratio features
        df_engineered = self._create_performance_ratios(df_engineered)
        
        # Time-based features
        df_engineered = self._create_time_features(df_engineered)
        
        # Network condition features
        df_engineered = self._create_network_features(df_engineered)
        
        # Frontend-specific features
        df_engineered = self._create_frontend_features(df_engineered)
        
        # Interaction features
        df_engineered = self._create_interaction_features(df_engineered)
        
        # Resource complexity features
        df_engineered = self._create_complexity_features(df_engineered)
        
        # Ensure preserved columns are still present
        for col in preserved_cols:
            if col not in df_engineered.columns and col in df.columns:
                df_engineered[col] = df[col]
        
        logger.info(f"Feature engineering completed. Original features: {len(df.columns)}, New features: {len(df_engineered.columns)}")
        
        return df_engineered
    
    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic derived features"""
        logger.info("Creating basic derived features")
        
        # Response time percentiles
        if 'avg_response_time' in df.columns:
            df['response_time_p95'] = df['avg_response_time'] * 1.5  # Approximate P95
            df['response_time_p99'] = df['avg_response_time'] * 2.0  # Approximate P99
            df['response_time_variance'] = df['max_response_time'] - df['min_response_time']
        
        # Throughput efficiency
        if all(col in df.columns for col in ['requests_per_sec', 'concurrent_users']):
            df['throughput_per_user'] = df['requests_per_sec'] / df['concurrent_users']
            df['user_efficiency'] = df['throughput_per_user'] / df['spawn_rate']
        
        # Load intensity
        if all(col in df.columns for col in ['concurrent_users', 'duration']):
            df['load_intensity'] = df['concurrent_users'] * df['spawn_rate'] / df['duration']
        
        return df
    
    def _create_load_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create load-related features"""
        logger.info("Creating load-related features")
        
        # Peak load ratio
        if 'concurrent_users' in df.columns:
            baseline_users = df['concurrent_users'].min()
            df['peak_load_ratio'] = df['concurrent_users'] / baseline_users if baseline_users > 0 else 1
        
        # Load progression
        if 'concurrent_users' in df.columns:
            df['load_progression'] = df['concurrent_users'].rank(pct=True)
        
        # Spawn rate efficiency
        if all(col in df.columns for col in ['spawn_rate', 'concurrent_users']):
            df['spawn_efficiency'] = df['concurrent_users'] / (df['spawn_rate'] * df['duration'])
        
        # Load categories
        if 'concurrent_users' in df.columns:
            df['load_category'] = pd.cut(
                df['concurrent_users'],
                bins=[0, 50, 100, 200, 500, float('inf')],
                labels=['light', 'medium', 'high', 'very_high', 'extreme']
            )
        
        return df
    
    def _create_reliability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create error and reliability features"""
        logger.info("Creating reliability features")
        
        # Error rate slope (approximation)
        if 'error_rate' in df.columns:
            df['error_rate_normalized'] = df['error_rate'] * 100  # Convert to percentage
            df['reliability_score'] = 100 - df['error_rate_normalized']
        
        # Failure patterns
        if all(col in df.columns for col in ['total_requests', 'total_failures']):
            df['success_rate'] = (df['total_requests'] - df['total_failures']) / df['total_requests']
            df['failure_ratio'] = df['total_failures'] / df['total_requests']
        
        # Error severity
        if 'error_rate' in df.columns:
            df['error_severity'] = pd.cut(
                df['error_rate'],
                bins=[0, 0.01, 0.05, 0.1, 0.2, 1.0],
                labels=['excellent', 'good', 'acceptable', 'poor', 'critical']
            )
        
        return df
    
    def _create_performance_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create performance ratio features"""
        logger.info("Creating performance ratio features")
        
        # Response time ratios
        if all(col in df.columns for col in ['avg_response_time', 'max_response_time', 'min_response_time']):
            df['response_time_consistency'] = df['min_response_time'] / df['max_response_time']
            df['response_time_spread'] = df['max_response_time'] / df['avg_response_time']
        
        # Throughput ratios
        if all(col in df.columns for col in ['requests_per_sec', 'concurrent_users']):
            baseline_rps = df['requests_per_sec'].min()
            df['throughput_ratio'] = df['requests_per_sec'] / baseline_rps if baseline_rps > 0 else 1
        
        # Performance efficiency
        if all(col in df.columns for col in ['avg_response_time', 'requests_per_sec']):
            df['performance_efficiency'] = df['requests_per_sec'] / df['avg_response_time']
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        logger.info("Creating time-based features")
        
        # Convert timestamp to datetime if it's a string
        if 'timestamp' in df.columns and df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if 'timestamp' in df.columns:
            # Time of day
            df['hour_of_day'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Time since epoch
            df['time_since_epoch'] = df['timestamp'].astype(np.int64) // 10**9
            
            # Time-based patterns
            df['is_business_hours'] = ((df['hour_of_day'] >= 9) & (df['hour_of_day'] <= 17)).astype(int)
        
        # Test duration features
        if 'duration' in df.columns:
            df['duration_minutes'] = df['duration'] / 60
            df['duration_category'] = pd.cut(
                df['duration'],
                bins=[0, 300, 600, 1200, float('inf')],
                labels=['short', 'medium', 'long', 'very_long']
            )
        
        return df
    
    def _create_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create network condition features"""
        logger.info("Creating network features")
        
        if 'network_condition' in df.columns:
            # Network speed mapping
            network_speeds = {
                'wifi': 50,      # Mbps
                '4g': 20,        # Mbps
                '3g': 3,         # Mbps
                'slow_3g': 1.5   # Mbps
            }
            
            df['network_speed_mbps'] = df['network_condition'].map(network_speeds)
            df['network_speed_kbps'] = df['network_speed_mbps'] * 1000
            
            # Network condition encoding
            df['network_condition_encoded'] = pd.Categorical(df['network_condition']).codes
            
            # Network impact on performance
            if 'avg_response_time' in df.columns:
                df['network_performance_impact'] = df['avg_response_time'] * df['network_speed_mbps'] / 50
        
        return df
    
    def _create_frontend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create frontend-specific features"""
        logger.info("Creating frontend features")
        
        # Lighthouse scores
        lighthouse_scores = ['performance_score', 'accessibility_score', 'best_practices_score', 'seo_score']
        
        for score in lighthouse_scores:
            if score in df.columns:
                # Handle None values by filling with 0.5 (neutral score)
                score_data = df[score].fillna(0.5)
                df[f'{score}_category'] = pd.cut(
                    score_data,
                    bins=[0, 0.5, 0.7, 0.9, 1.0],
                    labels=['poor', 'needs_improvement', 'good', 'excellent']
                )
        
        # Overall frontend score
        if all(score in df.columns for score in lighthouse_scores):
            # Fill None values with 0.5 before calculating mean and std
            lighthouse_data = df[lighthouse_scores].fillna(0.5)
            df['overall_frontend_score'] = lighthouse_data.mean(axis=1)
            df['frontend_consistency'] = lighthouse_data.std(axis=1)
        
        # Browser-specific features
        if 'browser' in df.columns:
            df['browser_encoded'] = pd.Categorical(df['browser']).codes
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction and user behavior features"""
        logger.info("Creating interaction features")
        
        # User interaction patterns
        if all(col in df.columns for col in ['concurrent_users', 'spawn_rate', 'duration']):
            df['user_session_duration'] = df['duration'] / df['concurrent_users']
            df['user_arrival_rate'] = df['spawn_rate'] / df['concurrent_users']
        
        # Load distribution
        if 'concurrent_users' in df.columns:
            df['load_distribution'] = df['concurrent_users'] / df['concurrent_users'].max()
        
        # Test intensity
        if all(col in df.columns for col in ['concurrent_users', 'duration', 'requests_per_sec']):
            df['test_intensity'] = df['concurrent_users'] * df['requests_per_sec'] / df['duration']
        
        return df
    
    def _create_complexity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create resource complexity features"""
        logger.info("Creating complexity features")
        
        # API complexity
        if 'api_endpoints' in df.columns:
            df['endpoint_count'] = df['api_endpoints'].apply(
                lambda x: len(x) if isinstance(x, list) else 1
            )
        
        # Payload complexity
        if 'payload_size' in df.columns:
            df['payload_size_kb'] = df['payload_size'] / 1024 if df['payload_size'].notna().any() else 0
            df['payload_complexity'] = pd.cut(
                df['payload_size_kb'],
                bins=[0, 1, 10, 100, 1000, float('inf')],
                labels=['tiny', 'small', 'medium', 'large', 'huge']
            )
        
        # Resource complexity score
        complexity_factors = []
        if 'endpoint_count' in df.columns:
            complexity_factors.append(df['endpoint_count'])
        if 'payload_size_kb' in df.columns:
            complexity_factors.append(df['payload_size_kb'] / 100)  # Normalize
        
        if complexity_factors:
            df['resource_complexity'] = np.mean(complexity_factors, axis=0)
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Normalize numerical features"""
        logger.info(f"Normalizing features using {method} scaling")
        
        df_normalized = df.copy()
        
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove columns that shouldn't be normalized
        exclude_cols = [
            'id', 'concurrent_users', 'duration', 'timestamp', 'time_since_epoch',
            'failure_target', 'reliability_target', 'performance_target', 'anomaly_target'
        ]
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit and transform
        df_normalized[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        
        # Store scaler for later use
        self.scalers[method] = scaler
        
        logger.info(f"Normalized {len(numerical_cols)} features")
        return df_normalized
    
    def select_features(self, df: pd.DataFrame, target_col: str, method: str = 'mutual_info', k: int = 20) -> pd.DataFrame:
        """Select the most important features"""
        logger.info(f"Selecting top {k} features using {method}")
        
        # Prepare data for feature selection
        X = df.select_dtypes(include=[np.number])
        y = df[target_col]
        
        # Remove target column from features
        if target_col in X.columns:
            X = X.drop(columns=[target_col])
        
        # Remove columns with too many missing values
        X = X.dropna(axis=1, thresh=len(X) * 0.5)
        
        # Fill remaining missing values
        X = X.fillna(X.mean())
        
        # Select features
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=min(k, X.shape[1]))
        elif method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Create new dataframe with selected features
        df_selected = df[selected_features + [target_col]]
        
        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        return df_selected
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for different ML tasks"""
        logger.info("Creating target variables")
        
        df_with_targets = df.copy()
        
        # Regression targets
        if 'avg_response_time' in df.columns:
            df_with_targets['latency_target'] = df['avg_response_time']
            df_with_targets['latency_log'] = np.log1p(df['avg_response_time'])
        
        if 'requests_per_sec' in df.columns:
            df_with_targets['throughput_target'] = df['requests_per_sec']
        
        # Classification targets
        if 'error_rate' in df.columns:
            df_with_targets['failure_target'] = (df['error_rate'] > 0.05).astype(int)  # 5% threshold
            df_with_targets['reliability_target'] = (df['error_rate'] < 0.01).astype(int)  # 1% threshold
        
        if 'avg_response_time' in df.columns:
            df_with_targets['performance_target'] = (df['avg_response_time'] < 1000).astype(int)  # 1s threshold
        
        # Anomaly detection targets (using statistical outliers)
        if 'avg_response_time' in df.columns:
            Q1 = df['avg_response_time'].quantile(0.25)
            Q3 = df['avg_response_time'].quantile(0.75)
            IQR = Q3 - Q1
            outlier_threshold = Q3 + 1.5 * IQR
            df_with_targets['anomaly_target'] = (df['avg_response_time'] > outlier_threshold).astype(int)
        
        logger.info("Target variables created")
        return df_with_targets
    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get a summary of engineered features"""
        summary = {
            'total_features': len(df.columns),
            'numerical_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object', 'category']).columns),
            'missing_values': df.isnull().sum().to_dict(),
            'feature_types': df.dtypes.to_dict(),
            'engineered_features': []
        }
        
        # Identify engineered features
        base_features = [
            'test_id', 'test_type', 'timestamp', 'total_requests', 'total_failures',
            'error_rate', 'avg_response_time', 'max_response_time', 'min_response_time',
            'requests_per_sec', 'concurrent_users', 'spawn_rate', 'duration',
            'network_condition', 'browser', 'target_url', 'performance_score',
            'accessibility_score', 'best_practices_score', 'seo_score', 'status', 'error'
        ]
        
        summary['engineered_features'] = [col for col in df.columns if col not in base_features]
        
        return summary

def main():
    """Example usage of feature engineering"""
    # Create sample data
    sample_data = {
        'test_id': [f'test_{i}' for i in range(100)],
        'test_type': ['api'] * 50 + ['frontend'] * 50,
        'concurrent_users': np.random.randint(10, 500, 100),
        'spawn_rate': np.random.uniform(1, 20, 100),
        'duration': np.random.randint(300, 1800, 100),
        'avg_response_time': np.random.uniform(100, 5000, 100),
        'max_response_time': np.random.uniform(200, 10000, 100),
        'min_response_time': np.random.uniform(50, 1000, 100),
        'requests_per_sec': np.random.uniform(10, 1000, 100),
        'error_rate': np.random.uniform(0, 0.1, 100),
        'network_condition': np.random.choice(['wifi', '4g', '3g'], 100),
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='H')
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize feature engineer
    engineer = PerformanceFeatureEngineer()
    
    # Engineer features
    df_engineered = engineer.engineer_features(df)
    
    # Create target variables
    df_with_targets = engineer.create_target_variables(df_engineered)
    
    # Normalize features
    df_normalized = engineer.normalize_features(df_with_targets)
    
    # Get feature summary
    summary = engineer.get_feature_summary(df_normalized)
    
    print("Feature Engineering Summary:")
    print(f"Original features: {len(df.columns)}")
    print(f"Engineered features: {len(df_normalized.columns)}")
    print(f"New features created: {len(summary['engineered_features'])}")
    print(f"Engineered features: {summary['engineered_features'][:10]}...")
    
    return df_normalized

if __name__ == "__main__":
    main()
