"""
Continuous Improvement for Performance ML System
Phase 4: Continuous Improvement - Step 1
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import json
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelRetrainer:
    """Handles model retraining and versioning"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.model_versions = {}
        self.retraining_history = []
        self.current_models = {}
        
    def retrain_model(self, model_name: str, new_data: pd.DataFrame, 
                     target_column: str, model_type: str = 'regression',
                     test_size: float = 0.2) -> Dict[str, Any]:
        """Retrain a model with new data"""
        logger.info(f"Retraining model: {model_name}")
        
        # Prepare data
        X = new_data.drop(columns=[target_column, 'test_id', 'timestamp'] if 'test_id' in new_data.columns else [target_column])
        y = new_data[target_column]
        
        # Remove rows with missing values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 50:
            logger.warning(f"Insufficient data for retraining {model_name}: {len(X)} samples")
            return {'success': False, 'error': 'Insufficient data'}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Select model type
        if model_type == 'regression':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            evaluation_metrics = ['mae', 'rmse', 'r2']
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            evaluation_metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        
        if model_type == 'regression':
            metrics = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
        else:
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2' if model_type == 'regression' else 'accuracy')
        
        # Create version info
        version_info = {
            'model_name': model_name,
            'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'training_date': datetime.now().isoformat(),
            'data_samples': len(X),
            'features': list(X.columns),
            'model_type': model_type,
            'metrics': metrics,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': dict(zip(X.columns, model.feature_importances_))
        }
        
        # Save model
        model_path = f"{self.model_dir}/{model_name}_{version_info['version']}.joblib"
        joblib.dump(model, model_path)
        
        # Save version info
        version_path = f"{self.model_dir}/{model_name}_{version_info['version']}_info.json"
        with open(version_path, 'w') as f:
            json.dump(version_info, f, indent=2)
        
        # Update current model
        self.current_models[model_name] = {
            'model': model,
            'version_info': version_info,
            'model_path': model_path
        }
        
        # Store retraining record
        retraining_record = {
            'timestamp': datetime.now(),
            'model_name': model_name,
            'version': version_info['version'],
            'data_samples': len(X),
            'metrics': metrics,
            'cv_score': cv_scores.mean()
        }
        self.retraining_history.append(retraining_record)
        
        logger.info(f"Model {model_name} retrained successfully. Version: {version_info['version']}")
        logger.info(f"Metrics: {metrics}")
        
        return {
            'success': True,
            'version_info': version_info,
            'model': model,
            'metrics': metrics
        }
    
    def load_model(self, model_name: str, version: Optional[str] = None) -> Any:
        """Load a specific model version"""
        if version is None:
            # Load latest version
            import glob
            model_files = glob.glob(f"{self.model_dir}/{model_name}_*.joblib")
            if not model_files:
                raise FileNotFoundError(f"No models found for {model_name}")
            
            # Get latest version
            latest_file = max(model_files, key=lambda x: x.split('_')[-1].replace('.joblib', ''))
            version = latest_file.split('_')[-1].replace('.joblib', '')
        
        model_path = f"{self.model_dir}/{model_name}_{version}.joblib"
        model = joblib.load(model_path)
        
        # Load version info
        info_path = f"{self.model_dir}/{model_name}_{version}_info.json"
        with open(info_path, 'r') as f:
            version_info = json.load(f)
        
        self.current_models[model_name] = {
            'model': model,
            'version_info': version_info,
            'model_path': model_path
        }
        
        return model
    
    def compare_model_versions(self, model_name: str, versions: List[str]) -> pd.DataFrame:
        """Compare different model versions"""
        comparison_data = []
        
        for version in versions:
            try:
                info_path = f"{self.model_dir}/{model_name}_{version}_info.json"
                with open(info_path, 'r') as f:
                    version_info = json.load(f)
                
                comparison_data.append({
                    'version': version,
                    'training_date': version_info['training_date'],
                    'data_samples': version_info['data_samples'],
                    'cv_mean': version_info['cv_mean'],
                    'cv_std': version_info['cv_std'],
                    **version_info['metrics']
                })
            except FileNotFoundError:
                logger.warning(f"Version info not found for {model_name}_{version}")
        
        return pd.DataFrame(comparison_data)
    
    def get_retraining_summary(self) -> Dict[str, Any]:
        """Get summary of retraining history"""
        if not self.retraining_history:
            return {'total_retrainings': 0}
        
        df_history = pd.DataFrame(self.retraining_history)
        
        summary = {
            'total_retrainings': len(df_history),
            'models_retrained': df_history['model_name'].nunique(),
            'recent_retrainings': len(df_history[df_history['timestamp'] > datetime.now() - timedelta(days=7)]),
            'retrainings_by_model': df_history['model_name'].value_counts().to_dict(),
            'average_cv_score': df_history['cv_score'].mean() if 'cv_score' in df_history.columns else None
        }
        
        return summary

class DriftDetector:
    """Detects model drift and data drift"""
    
    def __init__(self):
        self.drift_history = []
        self.baseline_stats = {}
        
    def calculate_data_statistics(self, df: pd.DataFrame, 
                                columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculate statistical summary of data"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        stats = {}
        for column in columns:
            if column in df.columns:
                col_data = df[column].dropna()
                if len(col_data) > 0:
                    stats[column] = {
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'median': float(col_data.median()),
                        'q25': float(col_data.quantile(0.25)),
                        'q75': float(col_data.quantile(0.75)),
                        'count': len(col_data)
                    }
        
        return stats
    
    def set_baseline(self, df: pd.DataFrame, baseline_name: str = 'initial'):
        """Set baseline statistics for drift detection"""
        logger.info(f"Setting baseline statistics: {baseline_name}")
        
        baseline_stats = self.calculate_data_statistics(df)
        self.baseline_stats[baseline_name] = {
            'timestamp': datetime.now(),
            'statistics': baseline_stats,
            'data_shape': df.shape
        }
        
        logger.info(f"Baseline set with {len(baseline_stats)} features")
    
    def detect_data_drift(self, current_data: pd.DataFrame, 
                         baseline_name: str = 'initial',
                         drift_threshold: float = 0.3) -> Dict[str, Any]:
        """Detect data drift by comparing current data with baseline"""
        logger.info(f"Detecting data drift against baseline: {baseline_name}")
        
        if baseline_name not in self.baseline_stats:
            raise ValueError(f"Baseline '{baseline_name}' not found")
        
        baseline = self.baseline_stats[baseline_name]
        current_stats = self.calculate_data_statistics(current_data)
        
        drift_results = {}
        drifted_features = []
        
        for column in current_stats.keys():
            if column in baseline['statistics']:
                baseline_stat = baseline['statistics'][column]
                current_stat = current_stats[column]
                
                # Calculate drift metrics
                mean_drift = abs(current_stat['mean'] - baseline_stat['mean']) / (abs(baseline_stat['mean']) + 1e-8)
                std_drift = abs(current_stat['std'] - baseline_stat['std']) / (abs(baseline_stat['std']) + 1e-8)
                
                # Distribution drift (using quantiles)
                q25_drift = abs(current_stat['q25'] - baseline_stat['q25']) / (abs(baseline_stat['q25']) + 1e-8)
                q75_drift = abs(current_stat['q75'] - baseline_stat['q75']) / (abs(baseline_stat['q75']) + 1e-8)
                
                # Overall drift score
                drift_score = np.mean([mean_drift, std_drift, q25_drift, q75_drift])
                
                drift_results[column] = {
                    'baseline': baseline_stat,
                    'current': current_stat,
                    'mean_drift': mean_drift,
                    'std_drift': std_drift,
                    'q25_drift': q25_drift,
                    'q75_drift': q75_drift,
                    'drift_score': drift_score,
                    'is_drifted': drift_score > drift_threshold
                }
                
                if drift_score > drift_threshold:
                    drifted_features.append(column)
        
        # Overall assessment
        total_features = len(drift_results)
        drifted_count = len(drifted_features)
        avg_drift = np.mean([result['drift_score'] for result in drift_results.values()])
        
        drift_assessment = {
            'timestamp': datetime.now(),
            'baseline_name': baseline_name,
            'total_features': total_features,
            'drifted_features': drifted_features,
            'drifted_count': drifted_count,
            'drift_rate': drifted_count / total_features if total_features > 0 else 0,
            'average_drift': avg_drift,
            'drift_threshold': drift_threshold,
            'significant_drift': drifted_count > total_features * 0.3 or avg_drift > drift_threshold,
            'detailed_results': drift_results
        }
        
        # Store drift record
        self.drift_history.append(drift_assessment)
        
        logger.info(f"Data drift detection completed. Drifted features: {drifted_count}/{total_features}")
        logger.info(f"Average drift: {avg_drift:.4f}")
        
        return drift_assessment
    
    def detect_model_drift(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                          baseline_performance: Dict[str, float]) -> Dict[str, Any]:
        """Detect model performance drift"""
        logger.info("Detecting model performance drift")
        
        # Get current predictions
        y_pred = model.predict(X_test)
        
        # Calculate current metrics
        if hasattr(model, 'predict_proba'):
            # Classification model
            current_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
        else:
            # Regression model
            current_metrics = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
        
        # Calculate drift
        drift_metrics = {}
        for metric in current_metrics.keys():
            if metric in baseline_performance:
                baseline_value = baseline_performance[metric]
                current_value = current_metrics[metric]
                
                # Calculate relative change
                if baseline_value != 0:
                    drift = abs(current_value - baseline_value) / abs(baseline_value)
                else:
                    drift = abs(current_value - baseline_value)
                
                drift_metrics[metric] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'drift': drift,
                    'improved': current_value > baseline_value if metric in ['accuracy', 'precision', 'recall', 'f1', 'r2'] else current_value < baseline_value
                }
        
        # Overall assessment
        avg_drift = np.mean([metric['drift'] for metric in drift_metrics.values()])
        improved_metrics = sum(1 for metric in drift_metrics.values() if metric['improved'])
        
        model_drift_assessment = {
            'timestamp': datetime.now(),
            'baseline_performance': baseline_performance,
            'current_performance': current_metrics,
            'drift_metrics': drift_metrics,
            'average_drift': avg_drift,
            'improved_metrics': improved_metrics,
            'total_metrics': len(drift_metrics),
            'significant_drift': avg_drift > 0.2,  # 20% threshold
            'model_degraded': improved_metrics < len(drift_metrics) / 2
        }
        
        logger.info(f"Model drift assessment: avg_drift={avg_drift:.4f}, improved_metrics={improved_metrics}")
        
        return model_drift_assessment
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift detection history"""
        if not self.drift_history:
            return {'total_drift_checks': 0}
        
        df_drift = pd.DataFrame(self.drift_history)
        
        summary = {
            'total_drift_checks': len(df_drift),
            'significant_drift_events': df_drift['significant_drift'].sum(),
            'recent_drift_checks': len(df_drift[df_drift['timestamp'] > datetime.now() - timedelta(days=7)]),
            'average_drift_rate': df_drift['drift_rate'].mean(),
            'baselines_used': df_drift['baseline_name'].unique().tolist()
        }
        
        return summary

class ContinuousImprovementOrchestrator:
    """Orchestrates continuous improvement processes"""
    
    def __init__(self, model_retrainer: ModelRetrainer, drift_detector: DriftDetector):
        self.model_retrainer = model_retrainer
        self.drift_detector = drift_detector
        self.improvement_history = []
        
    def run_continuous_improvement_cycle(self, new_data: pd.DataFrame,
                                       models_to_monitor: List[str],
                                       target_columns: List[str]) -> Dict[str, Any]:
        """Run a complete continuous improvement cycle"""
        logger.info("Starting continuous improvement cycle")
        
        cycle_results = {
            'timestamp': datetime.now(),
            'data_samples': len(new_data),
            'models_monitored': models_to_monitor,
            'drift_detection': {},
            'retraining_decisions': {},
            'improvements': {}
        }
        
        # Step 1: Detect data drift
        if not self.drift_detector.baseline_stats:
            logger.info("Setting initial baseline")
            self.drift_detector.set_baseline(new_data, 'initial')
        
        drift_assessment = self.drift_detector.detect_data_drift(new_data, 'initial')
        cycle_results['drift_detection'] = drift_assessment
        
        # Step 2: Check each model
        for model_name in models_to_monitor:
            logger.info(f"Checking model: {model_name}")
            
            # Determine target column for this model
            target_column = None
            for col in target_columns:
                if col in new_data.columns:
                    target_column = col
                    break
            
            if target_column is None:
                logger.warning(f"No target column found for model {model_name}")
                continue
            
            # Check if model needs retraining
            retraining_decision = self._assess_retraining_needs(
                model_name, new_data, target_column, drift_assessment
            )
            
            cycle_results['retraining_decisions'][model_name] = retraining_decision
            
            # Retrain if needed
            if retraining_decision['should_retrain']:
                logger.info(f"Retraining model: {model_name}")
                
                retraining_result = self.model_retrainer.retrain_model(
                    model_name, new_data, target_column
                )
                
                if retraining_result['success']:
                    cycle_results['improvements'][model_name] = {
                        'action': 'retrained',
                        'new_version': retraining_result['version_info']['version'],
                        'metrics': retraining_result['metrics']
                    }
                else:
                    cycle_results['improvements'][model_name] = {
                        'action': 'retraining_failed',
                        'error': retraining_result.get('error', 'Unknown error')
                    }
            else:
                cycle_results['improvements'][model_name] = {
                    'action': 'no_action_needed',
                    'reason': retraining_decision['reason']
                }
        
        # Store cycle results
        self.improvement_history.append(cycle_results)
        
        logger.info("Continuous improvement cycle completed")
        
        return cycle_results
    
    def _assess_retraining_needs(self, model_name: str, new_data: pd.DataFrame,
                               target_column: str, drift_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Assess whether a model needs retraining"""
        
        # Check data drift
        significant_drift = drift_assessment['significant_drift']
        drift_rate = drift_assessment['drift_rate']
        
        # Check data volume
        sufficient_data = len(new_data) >= 100
        
        # Check time since last retraining
        last_retraining = None
        for record in self.model_retrainer.retraining_history:
            if record['model_name'] == model_name:
                last_retraining = record['timestamp']
                break
        
        time_since_retraining = None
        if last_retraining:
            time_since_retraining = datetime.now() - last_retraining
        
        # Decision logic
        should_retrain = False
        reasons = []
        
        if significant_drift:
            should_retrain = True
            reasons.append("Significant data drift detected")
        
        if drift_rate > 0.5:
            should_retrain = True
            reasons.append("High drift rate (>50%)")
        
        if not sufficient_data:
            reasons.append("Insufficient new data")
        
        if time_since_retraining and time_since_retraining > timedelta(days=30):
            should_retrain = True
            reasons.append("More than 30 days since last retraining")
        
        if not reasons:
            reasons.append("No retraining criteria met")
        
        return {
            'should_retrain': should_retrain,
            'reasons': reasons,
            'significant_drift': significant_drift,
            'drift_rate': drift_rate,
            'sufficient_data': sufficient_data,
            'time_since_retraining': str(time_since_retraining) if time_since_retraining else None
        }
    
    def get_improvement_summary(self) -> Dict[str, Any]:
        """Get summary of continuous improvement activities"""
        if not self.improvement_history:
            return {'total_cycles': 0}
        
        df_improvements = pd.DataFrame(self.improvement_history)
        
        # Count actions
        total_retrainings = 0
        total_cycles = len(df_improvements)
        
        for cycle in self.improvement_history:
            for model, improvement in cycle['improvements'].items():
                if improvement['action'] == 'retrained':
                    total_retrainings += 1
        
        summary = {
            'total_cycles': total_cycles,
            'total_retrainings': total_retrainings,
            'recent_cycles': len(df_improvements[df_improvements['timestamp'] > datetime.now() - timedelta(days=7)]),
            'average_data_samples_per_cycle': df_improvements['data_samples'].mean(),
            'models_improved': len(set([
                model for cycle in self.improvement_history 
                for model, improvement in cycle['improvements'].items()
                if improvement['action'] == 'retrained'
            ]))
        }
        
        return summary

def main():
    """Example usage of continuous improvement"""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Initial data
    initial_data = {
        'concurrent_users': np.random.randint(10, 500, n_samples),
        'spawn_rate': np.random.uniform(1, 20, n_samples),
        'duration': np.random.randint(300, 1800, n_samples),
        'avg_response_time': np.random.uniform(100, 2000, n_samples),
        'error_rate': np.random.uniform(0, 0.1, n_samples)
    }
    
    df_initial = pd.DataFrame(initial_data)
    
    # New data with some drift
    new_data = {
        'concurrent_users': np.random.randint(20, 600, n_samples),  # Slightly higher
        'spawn_rate': np.random.uniform(1.5, 25, n_samples),  # Higher spawn rates
        'duration': np.random.randint(300, 1800, n_samples),
        'avg_response_time': np.random.uniform(150, 2500, n_samples),  # Higher latency
        'error_rate': np.random.uniform(0, 0.15, n_samples)  # Higher error rates
    }
    
    df_new = pd.DataFrame(new_data)
    
    # Initialize components
    model_retrainer = ModelRetrainer()
    drift_detector = DriftDetector()
    orchestrator = ContinuousImprovementOrchestrator(model_retrainer, drift_detector)
    
    # Set baseline
    drift_detector.set_baseline(df_initial, 'initial')
    
    # Detect drift
    drift_assessment = drift_detector.detect_data_drift(df_new, 'initial')
    
    print("Data Drift Assessment:")
    print(f"Significant drift: {drift_assessment['significant_drift']}")
    print(f"Drift rate: {drift_assessment['drift_rate']:.2%}")
    print(f"Drifted features: {drift_assessment['drifted_features']}")
    
    # Run continuous improvement cycle
    cycle_results = orchestrator.run_continuous_improvement_cycle(
        df_new, 
        models_to_monitor=['latency_model', 'error_model'],
        target_columns=['avg_response_time', 'error_rate']
    )
    
    print("\nContinuous Improvement Cycle Results:")
    print(f"Models monitored: {cycle_results['models_monitored']}")
    print(f"Retraining decisions: {cycle_results['retraining_decisions']}")
    print(f"Improvements: {cycle_results['improvements']}")
    
    # Get summaries
    retraining_summary = model_retrainer.get_retraining_summary()
    drift_summary = drift_detector.get_drift_summary()
    improvement_summary = orchestrator.get_improvement_summary()
    
    print(f"\nRetraining Summary: {retraining_summary}")
    print(f"Drift Summary: {drift_summary}")
    print(f"Improvement Summary: {improvement_summary}")

if __name__ == "__main__":
    main()
