"""
Anomaly Detection and Feedback Loop for Performance ML System
Phase 3: Optimization & Automation - Step 2
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PerformanceAnomalyDetector:
    """Detects anomalies in performance metrics"""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.models = {}
        self.scalers = {}
        self.thresholds = {}
        self.anomaly_history = []
        
    def fit_models(self, df: pd.DataFrame, target_columns: List[str]):
        """Fit multiple anomaly detection models"""
        logger.info(f"Fitting anomaly detection models for columns: {target_columns}")
        
        for column in target_columns:
            if column not in df.columns:
                logger.warning(f"Column {column} not found in dataframe")
                continue
            
            # Prepare data
            X = df[column].values.reshape(-1, 1)
            X = X[~np.isnan(X)]  # Remove NaN values
            
            if len(X) < 10:
                logger.warning(f"Insufficient data for column {column}")
                continue
            
            # Scale data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit multiple models
            models = {
                'isolation_forest': IsolationForest(contamination=self.contamination, random_state=42),
                'local_outlier_factor': LocalOutlierFactor(contamination=self.contamination, novelty=True),
                'elliptic_envelope': EllipticEnvelope(contamination=self.contamination, random_state=42)
            }
            
            fitted_models = {}
            for name, model in models.items():
                try:
                    model.fit(X_scaled)
                    fitted_models[name] = model
                except Exception as e:
                    logger.error(f"Error fitting {name} for column {column}: {e}")
            
            self.models[column] = fitted_models
            self.scalers[column] = scaler
            
            # Calculate statistical thresholds
            self.thresholds[column] = self._calculate_statistical_thresholds(X)
            
        logger.info(f"Fitted models for {len(self.models)} columns")
    
    def _calculate_statistical_thresholds(self, X: np.ndarray) -> Dict[str, float]:
        """Calculate statistical thresholds for anomaly detection"""
        mean = np.mean(X)
        std = np.std(X)
        
        return {
            'mean': mean,
            'std': std,
            'z_score_2': mean + 2 * std,  # 2 standard deviations
            'z_score_3': mean + 3 * std,  # 3 standard deviations
            'iqr_upper': np.percentile(X, 75) + 1.5 * (np.percentile(X, 75) - np.percentile(X, 25)),
            'iqr_lower': np.percentile(X, 25) - 1.5 * (np.percentile(X, 75) - np.percentile(X, 25))
        }
    
    def detect_anomalies(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Detect anomalies in performance data"""
        if columns is None:
            columns = list(self.models.keys())
        
        logger.info(f"Detecting anomalies for columns: {columns}")
        
        results = {}
        
        for column in columns:
            if column not in self.models:
                logger.warning(f"No models fitted for column {column}")
                continue
            
            # Prepare data
            X = df[column].values.reshape(-1, 1)
            X_clean = X[~np.isnan(X)]
            indices = np.where(~np.isnan(X))[0]
            
            if len(X_clean) == 0:
                logger.warning(f"No valid data for column {column}")
                continue
            
            # Scale data
            X_scaled = self.scalers[column].transform(X_clean)
            
            # Get predictions from all models
            predictions = {}
            for model_name, model in self.models[column].items():
                try:
                    if hasattr(model, 'predict'):
                        pred = model.predict(X_scaled)
                        # Convert to binary (1 for normal, -1 for anomaly)
                        predictions[model_name] = (pred == 1).astype(int)
                    elif hasattr(model, 'score_samples'):
                        scores = model.score_samples(X_scaled)
                        # Use threshold-based detection
                        threshold = np.percentile(scores, self.contamination * 100)
                        predictions[model_name] = (scores >= threshold).astype(int)
                except Exception as e:
                    logger.error(f"Error predicting with {model_name} for column {column}: {e}")
            
            # Combine predictions (ensemble approach)
            if predictions:
                ensemble_pred = np.mean(list(predictions.values()), axis=0)
                anomaly_mask = ensemble_pred < 0.5  # Majority vote
                
                # Statistical detection
                thresholds = self.thresholds[column]
                statistical_anomalies = (
                    (X_clean > thresholds['z_score_3'].flatten()) |
                    (X_clean < thresholds['iqr_lower'].flatten())
                ).flatten()
                
                # Combine ML and statistical detection
                final_anomalies = anomaly_mask | statistical_anomalies
                
                # Get anomaly details
                anomaly_indices = indices[final_anomalies]
                anomaly_values = X_clean[final_anomalies]
                
                results[column] = {
                    'anomaly_indices': anomaly_indices,
                    'anomaly_values': anomaly_values,
                    'anomaly_count': len(anomaly_indices),
                    'anomaly_rate': len(anomaly_indices) / len(X_clean),
                    'predictions': predictions,
                    'statistical_anomalies': statistical_anomalies,
                    'ensemble_predictions': ensemble_pred
                }
                
                # Log anomalies
                for i, (idx, value) in enumerate(zip(anomaly_indices, anomaly_values)):
                    anomaly_record = {
                        'timestamp': datetime.now(),
                        'column': column,
                        'index': int(idx),
                        'value': float(value),
                        'threshold_exceeded': float(value) > thresholds['z_score_3'],
                        'model_consensus': float(ensemble_pred[final_anomalies][i])
                    }
                    self.anomaly_history.append(anomaly_record)
        
        return results
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of detected anomalies"""
        if not self.anomaly_history:
            return {'total_anomalies': 0, 'anomalies_by_column': {}}
        
        df_anomalies = pd.DataFrame(self.anomaly_history)
        
        summary = {
            'total_anomalies': len(df_anomalies),
            'anomalies_by_column': df_anomalies['column'].value_counts().to_dict(),
            'recent_anomalies': len(df_anomalies[df_anomalies['timestamp'] > datetime.now() - timedelta(hours=24)]),
            'threshold_violations': df_anomalies['threshold_exceeded'].sum(),
            'average_model_consensus': df_anomalies['model_consensus'].mean()
        }
        
        return summary

class AnomalyFeedbackLoop:
    """Implements feedback loop for anomaly detection and model updates"""
    
    def __init__(self, performance_model, anomaly_detector: PerformanceAnomalyDetector):
        self.performance_model = performance_model
        self.anomaly_detector = anomaly_detector
        self.feedback_history = []
        self.model_drift_detected = False
        self.last_model_update = None
        
    def compare_predictions_vs_reality(self, predictions: Dict[str, float], 
                                     actual_results: Dict[str, float],
                                     threshold: float = 0.2) -> Dict[str, Any]:
        """Compare ML predictions with actual test results"""
        logger.info("Comparing predictions vs actual results")
        
        comparison = {}
        deviations = {}
        
        for metric in predictions.keys():
            if metric in actual_results:
                predicted = predictions[metric]
                actual = actual_results[metric]
                
                # Calculate deviation
                if actual != 0:
                    deviation = abs(predicted - actual) / actual
                else:
                    deviation = abs(predicted - actual)
                
                deviations[metric] = deviation
                
                comparison[metric] = {
                    'predicted': predicted,
                    'actual': actual,
                    'deviation': deviation,
                    'is_anomaly': deviation > threshold
                }
        
        # Overall assessment
        avg_deviation = np.mean(list(deviations.values()))
        anomaly_count = sum(1 for comp in comparison.values() if comp['is_anomaly'])
        
        overall_assessment = {
            'average_deviation': avg_deviation,
            'anomaly_count': anomaly_count,
            'is_significant_drift': avg_deviation > threshold or anomaly_count > 0,
            'comparison': comparison
        }
        
        # Store feedback
        feedback_record = {
            'timestamp': datetime.now(),
            'predictions': predictions,
            'actual_results': actual_results,
            'comparison': comparison,
            'overall_assessment': overall_assessment
        }
        self.feedback_history.append(feedback_record)
        
        logger.info(f"Prediction vs reality comparison: avg_deviation={avg_deviation:.4f}, anomalies={anomaly_count}")
        
        return overall_assessment
    
    def detect_model_drift(self, recent_data: pd.DataFrame, 
                          baseline_data: pd.DataFrame,
                          drift_threshold: float = 0.3) -> Dict[str, Any]:
        """Detect model drift by comparing recent data with baseline"""
        logger.info("Detecting model drift")
        
        # Compare distributions of key features
        drift_metrics = {}
        
        for column in recent_data.select_dtypes(include=[np.number]).columns:
            if column in baseline_data.columns:
                recent_mean = recent_data[column].mean()
                baseline_mean = baseline_data[column].mean()
                
                if baseline_mean != 0:
                    drift = abs(recent_mean - baseline_mean) / abs(baseline_mean)
                else:
                    drift = abs(recent_mean - baseline_mean)
                
                drift_metrics[column] = {
                    'recent_mean': recent_mean,
                    'baseline_mean': baseline_mean,
                    'drift': drift,
                    'is_drifted': drift > drift_threshold
                }
        
        # Overall drift assessment
        drifted_features = sum(1 for metric in drift_metrics.values() if metric['is_drifted'])
        avg_drift = np.mean([metric['drift'] for metric in drift_metrics.values()])
        
        drift_assessment = {
            'average_drift': avg_drift,
            'drifted_features': drifted_features,
            'total_features': len(drift_metrics),
            'drift_metrics': drift_metrics,
            'model_needs_retraining': drifted_features > len(drift_metrics) * 0.3 or avg_drift > drift_threshold
        }
        
        self.model_drift_detected = drift_assessment['model_needs_retraining']
        
        logger.info(f"Model drift assessment: avg_drift={avg_drift:.4f}, drifted_features={drifted_features}")
        
        return drift_assessment
    
    def trigger_model_retraining(self, new_data: pd.DataFrame, 
                               retraining_threshold: int = 100) -> bool:
        """Trigger model retraining based on various conditions"""
        logger.info("Checking if model retraining is needed")
        
        # Check conditions for retraining
        conditions = {
            'model_drift_detected': self.model_drift_detected,
            'sufficient_new_data': len(new_data) >= retraining_threshold,
            'significant_time_passed': (
                self.last_model_update is None or 
                datetime.now() - self.last_model_update > timedelta(days=7)
            ),
            'high_anomaly_rate': (
                len(self.anomaly_detector.anomaly_history) > 0 and
                len([a for a in self.anomaly_detector.anomaly_history 
                     if a['timestamp'] > datetime.now() - timedelta(hours=24)]) > 10
            )
        }
        
        should_retrain = any(conditions.values())
        
        if should_retrain:
            logger.info("Model retraining triggered")
            logger.info(f"Retraining conditions: {conditions}")
            self.last_model_update = datetime.now()
        
        return should_retrain
    
    def generate_anomaly_alerts(self, anomaly_results: Dict[str, Any], 
                              alert_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Generate alerts for detected anomalies"""
        alerts = []
        
        for column, result in anomaly_results.items():
            if result['anomaly_rate'] > alert_threshold:
                alert = {
                    'timestamp': datetime.now(),
                    'severity': 'high' if result['anomaly_rate'] > 0.2 else 'medium',
                    'column': column,
                    'anomaly_rate': result['anomaly_rate'],
                    'anomaly_count': result['anomaly_count'],
                    'message': f"High anomaly rate detected in {column}: {result['anomaly_rate']:.2%}",
                    'recommendations': [
                        "Review recent system changes",
                        "Check for infrastructure issues",
                        "Consider model retraining",
                        "Monitor system resources"
                    ]
                }
                alerts.append(alert)
        
        return alerts

class AutomatedTestCalibration:
    """Automatically calibrates test parameters based on ML insights"""
    
    def __init__(self, performance_model, anomaly_detector: PerformanceAnomalyDetector):
        self.performance_model = performance_model
        self.anomaly_detector = anomaly_detector
        self.calibration_history = []
        self.risk_assessments = {}
        
    def assess_test_risk(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk level of a test configuration"""
        logger.info("Assessing test configuration risk")
        
        # Extract features for risk assessment
        features = np.array([
            test_config.get('concurrent_users', 0),
            test_config.get('spawn_rate', 0),
            test_config.get('duration', 0),
            test_config.get('network_condition_encoded', 0)
        ]).reshape(1, -1)
        
        # Predict performance
        predicted_latency = self.performance_model.predict(features)[0]
        
        # Calculate risk factors
        risk_factors = {
            'high_concurrent_users': test_config.get('concurrent_users', 0) > 500,
            'high_spawn_rate': test_config.get('spawn_rate', 0) > 20,
            'long_duration': test_config.get('duration', 0) > 1800,
            'slow_network': test_config.get('network_condition_encoded', 0) > 1,
            'high_predicted_latency': predicted_latency > 2000
        }
        
        # Calculate risk score
        risk_score = sum(risk_factors.values()) / len(risk_factors)
        
        # Risk categories
        if risk_score >= 0.8:
            risk_level = 'critical'
        elif risk_score >= 0.6:
            risk_level = 'high'
        elif risk_score >= 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        risk_assessment = {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'predicted_latency': predicted_latency,
            'recommendations': self._generate_risk_recommendations(risk_factors)
        }
        
        self.risk_assessments[test_config.get('test_id', 'unknown')] = risk_assessment
        
        return risk_assessment
    
    def _generate_risk_recommendations(self, risk_factors: Dict[str, bool]) -> List[str]:
        """Generate recommendations based on risk factors"""
        recommendations = []
        
        if risk_factors['high_concurrent_users']:
            recommendations.append("Consider reducing concurrent users or running in stages")
        
        if risk_factors['high_spawn_rate']:
            recommendations.append("Reduce spawn rate to avoid overwhelming the system")
        
        if risk_factors['long_duration']:
            recommendations.append("Consider shorter test duration or monitoring intervals")
        
        if risk_factors['slow_network']:
            recommendations.append("Test on faster network or adjust expectations")
        
        if risk_factors['high_predicted_latency']:
            recommendations.append("System may not handle this load - reduce parameters")
        
        if not recommendations:
            recommendations.append("Configuration appears safe for testing")
        
        return recommendations
    
    def suggest_test_adjustments(self, test_config: Dict[str, Any], 
                               target_risk_level: str = 'medium') -> Dict[str, Any]:
        """Suggest adjustments to test configuration"""
        logger.info(f"Suggesting test adjustments for target risk level: {target_risk_level}")
        
        current_risk = self.assess_test_risk(test_config)
        
        if current_risk['risk_level'] == target_risk_level:
            return {
                'adjustments_needed': False,
                'current_config': test_config,
                'risk_assessment': current_risk
            }
        
        # Suggest adjustments
        adjusted_config = test_config.copy()
        adjustments = []
        
        if current_risk['risk_level'] in ['critical', 'high']:
            # Reduce risk
            if current_risk['risk_factors']['high_concurrent_users']:
                adjusted_config['concurrent_users'] = int(test_config.get('concurrent_users', 0) * 0.7)
                adjustments.append(f"Reduced concurrent users to {adjusted_config['concurrent_users']}")
            
            if current_risk['risk_factors']['high_spawn_rate']:
                adjusted_config['spawn_rate'] = test_config.get('spawn_rate', 0) * 0.7
                adjustments.append(f"Reduced spawn rate to {adjusted_config['spawn_rate']:.2f}")
            
            if current_risk['risk_factors']['long_duration']:
                adjusted_config['duration'] = int(test_config.get('duration', 0) * 0.8)
                adjustments.append(f"Reduced duration to {adjusted_config['duration']} seconds")
        
        elif current_risk['risk_level'] == 'low':
            # Increase load for more meaningful test
            adjusted_config['concurrent_users'] = int(test_config.get('concurrent_users', 0) * 1.3)
            adjusted_config['spawn_rate'] = test_config.get('spawn_rate', 0) * 1.2
            adjustments.append("Increased load for more comprehensive testing")
        
        # Reassess risk with adjusted config
        adjusted_risk = self.assess_test_risk(adjusted_config)
        
        return {
            'adjustments_needed': True,
            'original_config': test_config,
            'adjusted_config': adjusted_config,
            'adjustments': adjustments,
            'original_risk': current_risk,
            'adjusted_risk': adjusted_risk
        }
    
    def create_adaptive_test_plan(self, base_config: Dict[str, Any], 
                                target_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Create an adaptive test plan that adjusts based on results"""
        logger.info("Creating adaptive test plan")
        
        test_plan = {
            'phases': [],
            'adaptation_rules': [],
            'success_criteria': target_metrics
        }
        
        # Phase 1: Baseline test
        baseline_config = base_config.copy()
        baseline_config['concurrent_users'] = min(base_config.get('concurrent_users', 100), 50)
        baseline_config['spawn_rate'] = min(base_config.get('spawn_rate', 5), 2)
        
        test_plan['phases'].append({
            'phase': 1,
            'name': 'Baseline',
            'config': baseline_config,
            'duration': 300,
            'success_threshold': 0.8
        })
        
        # Phase 2: Progressive load
        progressive_config = base_config.copy()
        progressive_config['concurrent_users'] = int(base_config.get('concurrent_users', 100) * 0.7)
        progressive_config['spawn_rate'] = base_config.get('spawn_rate', 5) * 0.8
        
        test_plan['phases'].append({
            'phase': 2,
            'name': 'Progressive Load',
            'config': progressive_config,
            'duration': 600,
            'success_threshold': 0.7
        })
        
        # Phase 3: Target load
        test_plan['phases'].append({
            'phase': 3,
            'name': 'Target Load',
            'config': base_config,
            'duration': 900,
            'success_threshold': 0.6
        })
        
        # Adaptation rules
        test_plan['adaptation_rules'] = [
            {
                'condition': 'latency > 2000ms',
                'action': 'reduce_load',
                'adjustment': {'concurrent_users': 0.8, 'spawn_rate': 0.8}
            },
            {
                'condition': 'error_rate > 0.05',
                'action': 'reduce_load',
                'adjustment': {'concurrent_users': 0.7, 'spawn_rate': 0.7}
            },
            {
                'condition': 'latency < 500ms and error_rate < 0.01',
                'action': 'increase_load',
                'adjustment': {'concurrent_users': 1.2, 'spawn_rate': 1.1}
            }
        ]
        
        return test_plan

def main():
    """Example usage of anomaly detection and feedback loop"""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate normal performance data
    normal_data = {
        'avg_response_time': np.random.normal(500, 100, n_samples),
        'requests_per_sec': np.random.normal(100, 20, n_samples),
        'error_rate': np.random.normal(0.01, 0.005, n_samples)
    }
    
    # Add some anomalies
    anomaly_indices = np.random.choice(n_samples, size=50, replace=False)
    normal_data['avg_response_time'][anomaly_indices] = np.random.normal(2000, 300, len(anomaly_indices))
    normal_data['error_rate'][anomaly_indices] = np.random.normal(0.1, 0.02, len(anomaly_indices))
    
    df = pd.DataFrame(normal_data)
    
    # Initialize components
    anomaly_detector = PerformanceAnomalyDetector(contamination=0.05)
    
    # Fit models
    anomaly_detector.fit_models(df, ['avg_response_time', 'requests_per_sec', 'error_rate'])
    
    # Detect anomalies
    anomaly_results = anomaly_detector.detect_anomalies(df)
    
    # Get summary
    summary = anomaly_detector.get_anomaly_summary()
    
    print("Anomaly Detection Results:")
    print(f"Total anomalies detected: {summary['total_anomalies']}")
    print(f"Anomalies by column: {summary['anomalies_by_column']}")
    
    # Test feedback loop
    from sklearn.ensemble import RandomForestRegressor
    
    # Mock performance model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    X_sample = np.random.rand(100, 4)
    y_sample = np.random.normal(500, 100, 100)
    model.fit(X_sample, y_sample)
    
    feedback_loop = AnomalyFeedbackLoop(model, anomaly_detector)
    
    # Compare predictions vs reality
    predictions = {'avg_response_time': 450, 'error_rate': 0.015}
    actual_results = {'avg_response_time': 600, 'error_rate': 0.025}
    
    comparison = feedback_loop.compare_predictions_vs_reality(predictions, actual_results)
    
    print(f"\nPrediction vs Reality Comparison:")
    print(f"Average deviation: {comparison['average_deviation']:.4f}")
    print(f"Anomalies detected: {comparison['anomaly_count']}")
    
    # Test calibration
    test_config = {
        'test_id': 'test_001',
        'concurrent_users': 800,
        'spawn_rate': 25,
        'duration': 2400,
        'network_condition_encoded': 2
    }
    
    calibration = AutomatedTestCalibration(model, anomaly_detector)
    risk_assessment = calibration.assess_test_risk(test_config)
    
    print(f"\nRisk Assessment:")
    print(f"Risk level: {risk_assessment['risk_level']}")
    print(f"Risk score: {risk_assessment['risk_score']:.2f}")
    print(f"Recommendations: {risk_assessment['recommendations']}")

if __name__ == "__main__":
    main()
