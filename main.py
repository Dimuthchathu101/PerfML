#!/usr/bin/env python3
"""
Performance ML System - Main Orchestration Script
Complete 4-Phase Performance Testing and ML Pipeline
"""

import os
import sys
import logging
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import all components
from data_collection.parameterized_tests import TestParameterizer
from data_collection.data_pipeline import AutomatedTestRunner, DataStorage
from data_collection.feature_engineering import PerformanceFeatureEngineer
from models.model_development import ProblemFramer, ModelSelector, ModelTrainer, HyperparameterOptimizer, ModelEvaluator
from optimization.bayesian_optimization import BayesianOptimizer, LoadOptimizer, PerformancePredictor
from optimization.anomaly_detection import PerformanceAnomalyDetector, AnomalyFeedbackLoop, AutomatedTestCalibration
from continuous_improvement.model_retraining import ModelRetrainer, DriftDetector, ContinuousImprovementOrchestrator
from continuous_improvement.insight_generation import PerformanceInsightGenerator, PerformanceReporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('perfml.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PerformanceMLSystem:
    """Complete Performance ML System Orchestrator"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.setup_components()
        
    def setup_components(self):
        """Initialize all system components"""
        logger.info("Setting up Performance ML System components")
        
        # Phase 1: Data Collection & Preprocessing
        self.test_parameterizer = TestParameterizer(self.base_url)
        self.test_runner = AutomatedTestRunner(self.base_url)
        self.data_storage = DataStorage()
        self.feature_engineer = PerformanceFeatureEngineer()
        
        # Phase 2: Model Development
        self.problem_framer = ProblemFramer()
        self.model_selector = ModelSelector()
        self.model_trainer = ModelTrainer()
        self.hyperparameter_optimizer = HyperparameterOptimizer(n_trials=50)
        self.model_evaluator = ModelEvaluator()
        
        # Phase 3: Optimization & Automation
        self.anomaly_detector = PerformanceAnomalyDetector()
        self.feedback_loop = None  # Will be set after model training
        self.test_calibration = None  # Will be set after model training
        
        # Phase 4: Continuous Improvement
        self.model_retrainer = ModelRetrainer()
        self.drift_detector = DriftDetector()
        self.improvement_orchestrator = ContinuousImprovementOrchestrator(
            self.model_retrainer, self.drift_detector
        )
        self.insight_generator = PerformanceInsightGenerator()
        self.reporter = PerformanceReporter(self.insight_generator)
        
        logger.info("All components initialized successfully")
    
    def run_complete_workflow(self, demo_mode: bool = True):
        """Run the complete 4-phase workflow"""
        logger.info("Starting complete Performance ML workflow")
        
        try:
            # Phase 1: Data Collection & Preprocessing
            logger.info("=== Phase 1: Data Collection & Preprocessing ===")
            phase1_results = self.run_phase1(demo_mode)
            
            # Phase 2: Model Development
            logger.info("=== Phase 2: Model Development ===")
            phase2_results = self.run_phase2(phase1_results['engineered_data'])
            
            # Phase 3: Optimization & Automation
            logger.info("=== Phase 3: Optimization & Automation ===")
            phase3_results = self.run_phase3(phase2_results['trained_models'])
            
            # Phase 4: Continuous Improvement
            logger.info("=== Phase 4: Continuous Improvement ===")
            phase4_results = self.run_phase4(phase1_results['engineered_data'], phase2_results['trained_models'])
            
            # Generate final report
            logger.info("=== Generating Final Report ===")
            final_report = self.generate_final_report(phase1_results, phase2_results, phase3_results, phase4_results)
            
            logger.info("Complete workflow finished successfully!")
            return {
                'phase1': phase1_results,
                'phase2': phase2_results,
                'phase3': phase3_results,
                'phase4': phase4_results,
                'final_report': final_report
            }
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            raise
    
    def run_phase1(self, demo_mode: bool = True) -> dict:
        """Run Phase 1: Data Collection & Preprocessing"""
        logger.info("Starting Phase 1: Data Collection & Preprocessing")
        
        # Step 1: Parameterize Tests
        logger.info("Step 1: Generating parameterized test configurations")
        test_configs = self.test_parameterizer.generate_all_configs()
        self.test_parameterizer.save_configs(test_configs, "test_configurations.json")
        
        # Step 2: Build Data Pipeline
        logger.info("Step 2: Running automated tests")
        if demo_mode:
            # Use demo data instead of running actual tests
            demo_data = self.generate_demo_data()
            self.data_storage.store_results_batch(demo_data)
            raw_data = self.data_storage.get_results()
        else:
            # Run actual tests (limited subset for demo)
            demo_configs = test_configs[:5]  # First 5 tests
            test_results = self.test_runner.run_test_suite(demo_configs)
            raw_data = pd.DataFrame(test_results)
        
        # Step 3: Feature Engineering
        logger.info("Step 3: Engineering features")
        engineered_data = self.feature_engineer.engineer_features(raw_data)
        engineered_data = self.feature_engineer.create_target_variables(engineered_data)
        engineered_data = self.feature_engineer.normalize_features(engineered_data)
        
        # Export engineered data
        engineered_data.to_csv("engineered_data.csv", index=False)
        
        logger.info(f"Phase 1 completed. Generated {len(engineered_data)} engineered samples")
        
        return {
            'test_configs': test_configs,
            'raw_data': raw_data,
            'engineered_data': engineered_data
        }
    
    def run_phase2(self, engineered_data: pd.DataFrame) -> dict:
        """Run Phase 2: Model Development"""
        logger.info("Starting Phase 2: Model Development")
        
        # Step 1: Problem Framing
        logger.info("Step 1: Framing ML problems")
        
        # Regression problem for latency prediction
        X_reg, y_reg = self.problem_framer.frame_regression_problem(engineered_data, 'avg_response_time')
        
        # Classification problem for failure prediction
        X_clf, y_clf = self.problem_framer.frame_classification_problem(engineered_data, 'failure_target')
        
        # Step 2: Model Selection and Training
        logger.info("Step 2: Training models")
        
        # Get models
        regression_models = self.model_selector.get_models_for_problem('regression')
        classification_models = self.model_selector.get_models_for_problem('classification')
        
        trained_models = {}
        model_results = {}
        
        # Train regression models
        for name, model in regression_models.items():
            logger.info(f"Training regression model: {name}")
            result = self.model_trainer.train_regression_model(X_reg, y_reg, name, model, 'test_id')
            trained_models[f"{name}_regression"] = model
            model_results[f"{name}_regression"] = result
        
        # Train classification models
        for name, model in classification_models.items():
            logger.info(f"Training classification model: {name}")
            result = self.model_trainer.train_classification_model(X_clf, y_clf, name, model, 'test_id')
            trained_models[f"{name}_classification"] = model
            model_results[f"{name}_classification"] = result
        
        # Step 3: Hyperparameter Optimization
        logger.info("Step 3: Optimizing hyperparameters")
        best_regression_model = list(regression_models.keys())[0]  # Use first model for optimization
        optimization_result = self.hyperparameter_optimizer.optimize_random_forest_regression(X_reg, y_reg)
        
        # Train optimized model
        from sklearn.ensemble import RandomForestRegressor
        optimized_model = RandomForestRegressor(**optimization_result['best_params'], random_state=42)
        optimized_model.fit(X_reg, y_reg)
        trained_models['optimized_regression'] = optimized_model
        
        logger.info(f"Phase 2 completed. Trained {len(trained_models)} models")
        
        return {
            'trained_models': trained_models,
            'model_results': model_results,
            'optimization_result': optimization_result,
            'X_regression': X_reg,
            'y_regression': y_reg,
            'X_classification': X_clf,
            'y_classification': y_clf
        }
    
    def run_phase3(self, trained_models: dict) -> dict:
        """Run Phase 3: Optimization & Automation"""
        logger.info("Starting Phase 3: Optimization & Automation")
        
        # Get the best regression model for optimization
        best_model = trained_models.get('optimized_regression', list(trained_models.values())[0])
        
        # Step 1: Predict Optimal Load
        logger.info("Step 1: Predicting optimal load configurations")
        load_optimizer = LoadOptimizer(best_model)
        base_config = {'concurrent_users': 100, 'spawn_rate': 5.0, 'duration': 600}
        breaking_point = load_optimizer.find_breaking_point(base_config, n_iterations=10)
        load_recommendations = load_optimizer.generate_load_recommendations(breaking_point)
        
        # Step 2: Anomaly Detection
        logger.info("Step 2: Setting up anomaly detection")
        # Get some data for anomaly detection
        demo_data = self.generate_demo_data()
        df_demo = pd.DataFrame(demo_data)
        
        self.anomaly_detector.fit_models(df_demo, ['avg_response_time', 'requests_per_sec', 'error_rate'])
        anomaly_results = self.anomaly_detector.detect_anomalies(df_demo)
        
        # Step 3: Feedback Loop
        logger.info("Step 3: Setting up feedback loop")
        self.feedback_loop = AnomalyFeedbackLoop(best_model, self.anomaly_detector)
        self.test_calibration = AutomatedTestCalibration(best_model, self.anomaly_detector)
        
        # Test feedback loop
        predictions = {'avg_response_time': 450, 'error_rate': 0.015}
        actual_results = {'avg_response_time': 600, 'error_rate': 0.025}
        feedback_result = self.feedback_loop.compare_predictions_vs_reality(predictions, actual_results)
        
        # Test calibration
        test_config = {
            'test_id': 'test_001',
            'concurrent_users': 800,
            'spawn_rate': 25,
            'duration': 2400,
            'network_condition_encoded': 2
        }
        risk_assessment = self.test_calibration.assess_test_risk(test_config)
        
        logger.info("Phase 3 completed")
        
        return {
            'breaking_point': breaking_point,
            'load_recommendations': load_recommendations,
            'anomaly_results': anomaly_results,
            'feedback_result': feedback_result,
            'risk_assessment': risk_assessment
        }
    
    def run_phase4(self, engineered_data: pd.DataFrame, trained_models: dict) -> dict:
        """Run Phase 4: Continuous Improvement"""
        logger.info("Starting Phase 4: Continuous Improvement")
        
        # Step 1: Model Retraining
        logger.info("Step 1: Setting up continuous improvement")
        
        # Set baseline for drift detection
        self.drift_detector.set_baseline(engineered_data, 'initial')
        
        # Generate some new data with drift
        new_data = self.generate_demo_data_with_drift()
        df_new = pd.DataFrame(new_data)
        
        # Detect drift
        drift_assessment = self.drift_detector.detect_data_drift(df_new, 'initial')
        
        # Run continuous improvement cycle
        cycle_results = self.improvement_orchestrator.run_continuous_improvement_cycle(
            df_new,
            models_to_monitor=['latency_model', 'error_model'],
            target_columns=['avg_response_time', 'error_rate']
        )
        
        # Step 2: Insight Generation
        logger.info("Step 2: Generating insights")
        best_model = trained_models.get('optimized_regression', list(trained_models.values())[0])
        
        # Generate model insights
        feature_cols = [col for col in engineered_data.columns if col not in ['test_id', 'timestamp', 'status', 'error']]
        X_sample = engineered_data[feature_cols].fillna(engineered_data[feature_cols].mean()).iloc[:100]
        
        model_insights = self.insight_generator.generate_model_explanations(best_model, X_sample, "latency_model")
        trend_analysis = self.insight_generator.detect_performance_trends(engineered_data)
        cluster_analysis = self.insight_generator.cluster_performance_patterns(engineered_data)
        
        # Step 3: Reporting
        logger.info("Step 3: Generating comprehensive report")
        models = {'latency_model': best_model}
        report = self.reporter.generate_comprehensive_report(engineered_data, models, "Performance ML System Report")
        
        # Export report
        report_path = self.reporter.export_report(report, 'json')
        
        logger.info("Phase 4 completed")
        
        return {
            'drift_assessment': drift_assessment,
            'cycle_results': cycle_results,
            'model_insights': model_insights,
            'trend_analysis': trend_analysis,
            'cluster_analysis': cluster_analysis,
            'report': report,
            'report_path': report_path
        }
    
    def generate_final_report(self, phase1_results, phase2_results, phase3_results, phase4_results) -> dict:
        """Generate final comprehensive report"""
        logger.info("Generating final comprehensive report")
        
        final_report = {
            'system_overview': {
                'title': 'Performance ML System - Complete Workflow Report',
                'timestamp': datetime.now().isoformat(),
                'phases_completed': ['Data Collection', 'Model Development', 'Optimization', 'Continuous Improvement']
            },
            'phase_summaries': {
                'phase1': {
                    'test_configurations': len(phase1_results['test_configs']),
                    'raw_data_samples': len(phase1_results['raw_data']),
                    'engineered_features': len(phase1_results['engineered_data'].columns)
                },
                'phase2': {
                    'models_trained': len(phase2_results['trained_models']),
                    'optimization_completed': True,
                    'best_model': 'optimized_regression'
                },
                'phase3': {
                    'breaking_point_found': phase3_results['breaking_point']['is_breaking_point'],
                    'anomalies_detected': len(phase3_results['anomaly_results']),
                    'feedback_loop_active': True
                },
                'phase4': {
                    'drift_detected': phase4_results['drift_assessment']['significant_drift'],
                    'improvement_cycle_completed': True,
                    'insights_generated': True,
                    'report_exported': phase4_results['report_path']
                }
            },
            'key_insights': {
                'top_features': phase4_results['model_insights'].get('top_features', []),
                'performance_trends': phase4_results['trend_analysis'].get('overall_trends', {}),
                'cluster_types': phase4_results['cluster_analysis'].get('cluster_types', {}),
                'recommendations': phase4_results['report'].get('recommendations', [])
            },
            'system_status': 'OPERATIONAL',
            'next_steps': [
                "Monitor system performance and model drift",
                "Run regular continuous improvement cycles",
                "Implement automated alerts for anomalies",
                "Scale testing based on load recommendations"
            ]
        }
        
        # Save final report
        import json
        with open('final_report.json', 'w') as f:
            json.dump(final_report, f, indent=2)
        
        logger.info("Final report generated and saved to final_report.json")
        return final_report
    
    def generate_demo_data(self) -> list:
        """Generate demo data for testing"""
        np.random.seed(42)
        n_samples = 500
        
        demo_data = []
        for i in range(n_samples):
            concurrent_users = np.random.randint(10, 500)
            spawn_rate = np.random.uniform(1, 20)
            duration = np.random.randint(300, 1800)
            
            # Simulate realistic performance metrics
            base_latency = 100 + concurrent_users * 2 + spawn_rate * 50
            avg_response_time = base_latency + np.random.normal(0, 100)
            requests_per_sec = concurrent_users * spawn_rate * 0.8 + np.random.normal(0, 10)
            error_rate = max(0, (concurrent_users * spawn_rate) / 10000 - 0.5) * 0.1 + np.random.uniform(0, 0.02)
            
            demo_data.append({
                'test_id': f'test_{i}',
                'test_type': 'api',
                'timestamp': datetime.now().isoformat(),
                'concurrent_users': concurrent_users,
                'spawn_rate': spawn_rate,
                'duration': duration,
                'avg_response_time': max(0, avg_response_time),
                'requests_per_sec': max(0, requests_per_sec),
                'error_rate': min(1, error_rate),
                'network_condition': np.random.choice(['wifi', '4g', '3g']),
                'status': 'success'
            })
        
        return demo_data
    
    def generate_demo_data_with_drift(self) -> list:
        """Generate demo data with drift for testing"""
        np.random.seed(123)
        n_samples = 200
        
        demo_data = []
        for i in range(n_samples):
            concurrent_users = np.random.randint(20, 600)  # Higher range
            spawn_rate = np.random.uniform(1.5, 25)  # Higher spawn rates
            duration = np.random.randint(300, 1800)
            
            # Simulate degraded performance
            base_latency = 150 + concurrent_users * 2.5 + spawn_rate * 60  # Higher latency
            avg_response_time = base_latency + np.random.normal(0, 150)
            requests_per_sec = concurrent_users * spawn_rate * 0.7 + np.random.normal(0, 15)
            error_rate = max(0, (concurrent_users * spawn_rate) / 8000 - 0.6) * 0.15 + np.random.uniform(0, 0.03)
            
            demo_data.append({
                'test_id': f'test_drift_{i}',
                'test_type': 'api',
                'timestamp': datetime.now().isoformat(),
                'concurrent_users': concurrent_users,
                'spawn_rate': spawn_rate,
                'duration': duration,
                'avg_response_time': max(0, avg_response_time),
                'requests_per_sec': max(0, requests_per_sec),
                'error_rate': min(1, error_rate),
                'network_condition': np.random.choice(['wifi', '4g', '3g']),
                'status': 'success'
            })
        
        return demo_data

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Performance ML System')
    parser.add_argument('--base-url', default='http://localhost:5000', help='Base URL for testing')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode (no actual tests)')
    parser.add_argument('--phase', choices=['1', '2', '3', '4', 'all'], default='all', help='Run specific phase or all')
    
    args = parser.parse_args()
    
    # Create system
    system = PerformanceMLSystem(args.base_url)
    
    if args.phase == 'all':
        # Run complete workflow
        results = system.run_complete_workflow(demo_mode=args.demo)
        
        print("\n" + "="*60)
        print("PERFORMANCE ML SYSTEM - WORKFLOW COMPLETED")
        print("="*60)
        print(f"Phase 1: Generated {len(results['phase1']['engineered_data'])} engineered samples")
        print(f"Phase 2: Trained {len(results['phase2']['trained_models'])} models")
        print(f"Phase 3: Found breaking point at {results['phase3']['breaking_point']['concurrent_users']} users")
        print(f"Phase 4: Generated insights and exported report to {results['phase4']['report_path']}")
        print("\nSystem Status: OPERATIONAL")
        print("Check the generated files for detailed results:")
        print("- engineered_data.csv: Engineered features")
        print("- final_report.json: Complete system report")
        print("- perfml.log: Detailed execution log")
        
    else:
        # Run specific phase
        print(f"Running Phase {args.phase} only")
        # Implementation for individual phases would go here
        pass

if __name__ == "__main__":
    main()
