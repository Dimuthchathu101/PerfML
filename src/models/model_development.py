"""
Model Development for Performance ML System
Phase 2: Model Development
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
import optuna
import joblib
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ProblemFramer:
    """Frames different ML problems for performance prediction"""
    
    def __init__(self):
        self.problem_types = {
            'regression': ['latency_prediction', 'throughput_prediction'],
            'classification': ['failure_prediction', 'performance_classification'],
            'anomaly_detection': ['performance_anomaly']
        }
    
    def frame_regression_problem(self, df: pd.DataFrame, target: str = 'avg_response_time') -> Tuple[pd.DataFrame, pd.Series]:
        """Frame regression problem for predicting continuous values"""
        logger.info(f"Framing regression problem for target: {target}")
        
        # Select features and target
        feature_cols = [col for col in df.columns if col not in [
            'test_id', 'timestamp', 'status', 'error', target
        ] and df[col].dtype in ['int64', 'float64']]
        
        X = df[feature_cols].fillna(df[feature_cols].mean())
        y = df[target]
        
        # Remove rows with missing target
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Regression problem: {X.shape[1]} features, {len(y)} samples")
        return X, y
    
    def frame_classification_problem(self, df: pd.DataFrame, target: str = 'failure_target') -> Tuple[pd.DataFrame, pd.Series]:
        """Frame classification problem for predicting discrete outcomes"""
        logger.info(f"Framing classification problem for target: {target}")
        
        # Select features and target
        feature_cols = [col for col in df.columns if col not in [
            'test_id', 'timestamp', 'status', 'error', target
        ] and df[col].dtype in ['int64', 'float64']]
        
        X = df[feature_cols].fillna(df[feature_cols].mean())
        y = df[target]
        
        # Remove rows with missing target
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Classification problem: {X.shape[1]} features, {len(y)} samples, {y.value_counts().to_dict()}")
        return X, y
    
    def frame_anomaly_detection_problem(self, df: pd.DataFrame, target: str = 'anomaly_target') -> Tuple[pd.DataFrame, pd.Series]:
        """Frame anomaly detection problem"""
        logger.info(f"Framing anomaly detection problem for target: {target}")
        
        # Select features and target
        feature_cols = [col for col in df.columns if col not in [
            'test_id', 'timestamp', 'status', 'error', target
        ] and df[col].dtype in ['int64', 'float64']]
        
        X = df[feature_cols].fillna(df[feature_cols].mean())
        y = df[target]
        
        # Remove rows with missing target
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Anomaly detection problem: {X.shape[1]} features, {len(y)} samples, anomalies: {y.sum()}")
        return X, y

class ModelSelector:
    """Selects appropriate models for different problem types"""
    
    def __init__(self):
        self.models = {
            'regression': {
                'random_forest': RandomForestRegressor(random_state=42),
                'xgboost': XGBRegressor(random_state=42),
                'linear': LinearRegression()
            },
            'classification': {
                'random_forest': RandomForestClassifier(random_state=42),
                'xgboost': XGBClassifier(random_state=42),
                'logistic': LogisticRegression(random_state=42)
            }
        }
    
    def get_models_for_problem(self, problem_type: str) -> Dict[str, Any]:
        """Get models for a specific problem type"""
        return self.models.get(problem_type, {})
    
    def get_baseline_model(self, problem_type: str) -> Any:
        """Get a baseline model for quick testing"""
        if problem_type == 'regression':
            return LinearRegression()
        elif problem_type == 'classification':
            return LogisticRegression(random_state=42)
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")

class ModelTrainer:
    """Trains and validates ML models"""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.trained_models = {}
        self.model_metrics = {}
    
    def train_test_split_by_scenario(self, X: pd.DataFrame, y: pd.Series, scenario_col: str = 'test_id') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data by scenario to avoid data leakage"""
        logger.info("Splitting data by scenario to avoid leakage")
        
        # Get unique scenarios
        scenarios = X[scenario_col].unique()
        
        # Split scenarios
        train_scenarios, test_scenarios = train_test_split(
            scenarios, test_size=self.test_size, random_state=self.random_state
        )
        
        # Split data based on scenarios
        train_mask = X[scenario_col].isin(train_scenarios)
        test_mask = X[scenario_col].isin(test_scenarios)
        
        X_train = X[train_mask].drop(columns=[scenario_col])
        X_test = X[test_mask].drop(columns=[scenario_col])
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        return X_train, X_test, y_train, y_test
    
    def train_regression_model(self, X: pd.DataFrame, y: pd.Series, model_name: str, 
                             model: Any, scenario_col: str = 'test_id') -> Dict[str, Any]:
        """Train a regression model"""
        logger.info(f"Training regression model: {model_name}")
        
        # Split data
        X_train, X_test, y_train, y_test = self.train_test_split_by_scenario(X, y, scenario_col)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        
        # Store model and metrics
        self.trained_models[model_name] = model
        self.model_metrics[model_name] = metrics
        
        logger.info(f"Model {model_name} trained. Test R²: {metrics['test_r2']:.4f}, Test RMSE: {metrics['test_rmse']:.4f}")
        
        return {
            'model': model,
            'metrics': metrics,
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test
            }
        }
    
    def train_classification_model(self, X: pd.DataFrame, y: pd.Series, model_name: str,
                                 model: Any, scenario_col: str = 'test_id') -> Dict[str, Any]:
        """Train a classification model"""
        logger.info(f"Training classification model: {model_name}")
        
        # Split data
        X_train, X_test, y_train, y_test = self.train_test_split_by_scenario(X, y, scenario_col)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'train_precision': precision_score(y_train, y_pred_train, average='weighted'),
            'test_precision': precision_score(y_test, y_pred_test, average='weighted'),
            'train_recall': recall_score(y_train, y_pred_train, average='weighted'),
            'test_recall': recall_score(y_test, y_pred_test, average='weighted'),
            'train_f1': f1_score(y_train, y_pred_train, average='weighted'),
            'test_f1': f1_score(y_test, y_pred_test, average='weighted')
        }
        
        # Store model and metrics
        self.trained_models[model_name] = model
        self.model_metrics[model_name] = metrics
        
        logger.info(f"Model {model_name} trained. Test Accuracy: {metrics['test_accuracy']:.4f}, Test F1: {metrics['test_f1']:.4f}")
        
        return {
            'model': model,
            'metrics': metrics,
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test
            }
        }
    
    def cross_validate_model(self, X: pd.DataFrame, y: pd.Series, model: Any, 
                           cv_folds: int = 5, problem_type: str = 'regression') -> Dict[str, float]:
        """Perform cross-validation"""
        logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        # Use time series split for performance data
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        if problem_type == 'regression':
            scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(-scores)
            return {
                'cv_rmse_mean': rmse_scores.mean(),
                'cv_rmse_std': rmse_scores.std(),
                'cv_r2_mean': cross_val_score(model, X, y, cv=tscv, scoring='r2').mean()
            }
        else:
            scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
            return {
                'cv_accuracy_mean': scores.mean(),
                'cv_accuracy_std': scores.std(),
                'cv_f1_mean': cross_val_score(model, X, y, cv=tscv, scoring='f1_weighted').mean()
            }

class HyperparameterOptimizer:
    """Optimizes hyperparameters using Optuna"""
    
    def __init__(self, n_trials: int = 100):
        self.n_trials = n_trials
    
    def optimize_random_forest_regression(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize Random Forest regression hyperparameters"""
        logger.info("Optimizing Random Forest regression hyperparameters")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
            
            model = RandomForestRegressor(**params, random_state=42)
            scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            return np.sqrt(-scores.mean())
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        logger.info(f"Best RMSE: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'study': study
        }
    
    def optimize_xgboost_regression(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize XGBoost regression hyperparameters"""
        logger.info("Optimizing XGBoost regression hyperparameters")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
            }
            
            model = XGBRegressor(**params, random_state=42)
            scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            return np.sqrt(-scores.mean())
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        logger.info(f"Best RMSE: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'study': study
        }
    
    def optimize_classification_model(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'random_forest') -> Dict[str, Any]:
        """Optimize classification model hyperparameters"""
        logger.info(f"Optimizing {model_type} classification hyperparameters")
        
        def objective(trial):
            if model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                }
                model = RandomForestClassifier(**params, random_state=42)
            elif model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                }
                model = XGBClassifier(**params, random_state=42)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
            return -scores.mean()  # Negative because we want to maximize F1
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        logger.info(f"Best F1: {-study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")
        
        return {
            'best_params': study.best_params,
            'best_score': -study.best_value,
            'study': study
        }

class ModelEvaluator:
    """Evaluates model performance and provides insights"""
    
    def __init__(self):
        self.evaluation_results = {}
    
    def evaluate_regression_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                                model_name: str) -> Dict[str, Any]:
        """Evaluate regression model performance"""
        logger.info(f"Evaluating regression model: {model_name}")
        
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Calculate percentage errors
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        evaluation = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'predictions': y_pred,
            'actual': y_test
        }
        
        self.evaluation_results[model_name] = evaluation
        
        logger.info(f"Model {model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
        
        return evaluation
    
    def evaluate_classification_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                                   model_name: str) -> Dict[str, Any]:
        """Evaluate classification model performance"""
        logger.info(f"Evaluating classification model: {model_name}")
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        evaluation = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'classification_report': class_report,
            'predictions': y_pred,
            'prediction_proba': y_pred_proba,
            'actual': y_test
        }
        
        self.evaluation_results[model_name] = evaluation
        
        logger.info(f"Model {model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return evaluation
    
    def compare_models(self, problem_type: str = 'regression') -> pd.DataFrame:
        """Compare multiple models"""
        if not self.evaluation_results:
            logger.warning("No evaluation results to compare")
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, results in self.evaluation_results.items():
            if problem_type == 'regression':
                comparison_data.append({
                    'model': model_name,
                    'mae': results['mae'],
                    'rmse': results['rmse'],
                    'r2': results['r2'],
                    'mape': results['mape']
                })
            else:
                comparison_data.append({
                    'model': model_name,
                    'accuracy': results['accuracy'],
                    'precision': results['precision'],
                    'recall': results['recall'],
                    'f1': results['f1']
                })
        
        return pd.DataFrame(comparison_data)

def main():
    """Example usage of model development"""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'test_id': [f'test_{i}' for i in range(n_samples)],
        'concurrent_users': np.random.randint(10, 500, n_samples),
        'spawn_rate': np.random.uniform(1, 20, n_samples),
        'duration': np.random.randint(300, 1800, n_samples),
        'avg_response_time': np.random.uniform(100, 5000, n_samples),
        'error_rate': np.random.uniform(0, 0.1, n_samples),
        'network_condition_encoded': np.random.randint(0, 4, n_samples),
        'load_intensity': np.random.uniform(0.1, 10, n_samples),
        'throughput_per_user': np.random.uniform(0.1, 5, n_samples)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize components
    framer = ProblemFramer()
    selector = ModelSelector()
    trainer = ModelTrainer()
    optimizer = HyperparameterOptimizer(n_trials=20)  # Reduced for demo
    evaluator = ModelEvaluator()
    
    # Frame regression problem
    X_reg, y_reg = framer.frame_regression_problem(df, 'avg_response_time')
    
    # Get models
    regression_models = selector.get_models_for_problem('regression')
    
    # Train and evaluate models
    for name, model in regression_models.items():
        # Train model
        result = trainer.train_regression_model(X_reg, y_reg, name, model, 'test_id')
        
        # Evaluate model
        X_train, X_test, y_train, y_test = trainer.train_test_split_by_scenario(X_reg, y_reg, 'test_id')
        evaluator.evaluate_regression_model(model, X_test, y_test, name)
    
    # Compare models
    comparison = evaluator.compare_models('regression')
    print("\nModel Comparison:")
    print(comparison)
    
    # Optimize best model
    best_model_name = comparison.loc[comparison['r2'].idxmax(), 'model']
    print(f"\nOptimizing best model: {best_model_name}")
    
    if 'random_forest' in best_model_name:
        optimization_result = optimizer.optimize_random_forest_regression(X_reg, y_reg)
    elif 'xgboost' in best_model_name:
        optimization_result = optimizer.optimize_xgboost_regression(X_reg, y_reg)
    
    print(f"Optimization completed. Best score: {optimization_result['best_score']:.4f}")

if __name__ == "__main__":
    main()
