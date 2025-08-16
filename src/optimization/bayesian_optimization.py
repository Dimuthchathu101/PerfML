"""
Bayesian Optimization for Performance ML System
Phase 3: Optimization & Automation - Step 1
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class BayesianOptimizer:
    """Bayesian optimization for finding optimal performance configurations"""
    
    def __init__(self, model, feature_names: List[str], bounds: Dict[str, Tuple[float, float]]):
        self.model = model
        self.feature_names = feature_names
        self.bounds = bounds
        self.scaler = StandardScaler()
        self.gp = None
        self.X_observed = []
        self.y_observed = []
        self.optimization_history = []
        
    def objective_function(self, X: np.ndarray) -> float:
        """Objective function to optimize (e.g., minimize latency, maximize throughput)"""
        # Reshape for single prediction
        X_reshaped = X.reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X_reshaped)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        
        return prediction
    
    def acquisition_function(self, X: np.ndarray, xi: float = 0.01) -> float:
        """Expected Improvement acquisition function"""
        if self.gp is None:
            return 0.0
        
        X_reshaped = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X_reshaped)
        
        # Get GP prediction
        mean, std = self.gp.predict(X_scaled, return_std=True)
        
        # Calculate expected improvement
        best_observed = min(self.y_observed) if self.y_observed else 0
        improvement = best_observed - mean - xi
        
        # Calculate acquisition value
        if std > 0:
            z = improvement / std
            acquisition = improvement * norm.cdf(z) + std * norm.pdf(z)
        else:
            acquisition = 0.0
        
        return acquisition
    
    def update_gp(self):
        """Update Gaussian Process with observed data"""
        if len(self.X_observed) < 2:
            return
        
        X_array = np.array(self.X_observed)
        y_array = np.array(self.y_observed)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_array)
        
        # Define kernel
        kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0] * X_scaled.shape[1], (1e-2, 1e2))
        
        # Fit Gaussian Process
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, random_state=42)
        self.gp.fit(X_scaled, y_array)
    
    def suggest_next_point(self) -> Dict[str, float]:
        """Suggest the next point to evaluate using acquisition function"""
        if len(self.X_observed) < 2:
            # Random initialization
            next_point = {}
            for feature, (low, high) in self.bounds.items():
                next_point[feature] = np.random.uniform(low, high)
            return next_point
        
        # Update GP
        self.update_gp()
        
        # Optimize acquisition function
        def neg_acquisition(X):
            return -self.acquisition_function(X)
        
        # Generate random starting points
        n_starts = 10
        best_acquisition = float('inf')
        best_X = None
        
        for _ in range(n_starts):
            # Random starting point
            X_start = []
            for feature in self.feature_names:
                low, high = self.bounds[feature]
                X_start.append(np.random.uniform(low, high))
            X_start = np.array(X_start)
            
            # Optimize
            result = minimize(neg_acquisition, X_start, method='L-BFGS-B',
                            bounds=[self.bounds[feature] for feature in self.feature_names])
            
            if result.fun < best_acquisition:
                best_acquisition = result.fun
                best_X = result.x
        
        # Convert to dictionary
        next_point = {}
        for i, feature in enumerate(self.feature_names):
            next_point[feature] = best_X[i]
        
        return next_point
    
    def add_observation(self, X: Dict[str, float], y: float):
        """Add a new observation to the optimization history"""
        # Convert to array
        X_array = [X[feature] for feature in self.feature_names]
        
        self.X_observed.append(X_array)
        self.y_observed.append(y)
        
        # Store optimization history
        observation = {
            'timestamp': datetime.now(),
            'X': X.copy(),
            'y': y,
            'iteration': len(self.X_observed)
        }
        self.optimization_history.append(observation)
        
        logger.info(f"Added observation {len(self.X_observed)}: {y:.4f} for {X}")
    
    def optimize(self, n_iterations: int = 50, initial_points: int = 5) -> Dict[str, Any]:
        """Run Bayesian optimization"""
        logger.info(f"Starting Bayesian optimization with {n_iterations} iterations")
        
        # Initial random points
        for _ in range(initial_points):
            X_random = {}
            for feature, (low, high) in self.bounds.items():
                X_random[feature] = np.random.uniform(low, high)
            
            y_random = self.objective_function(np.array([X_random[feature] for feature in self.feature_names]))
            self.add_observation(X_random, y_random)
        
        # Bayesian optimization iterations
        for i in range(n_iterations):
            logger.info(f"Optimization iteration {i + 1}/{n_iterations}")
            
            # Suggest next point
            next_point = self.suggest_next_point()
            
            # Evaluate objective function
            y_next = self.objective_function(np.array([next_point[feature] for feature in self.feature_names]))
            
            # Add observation
            self.add_observation(next_point, y_next)
        
        # Find best solution
        best_idx = np.argmin(self.y_observed)
        best_X = self.X_observed[best_idx]
        best_y = self.y_observed[best_idx]
        
        best_solution = {}
        for i, feature in enumerate(self.feature_names):
            best_solution[feature] = best_X[i]
        
        results = {
            'best_solution': best_solution,
            'best_objective': best_y,
            'optimization_history': self.optimization_history,
            'X_observed': self.X_observed,
            'y_observed': self.y_observed
        }
        
        logger.info(f"Optimization completed. Best objective: {best_y:.4f}")
        logger.info(f"Best solution: {best_solution}")
        
        return results
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history"""
        if not self.optimization_history:
            logger.warning("No optimization history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Objective function values over iterations
        iterations = [obs['iteration'] for obs in self.optimization_history]
        objectives = [obs['y'] for obs in self.optimization_history]
        
        axes[0, 0].plot(iterations, objectives, 'b-o')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Objective Value')
        axes[0, 0].set_title('Objective Function Values')
        axes[0, 0].grid(True)
        
        # Best objective so far
        best_so_far = [min(objectives[:i+1]) for i in range(len(objectives))]
        axes[0, 1].plot(iterations, best_so_far, 'r-o')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Best Objective Value')
        axes[0, 1].set_title('Best Objective So Far')
        axes[0, 1].grid(True)
        
        # Feature values over iterations
        for i, feature in enumerate(self.feature_names):
            feature_values = [obs['X'][feature] for obs in self.optimization_history]
            axes[1, 0].plot(iterations, feature_values, label=feature, marker='o')
        
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Feature Values')
        axes[1, 0].set_title('Feature Values Over Iterations')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Distribution of objective values
        axes[1, 1].hist(objectives, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Objective Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Objective Values')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Optimization history plot saved to {save_path}")
        
        plt.show()

class LoadOptimizer:
    """Optimizes load testing configurations to find breaking points"""
    
    def __init__(self, performance_model, latency_threshold: float = 2000, error_threshold: float = 0.05):
        self.performance_model = performance_model
        self.latency_threshold = latency_threshold  # ms
        self.error_threshold = error_threshold  # 5%
        self.optimizer = None
        
    def setup_optimization(self, base_config: Dict[str, Any]):
        """Setup optimization problem"""
        # Define bounds for optimization
        bounds = {
            'concurrent_users': (10, 1000),
            'spawn_rate': (0.1, 50.0),
            'duration': (300, 3600)
        }
        
        # Feature names for the model
        feature_names = ['concurrent_users', 'spawn_rate', 'duration', 'network_condition_encoded']
        
        # Create Bayesian optimizer
        self.optimizer = BayesianOptimizer(
            model=self.performance_model,
            feature_names=feature_names,
            bounds=bounds
        )
        
        # Override objective function for load optimization
        self.optimizer.objective_function = self._load_objective_function
        
        return self.optimizer
    
    def _load_objective_function(self, X: np.ndarray) -> float:
        """Objective function for load optimization"""
        # Extract parameters
        concurrent_users, spawn_rate, duration = X[:3]
        network_condition = 0  # Default to WiFi
        
        # Create feature vector
        features = np.array([concurrent_users, spawn_rate, duration, network_condition])
        
        # Predict latency
        predicted_latency = self.performance_model.predict(features.reshape(1, -1))[0]
        
        # Calculate penalty for exceeding thresholds
        latency_penalty = max(0, predicted_latency - self.latency_threshold) * 10
        
        # Estimate error rate (simplified)
        estimated_error_rate = max(0, (concurrent_users * spawn_rate) / 1000 - 0.5) * 0.1
        error_penalty = max(0, estimated_error_rate - self.error_threshold) * 1000
        
        # Total objective (minimize)
        objective = predicted_latency + latency_penalty + error_penalty
        
        return objective
    
    def find_breaking_point(self, base_config: Dict[str, Any], n_iterations: int = 30) -> Dict[str, Any]:
        """Find the breaking point for the system"""
        logger.info("Finding system breaking point")
        
        # Setup optimization
        optimizer = self.setup_optimization(base_config)
        
        # Run optimization
        results = optimizer.optimize(n_iterations=n_iterations, initial_points=5)
        
        # Analyze results
        breaking_point = self._analyze_breaking_point(results)
        
        return breaking_point
    
    def _analyze_breaking_point(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze optimization results to find breaking point"""
        best_solution = optimization_results['best_solution']
        best_objective = optimization_results['best_objective']
        
        # Calculate predicted performance at breaking point
        concurrent_users = best_solution['concurrent_users']
        spawn_rate = best_solution['spawn_rate']
        duration = best_solution['duration']
        
        # Predict latency
        features = np.array([concurrent_users, spawn_rate, duration, 0])  # WiFi
        predicted_latency = self.performance_model.predict(features.reshape(1, -1))[0]
        
        # Estimate throughput
        estimated_throughput = concurrent_users * spawn_rate
        
        # Estimate error rate
        estimated_error_rate = max(0, (concurrent_users * spawn_rate) / 1000 - 0.5) * 0.1
        
        breaking_point = {
            'concurrent_users': int(concurrent_users),
            'spawn_rate': round(spawn_rate, 2),
            'duration': int(duration),
            'predicted_latency': round(predicted_latency, 2),
            'estimated_throughput': round(estimated_throughput, 2),
            'estimated_error_rate': round(estimated_error_rate, 4),
            'objective_value': round(best_objective, 2),
            'is_breaking_point': predicted_latency > self.latency_threshold or estimated_error_rate > self.error_threshold
        }
        
        logger.info(f"Breaking point found: {breaking_point}")
        
        return breaking_point
    
    def generate_load_recommendations(self, breaking_point: Dict[str, Any]) -> Dict[str, Any]:
        """Generate load testing recommendations based on breaking point"""
        safe_users = int(breaking_point['concurrent_users'] * 0.8)  # 80% of breaking point
        safe_spawn_rate = breaking_point['spawn_rate'] * 0.8
        
        recommendations = {
            'safe_load': {
                'concurrent_users': safe_users,
                'spawn_rate': round(safe_spawn_rate, 2),
                'duration': breaking_point['duration'],
                'expected_latency': round(breaking_point['predicted_latency'] * 0.7, 2),
                'expected_throughput': round(breaking_point['estimated_throughput'] * 0.8, 2)
            },
            'stress_test': {
                'concurrent_users': int(breaking_point['concurrent_users'] * 1.1),
                'spawn_rate': round(breaking_point['spawn_rate'] * 1.1, 2),
                'duration': breaking_point['duration'],
                'expected_latency': round(breaking_point['predicted_latency'] * 1.2, 2),
                'expected_throughput': round(breaking_point['estimated_throughput'] * 1.1, 2)
            },
            'breaking_point': breaking_point,
            'recommendations': [
                f"Safe load: {safe_users} users at {safe_spawn_rate} users/sec",
                f"Stress test: {int(breaking_point['concurrent_users'] * 1.1)} users at {round(breaking_point['spawn_rate'] * 1.1, 2)} users/sec",
                f"Expected breaking point: {breaking_point['concurrent_users']} users",
                f"Monitor latency threshold: {self.latency_threshold}ms",
                f"Monitor error rate threshold: {self.error_threshold * 100}%"
            ]
        }
        
        return recommendations

class PerformancePredictor:
    """Predicts optimal performance configurations"""
    
    def __init__(self, latency_model, throughput_model, error_model):
        self.latency_model = latency_model
        self.throughput_model = throughput_model
        self.error_model = error_model
    
    def predict_optimal_config(self, target_latency: float = 1000, 
                             target_throughput: float = 100,
                             max_error_rate: float = 0.01) -> Dict[str, Any]:
        """Predict optimal configuration for given targets"""
        logger.info(f"Predicting optimal config: latency<{target_latency}ms, throughput>{target_throughput}/s, error<{max_error_rate*100}%")
        
        # Define optimization bounds
        bounds = {
            'concurrent_users': (10, 500),
            'spawn_rate': (0.1, 20.0),
            'duration': (300, 1800)
        }
        
        # Create optimizer
        optimizer = BayesianOptimizer(
            model=self.latency_model,
            feature_names=['concurrent_users', 'spawn_rate', 'duration', 'network_condition_encoded'],
            bounds=bounds
        )
        
        # Custom objective function
        def objective(X):
            users, spawn_rate, duration = X[:3]
            network = 0  # WiFi
            
            # Predict performance
            features = np.array([users, spawn_rate, duration, network])
            predicted_latency = self.latency_model.predict(features.reshape(1, -1))[0]
            predicted_throughput = users * spawn_rate
            predicted_error = max(0, (users * spawn_rate) / 1000 - 0.5) * 0.1
            
            # Calculate penalties
            latency_penalty = max(0, predicted_latency - target_latency) * 10
            throughput_penalty = max(0, target_throughput - predicted_throughput) * 5
            error_penalty = max(0, predicted_error - max_error_rate) * 1000
            
            return predicted_latency + latency_penalty + throughput_penalty + error_penalty
        
        optimizer.objective_function = objective
        
        # Run optimization
        results = optimizer.optimize(n_iterations=30, initial_points=5)
        
        # Extract optimal configuration
        optimal_config = results['best_solution']
        
        # Predict performance at optimal config
        users = optimal_config['concurrent_users']
        spawn_rate = optimal_config['spawn_rate']
        duration = optimal_config['duration']
        
        features = np.array([users, spawn_rate, duration, 0])
        predicted_latency = self.latency_model.predict(features.reshape(1, -1))[0]
        predicted_throughput = users * spawn_rate
        predicted_error = max(0, (users * spawn_rate) / 1000 - 0.5) * 0.1
        
        optimal_config.update({
            'predicted_latency': round(predicted_latency, 2),
            'predicted_throughput': round(predicted_throughput, 2),
            'predicted_error_rate': round(predicted_error, 4),
            'meets_targets': (
                predicted_latency <= target_latency and
                predicted_throughput >= target_throughput and
                predicted_error <= max_error_rate
            )
        })
        
        logger.info(f"Optimal configuration: {optimal_config}")
        
        return optimal_config

def main():
    """Example usage of Bayesian optimization"""
    # Create a mock performance model
    from sklearn.ensemble import RandomForestRegressor
    
    # Generate sample data for training
    np.random.seed(42)
    n_samples = 1000
    
    X_sample = np.random.rand(n_samples, 4)
    X_sample[:, 0] = np.random.randint(10, 500, n_samples)  # concurrent_users
    X_sample[:, 1] = np.random.uniform(0.1, 20, n_samples)  # spawn_rate
    X_sample[:, 2] = np.random.randint(300, 1800, n_samples)  # duration
    X_sample[:, 3] = np.random.randint(0, 4, n_samples)  # network_condition
    
    # Create realistic latency function
    y_sample = (
        100 +  # Base latency
        X_sample[:, 0] * 2 +  # Users impact
        X_sample[:, 1] * 50 +  # Spawn rate impact
        X_sample[:, 2] * 0.01 +  # Duration impact
        X_sample[:, 3] * 200 +  # Network impact
        np.random.normal(0, 50, n_samples)  # Noise
    )
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_sample, y_sample)
    
    # Test optimization
    bounds = {
        'concurrent_users': (10, 500),
        'spawn_rate': (0.1, 20.0),
        'duration': (300, 1800),
        'network_condition_encoded': (0, 3)
    }
    
    feature_names = ['concurrent_users', 'spawn_rate', 'duration', 'network_condition_encoded']
    
    optimizer = BayesianOptimizer(model, feature_names, bounds)
    
    # Run optimization
    results = optimizer.optimize(n_iterations=20, initial_points=3)
    
    print("Optimization Results:")
    print(f"Best solution: {results['best_solution']}")
    print(f"Best objective: {results['best_objective']:.2f}")
    
    # Test load optimizer
    load_optimizer = LoadOptimizer(model)
    base_config = {'concurrent_users': 100, 'spawn_rate': 5.0, 'duration': 600}
    
    breaking_point = load_optimizer.find_breaking_point(base_config, n_iterations=15)
    recommendations = load_optimizer.generate_load_recommendations(breaking_point)
    
    print("\nLoad Optimization Results:")
    print(f"Breaking point: {breaking_point}")
    print(f"Recommendations: {recommendations['recommendations']}")

if __name__ == "__main__":
    main()
