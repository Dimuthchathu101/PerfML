# Performance ML System 🚀

A comprehensive Machine Learning-driven performance testing and optimization system that transforms traditional load testing into an intelligent, self-learning platform.

## 🎯 Overview

This system implements a complete 4-phase approach to performance testing with ML:

1. **Data Collection & Preprocessing** - Parameterized tests, automated data pipeline, feature engineering
2. **Model Development** - Problem framing, model selection, training & validation
3. **Optimization & Automation** - Bayesian optimization, anomaly detection, feedback loops
4. **Continuous Improvement** - Model retraining, drift detection, insight generation

## 🏗️ Architecture

```
PerfML/
├── dummy_website/           # Test website for performance testing
│   ├── app.py              # Flask application with various endpoints
│   └── templates/          # HTML templates for frontend testing
├── src/                    # Main source code
│   ├── data_collection/    # Phase 1: Data collection components
│   ├── models/             # Phase 2: Model development
│   ├── optimization/       # Phase 3: Optimization & automation
│   └── continuous_improvement/ # Phase 4: Continuous improvement
├── main.py                 # Main orchestration script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd PerfML

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the Dummy Website

```bash
# Start the test website
cd dummy_website
python app.py
```

The website will be available at `http://localhost:5000`

### 3. Run the Complete System

```bash
# Run the complete 4-phase workflow in demo mode
python main.py --demo

# Run with actual testing (requires the website to be running)
python main.py --base-url http://localhost:5000
```

## 📊 System Components

### Phase 1: Data Collection & Preprocessing

#### Parameterized Tests (`src/data_collection/parameterized_tests.py`)
- Generates comprehensive test configurations
- Supports API, frontend, load, and stress testing
- Configurable network conditions, browsers, user loads

#### Data Pipeline (`src/data_collection/data_pipeline.py`)
- Automated test execution with Locust and Lighthouse
- Structured data storage in SQLite
- Parallel test execution with result aggregation

#### Feature Engineering (`src/data_collection/feature_engineering.py`)
- Creates 50+ derived features from raw metrics
- Handles normalization and feature selection
- Generates target variables for different ML tasks

### Phase 2: Model Development

#### Problem Framing (`src/models/model_development.py`)
- Regression: Predict latency given load conditions
- Classification: Predict failure probability
- Anomaly Detection: Identify performance outliers

#### Model Training
- Multiple algorithms: Random Forest, XGBoost, Linear models
- Cross-validation with scenario-based splits
- Hyperparameter optimization with Optuna

### Phase 3: Optimization & Automation

#### Bayesian Optimization (`src/optimization/bayesian_optimization.py`)
- Finds optimal load configurations
- Predicts breaking points
- Generates load recommendations

#### Anomaly Detection (`src/optimization/anomaly_detection.py`)
- Detects performance anomalies
- Implements feedback loops
- Automated test calibration

### Phase 4: Continuous Improvement

#### Model Retraining (`src/continuous_improvement/model_retraining.py`)
- Automatic model retraining
- Drift detection and monitoring
- Version control for models

#### Insight Generation (`src/continuous_improvement/insight_generation.py`)
- Feature importance analysis
- Performance trend detection
- Clustering of performance patterns
- Automated reporting

## 🔧 Usage Examples

### Basic Demo Run

```bash
# Run complete system with demo data
python main.py --demo
```

### Custom Configuration

```bash
# Run with custom base URL
python main.py --base-url http://your-app.com

# Run specific phase only
python main.py --phase 1  # Data collection only
python main.py --phase 2  # Model development only
python main.py --phase 3  # Optimization only
python main.py --phase 4  # Continuous improvement only
```

### Individual Component Usage

```python
# Generate test configurations
from src.data_collection.parameterized_tests import TestParameterizer
parameterizer = TestParameterizer()
configs = parameterizer.generate_all_configs()

# Run automated tests
from src.data_collection.data_pipeline import AutomatedTestRunner
runner = AutomatedTestRunner()
results = runner.run_test_suite(configs)

# Engineer features
from src.data_collection.feature_engineering import PerformanceFeatureEngineer
engineer = PerformanceFeatureEngineer()
engineered_data = engineer.engineer_features(raw_data)

# Train models
from src.models.model_development import ModelTrainer
trainer = ModelTrainer()
model = trainer.train_regression_model(X, y, "latency_model", RandomForestRegressor())

# Optimize load
from src.optimization.bayesian_optimization import LoadOptimizer
optimizer = LoadOptimizer(model)
breaking_point = optimizer.find_breaking_point(base_config)

# Generate insights
from src.continuous_improvement.insight_generation import PerformanceInsightGenerator
insight_gen = PerformanceInsightGenerator()
insights = insight_gen.generate_model_explanations(model, X)
```

## 📈 Key Features

### 🔍 Intelligent Test Generation
- **Parameterized Tests**: 100+ test configurations covering various scenarios
- **Network Simulation**: WiFi, 4G, 3G, slow 3G conditions
- **Browser Testing**: Chrome, Firefox, Safari, Edge
- **Load Patterns**: Ramp-up, sustained, peak, stress testing

### 🤖 ML-Powered Analysis
- **Regression Models**: Predict latency, throughput, error rates
- **Classification Models**: Predict failure probability
- **Anomaly Detection**: Identify performance outliers
- **Feature Engineering**: 50+ derived features from raw metrics

### 🎯 Optimization Engine
- **Bayesian Optimization**: Find optimal load configurations
- **Breaking Point Detection**: Predict system limits
- **Load Recommendations**: Safe, stress, and optimal load levels
- **Automated Calibration**: Self-adjusting test parameters

### 📊 Continuous Monitoring
- **Drift Detection**: Monitor data and model drift
- **Automated Retraining**: Retrain models when needed
- **Performance Trends**: Track improvements over time
- **Anomaly Alerts**: Real-time performance monitoring

### 📋 Comprehensive Reporting
- **Model Insights**: Feature importance and explanations
- **Performance Trends**: Historical analysis and predictions
- **Cluster Analysis**: Group similar performance patterns
- **Actionable Recommendations**: Specific optimization suggestions

## 📊 Output Files

After running the system, you'll get:

- `engineered_data.csv` - Processed features and targets
- `test_configurations.json` - Generated test configurations
- `performance_data.db` - SQLite database with test results
- `final_report.json` - Comprehensive system report
- `perfml.log` - Detailed execution log
- `models/` - Trained ML models with versioning

## 🛠️ Configuration

### Environment Variables

```bash
export PERFML_BASE_URL=http://localhost:5000
export PERFML_DB_PATH=performance_data.db
export PERFML_MODEL_DIR=models/
export PERFML_LOG_LEVEL=INFO
```

### Custom Test Configurations

Edit `src/data_collection/parameterized_tests.py` to customize:
- Test scenarios
- Network conditions
- Browser configurations
- Load patterns

## 🔧 Dependencies

### Core ML Libraries
- `scikit-learn` - Machine learning algorithms
- `xgboost` - Gradient boosting
- `pandas` - Data manipulation
- `numpy` - Numerical computing

### Testing Tools
- `locust` - Load testing
- `lighthouse-ci` - Frontend performance testing
- `selenium` - Browser automation

### Optimization
- `optuna` - Hyperparameter optimization
- `scikit-optimize` - Bayesian optimization

### Visualization & Reporting
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `plotly` - Interactive charts

## 🚀 Advanced Usage

### Custom Model Integration

```python
# Add your custom model
from src.models.model_development import ModelSelector
selector = ModelSelector()
selector.models['regression']['custom_model'] = YourCustomModel()
```

### Custom Feature Engineering

```python
# Add custom features
from src.data_collection.feature_engineering import PerformanceFeatureEngineer
engineer = PerformanceFeatureEngineer()

def custom_feature_method(self, df):
    df['custom_feature'] = df['feature1'] * df['feature2']
    return df

engineer._create_custom_features = custom_feature_method
```

### Custom Optimization Objectives

```python
# Define custom optimization objective
from src.optimization.bayesian_optimization import BayesianOptimizer

def custom_objective(X):
    # Your custom objective function
    return custom_score

optimizer = BayesianOptimizer(model, feature_names, bounds)
optimizer.objective_function = custom_objective
```

## 📚 API Reference

### Main Classes

#### `PerformanceMLSystem`
Main orchestrator class that runs the complete workflow.

```python
system = PerformanceMLSystem(base_url="http://localhost:5000")
results = system.run_complete_workflow(demo_mode=True)
```

#### `TestParameterizer`
Generates parameterized test configurations.

```python
parameterizer = TestParameterizer()
configs = parameterizer.generate_api_test_configs()
```

#### `AutomatedTestRunner`
Runs automated performance tests.

```python
runner = AutomatedTestRunner()
results = runner.run_test_suite(configs)
```

#### `PerformanceFeatureEngineer`
Engineers features from raw performance data.

```python
engineer = PerformanceFeatureEngineer()
engineered_data = engineer.engineer_features(raw_data)
```

#### `LoadOptimizer`
Optimizes load testing configurations.

```python
optimizer = LoadOptimizer(model)
breaking_point = optimizer.find_breaking_point(base_config)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation in each module
- Review the example usage in `main.py`

## 🎯 Roadmap

- [ ] Real-time monitoring dashboard
- [ ] Integration with CI/CD pipelines
- [ ] Support for microservices testing
- [ ] Advanced visualization components
- [ ] Cloud deployment support
- [ ] API for external integrations

---

**Performance ML System** - Transforming performance testing with the power of Machine Learning! 🚀
