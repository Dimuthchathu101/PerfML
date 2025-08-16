"""
Insight Generation and Reporting for Performance ML System
Phase 4: Continuous Improvement - Step 2
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PerformanceInsightGenerator:
    """Generates insights from performance data and ML models"""
    
    def __init__(self):
        self.insights_history = []
        self.feature_importance_cache = {}
        
    def generate_model_explanations(self, model, X: pd.DataFrame, 
                                  model_name: str = "performance_model") -> Dict[str, Any]:
        """Generate explanations for model predictions"""
        logger.info(f"Generating explanations for {model_name}")
        
        try:
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                feature_names = X.columns.tolist()
                
                # Create feature importance dictionary
                importance_dict = dict(zip(feature_names, feature_importance))
                sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                
                # Generate insights
                insights = {
                    'model_name': model_name,
                    'timestamp': datetime.now().isoformat(),
                    'top_features': sorted_importance[:10],
                    'feature_importance': importance_dict,
                    'total_features': len(feature_names),
                    'explanation_summary': self._generate_explanation_summary(sorted_importance)
                }
                
                # Store in cache
                self.feature_importance_cache[model_name] = insights
                
                logger.info(f"Generated explanations for {len(feature_names)} features")
                
                return insights
                
        except Exception as e:
            logger.error(f"Error generating explanations: {e}")
            return {
                'model_name': model_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_explanation_summary(self, sorted_importance: List[Tuple[str, float]]) -> List[str]:
        """Generate human-readable explanation summary"""
        summary = []
        
        if len(sorted_importance) > 0:
            top_feature, top_importance = sorted_importance[0]
            summary.append(f"'{top_feature}' is the most important feature ({top_importance:.2%})")
        
        if len(sorted_importance) > 1:
            second_feature, second_importance = sorted_importance[1]
            summary.append(f"'{second_feature}' is the second most important feature ({second_importance:.2%})")
        
        # Identify feature categories
        load_features = [f for f, _ in sorted_importance if 'load' in f.lower() or 'user' in f.lower()]
        network_features = [f for f, _ in sorted_importance if 'network' in f.lower()]
        time_features = [f for f, _ in sorted_importance if 'time' in f.lower() or 'duration' in f.lower()]
        
        if load_features:
            summary.append(f"Load-related features ({len(load_features)} total) significantly impact performance")
        
        if network_features:
            summary.append(f"Network conditions ({len(network_features)} features) are important predictors")
        
        if time_features:
            summary.append(f"Time-based features ({len(time_features)} features) show temporal patterns")
        
        return summary
    
    def cluster_performance_patterns(self, df: pd.DataFrame, 
                                   n_clusters: int = 3) -> Dict[str, Any]:
        """Cluster performance patterns to identify common scenarios"""
        logger.info(f"Clustering performance patterns into {n_clusters} groups")
        
        # Select numerical features for clustering
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['test_id', 'timestamp', 'id']
        feature_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        if len(feature_cols) < 2:
            logger.warning("Insufficient features for clustering")
            return {'error': 'Insufficient features for clustering'}
        
        # Prepare data
        X = df[feature_cols].fillna(df[feature_cols].mean())
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to dataframe
        df_clustered = df.copy()
        df_clustered['cluster'] = cluster_labels
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(n_clusters):
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
            
            cluster_analysis[cluster_id] = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df) * 100,
                'characteristics': self._analyze_cluster_characteristics(cluster_data, feature_cols),
                'performance_profile': self._generate_performance_profile(cluster_data)
            }
        
        # Identify cluster types
        cluster_types = self._identify_cluster_types(cluster_analysis)
        
        insights = {
            'n_clusters': n_clusters,
            'cluster_analysis': cluster_analysis,
            'cluster_types': cluster_types,
            'feature_importance': dict(zip(feature_cols, kmeans.cluster_centers_.std(axis=0)))
        }
        
        logger.info(f"Clustering completed. Cluster sizes: {[cluster_analysis[i]['size'] for i in range(n_clusters)]}")
        
        return insights
    
    def _analyze_cluster_characteristics(self, cluster_data: pd.DataFrame, 
                                       feature_cols: List[str]) -> Dict[str, Any]:
        """Analyze characteristics of a cluster"""
        characteristics = {}
        
        for feature in feature_cols:
            if feature in cluster_data.columns:
                values = cluster_data[feature].dropna()
                if len(values) > 0:
                    characteristics[feature] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'median': float(values.median())
                    }
        
        return characteristics
    
    def _generate_performance_profile(self, cluster_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate performance profile for a cluster"""
        profile = {}
        
        # Check for common performance metrics
        performance_metrics = ['avg_response_time', 'requests_per_sec', 'error_rate', 
                             'latency_target', 'throughput_target']
        
        for metric in performance_metrics:
            if metric in cluster_data.columns:
                values = cluster_data[metric].dropna()
                if len(values) > 0:
                    profile[metric] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'range': f"{float(values.min()):.2f} - {float(values.max()):.2f}"
                    }
        
        return profile
    
    def _identify_cluster_types(self, cluster_analysis: Dict[int, Dict]) -> Dict[int, str]:
        """Identify the type of each cluster"""
        cluster_types = {}
        
        for cluster_id, analysis in cluster_analysis.items():
            profile = analysis['performance_profile']
            
            # Determine cluster type based on performance characteristics
            if 'avg_response_time' in profile:
                avg_latency = profile['avg_response_time']['mean']
                if avg_latency < 500:
                    cluster_type = "High Performance"
                elif avg_latency < 1500:
                    cluster_type = "Medium Performance"
                else:
                    cluster_type = "Low Performance"
            elif 'error_rate' in profile:
                avg_error = profile['error_rate']['mean']
                if avg_error < 0.01:
                    cluster_type = "Reliable"
                elif avg_error < 0.05:
                    cluster_type = "Moderate Reliability"
                else:
                    cluster_type = "Unreliable"
            else:
                cluster_type = f"Cluster {cluster_id}"
            
            cluster_types[cluster_id] = cluster_type
        
        return cluster_types
    
    def detect_performance_trends(self, df: pd.DataFrame, 
                                time_column: str = 'timestamp',
                                metric_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Detect trends in performance metrics over time"""
        logger.info("Detecting performance trends over time")
        
        if time_column not in df.columns:
            logger.warning(f"Time column '{time_column}' not found")
            return {'error': f"Time column '{time_column}' not found"}
        
        # Convert timestamp to datetime if needed
        if df[time_column].dtype == 'object':
            df[time_column] = pd.to_datetime(df[time_column])
        
        # Sort by time
        df_sorted = df.sort_values(time_column)
        
        # Default metrics to analyze
        if metric_columns is None:
            metric_columns = ['avg_response_time', 'requests_per_sec', 'error_rate']
        
        trend_analysis = {}
        
        for metric in metric_columns:
            if metric in df_sorted.columns:
                values = df_sorted[metric].dropna()
                if len(values) > 10:  # Need sufficient data points
                    trend_analysis[metric] = self._analyze_metric_trend(values, df_sorted[time_column])
        
        # Overall trend summary
        overall_trends = self._summarize_overall_trends(trend_analysis)
        
        insights = {
            'trend_analysis': trend_analysis,
            'overall_trends': overall_trends,
            'time_period': {
                'start': df_sorted[time_column].min().isoformat(),
                'end': df_sorted[time_column].max().isoformat(),
                'duration_days': (df_sorted[time_column].max() - df_sorted[time_column].min()).days
            }
        }
        
        logger.info(f"Trend analysis completed for {len(trend_analysis)} metrics")
        
        return insights
    
    def _analyze_metric_trend(self, values: pd.Series, timestamps: pd.Series) -> Dict[str, Any]:
        """Analyze trend for a specific metric"""
        # Calculate trend using linear regression
        x = np.arange(len(values))
        y = values.values
        
        # Simple linear regression
        slope, intercept = np.polyfit(x, y, 1)
        
        # Calculate trend strength (R-squared)
        y_pred = slope * x + intercept
        r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        
        # Determine trend direction and significance
        if abs(slope) < 0.01 * np.std(y):
            trend_direction = "Stable"
        elif slope > 0:
            trend_direction = "Increasing"
        else:
            trend_direction = "Decreasing"
        
        trend_significance = "Significant" if r_squared > 0.3 else "Weak"
        
        return {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_squared),
            'trend_direction': trend_direction,
            'trend_significance': trend_significance,
            'mean_value': float(values.mean()),
            'std_value': float(values.std()),
            'change_percentage': float((slope * len(values)) / values.mean() * 100) if values.mean() != 0 else 0
        }
    
    def _summarize_overall_trends(self, trend_analysis: Dict[str, Dict]) -> Dict[str, Any]:
        """Summarize overall trends across all metrics"""
        summary = {
            'improving_metrics': [],
            'degrading_metrics': [],
            'stable_metrics': [],
            'significant_trends': [],
            'overall_performance_direction': 'Unknown'
        }
        
        for metric, analysis in trend_analysis.items():
            direction = analysis['trend_direction']
            significance = analysis['trend_significance']
            
            if direction == "Increasing":
                if "latency" in metric.lower() or "error" in metric.lower():
                    summary['degrading_metrics'].append(metric)
                else:
                    summary['improving_metrics'].append(metric)
            elif direction == "Decreasing":
                if "latency" in metric.lower() or "error" in metric.lower():
                    summary['improving_metrics'].append(metric)
                else:
                    summary['degrading_metrics'].append(metric)
            else:
                summary['stable_metrics'].append(metric)
            
            if significance == "Significant":
                summary['significant_trends'].append(metric)
        
        # Determine overall direction
        improving_count = len(summary['improving_metrics'])
        degrading_count = len(summary['degrading_metrics'])
        
        if improving_count > degrading_count:
            summary['overall_performance_direction'] = 'Improving'
        elif degrading_count > improving_count:
            summary['overall_performance_direction'] = 'Degrading'
        else:
            summary['overall_performance_direction'] = 'Stable'
        
        return summary

class PerformanceReporter:
    """Generates comprehensive performance reports"""
    
    def __init__(self, insight_generator: PerformanceInsightGenerator):
        self.insight_generator = insight_generator
        self.report_history = []
        
    def generate_comprehensive_report(self, df: pd.DataFrame, models: Dict[str, Any],
                                   report_title: str = "Performance ML Report") -> Dict[str, Any]:
        """Generate a comprehensive performance report"""
        logger.info("Generating comprehensive performance report")
        
        report = {
            'title': report_title,
            'timestamp': datetime.now().isoformat(),
            'data_summary': self._generate_data_summary(df),
            'model_insights': {},
            'performance_analysis': {},
            'recommendations': []
        }
        
        # Generate model insights
        for model_name, model in models.items():
            if hasattr(model, 'predict'):
                # Prepare features for explanation
                feature_cols = [col for col in df.columns if col not in ['test_id', 'timestamp', 'status', 'error']]
                X_sample = df[feature_cols].fillna(df[feature_cols].mean()).iloc[:100]  # Sample for explanation
                
                model_insights = self.insight_generator.generate_model_explanations(
                    model, X_sample, model_name
                )
                report['model_insights'][model_name] = model_insights
        
        # Generate performance analysis
        report['performance_analysis'] = {
            'trends': self.insight_generator.detect_performance_trends(df),
            'clusters': self.insight_generator.cluster_performance_patterns(df)
        }
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        # Store report
        self.report_history.append(report)
        
        logger.info("Comprehensive report generated successfully")
        
        return report
    
    def _generate_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary of the dataset"""
        summary = {
            'total_records': len(df),
            'date_range': {
                'start': df['timestamp'].min().isoformat() if 'timestamp' in df.columns else 'Unknown',
                'end': df['timestamp'].max().isoformat() if 'timestamp' in df.columns else 'Unknown'
            },
            'features': {
                'total': len(df.columns),
                'numerical': len(df.select_dtypes(include=[np.number]).columns),
                'categorical': len(df.select_dtypes(include=['object', 'category']).columns)
            },
            'missing_data': df.isnull().sum().to_dict(),
            'key_metrics': {}
        }
        
        # Calculate key performance metrics
        performance_metrics = ['avg_response_time', 'requests_per_sec', 'error_rate']
        for metric in performance_metrics:
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    summary['key_metrics'][metric] = {
                        'mean': float(values.mean()),
                        'median': float(values.median()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max())
                    }
        
        return summary
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on insights"""
        recommendations = []
        
        # Model insights recommendations
        for model_name, insights in report['model_insights'].items():
            if 'top_features' in insights:
                top_feature = insights['top_features'][0][0] if insights['top_features'] else None
                if top_feature:
                    recommendations.append(f"Focus optimization efforts on '{top_feature}' as it's the most important feature for {model_name}")
        
        # Performance trends recommendations
        if 'trends' in report['performance_analysis']:
            trends = report['performance_analysis']['trends']
            if 'overall_trends' in trends:
                overall_trends = trends['overall_trends']
                
                if overall_trends['overall_performance_direction'] == 'Degrading':
                    recommendations.append("Performance is degrading - investigate recent changes and consider immediate optimization")
                elif overall_trends['overall_performance_direction'] == 'Improving':
                    recommendations.append("Performance is improving - continue current optimization strategies")
                
                if overall_trends['degrading_metrics']:
                    recommendations.append(f"Focus on improving: {', '.join(overall_trends['degrading_metrics'])}")
        
        # Clustering recommendations
        if 'clusters' in report['performance_analysis']:
            clusters = report['performance_analysis']['clusters']
            if 'cluster_types' in clusters:
                cluster_types = clusters['cluster_types']
                low_performance_clusters = [k for k, v in cluster_types.items() if 'Low' in v or 'Unreliable' in v]
                
                if low_performance_clusters:
                    recommendations.append(f"Investigate clusters {low_performance_clusters} which show poor performance patterns")
        
        # General recommendations
        recommendations.extend([
            "Monitor model drift regularly and retrain when necessary",
            "Set up automated alerts for performance anomalies",
            "Consider A/B testing for performance optimizations",
            "Document successful optimization strategies for future reference"
        ])
        
        return recommendations
    
    def export_report(self, report: Dict[str, Any], 
                     format: str = 'json',
                     file_path: Optional[str] = None) -> str:
        """Export report to various formats"""
        logger.info(f"Exporting report in {format} format")
        
        if format == 'json':
            content = json.dumps(report, indent=2)
            extension = 'json'
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if file_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = f"performance_report_{timestamp}.{extension}"
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Report exported to {file_path}")
        return file_path

def main():
    """Example usage of insight generation and reporting"""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'test_id': [f'test_{i}' for i in range(n_samples)],
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'concurrent_users': np.random.randint(10, 500, n_samples),
        'spawn_rate': np.random.uniform(1, 20, n_samples),
        'duration': np.random.randint(300, 1800, n_samples),
        'avg_response_time': np.random.uniform(100, 2000, n_samples),
        'requests_per_sec': np.random.uniform(10, 1000, n_samples),
        'error_rate': np.random.uniform(0, 0.1, n_samples),
        'network_condition_encoded': np.random.randint(0, 4, n_samples)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Create mock model
    from sklearn.ensemble import RandomForestRegressor
    
    X_sample = df[['concurrent_users', 'spawn_rate', 'duration', 'network_condition_encoded']].fillna(0)
    y_sample = df['avg_response_time']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_sample, y_sample)
    
    # Initialize components
    insight_generator = PerformanceInsightGenerator()
    reporter = PerformanceReporter(insight_generator)
    
    # Generate insights
    model_insights = insight_generator.generate_model_explanations(model, X_sample, "latency_model")
    trend_analysis = insight_generator.detect_performance_trends(df)
    cluster_analysis = insight_generator.cluster_performance_patterns(df)
    
    print("Model Insights:")
    print(f"Top features: {model_insights['top_features'][:3]}")
    print(f"Explanation summary: {model_insights['explanation_summary']}")
    
    print("\nTrend Analysis:")
    print(f"Overall direction: {trend_analysis['overall_trends']['overall_performance_direction']}")
    print(f"Significant trends: {trend_analysis['overall_trends']['significant_trends']}")
    
    print("\nCluster Analysis:")
    print(f"Cluster types: {cluster_analysis['cluster_types']}")
    
    # Generate comprehensive report
    models = {'latency_model': model}
    report = reporter.generate_comprehensive_report(df, models, "Performance ML Demo Report")
    
    print("\nReport Generated:")
    print(f"Title: {report['title']}")
    print(f"Recommendations: {len(report['recommendations'])}")
    print(f"Data summary: {report['data_summary']['total_records']} records")

if __name__ == "__main__":
    main()
