import json
import os

class MetricsHandler:
    """Handle model metrics loading and processing"""
    
    @staticmethod
    def load_metrics(metrics_file=None):
        """Load pre-computed metrics from JSON"""
        if metrics_file is None:
            # Use absolute path relative to project root
            metrics_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'model_metrics.json')
        
        try:
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    return json.load(f)
        except FileNotFoundError:
            print(f"Metrics file not found at {metrics_file}, using defaults")
        except Exception as e:
            print(f"Error loading metrics: {e}")
        
        # Return default metrics
        return MetricsHandler.get_default_metrics()
    
    @staticmethod
    def get_default_metrics():
        """Return default metrics from your analysis"""
        return {
            'overall_accuracy': 0.9998,
            'macro_f1': 0.9998,
            'weighted_f1': 0.9998,
            'cohen_kappa': 0.9997,
            'macro_auc': 1.0000,
            'per_class_metrics': {
                'MildDemented': {
                    'precision': 1.0000,
                    'recall': 1.0000,
                    'f1': 1.0000,
                    'support': 1019,
                    'auc': 1.0000
                },
                'ModerateDemented': {
                    'precision': 1.0000,
                    'recall': 1.0000,
                    'f1': 1.0000,
                    'support': 1004,
                    'auc': 1.0000
                },
                'NonDemented': {
                    'precision': 1.0000,
                    'recall': 0.9992,
                    'f1': 0.9996,
                    'support': 1298,
                    'auc': 1.0000
                },
                'VeryMildDemented': {
                    'precision': 0.9991,
                    'recall': 1.0000,
                    'f1': 0.9995,
                    'support': 1079,
                    'auc': 1.0000
                }
            },
            'confusion_matrix': [
                [1019, 0, 0, 0],
                [0, 1004, 0, 0],
                [0, 0, 1297, 1],
                [0, 0, 0, 1079]
            ]
        }
