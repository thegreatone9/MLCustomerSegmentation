import joblib
import numpy as np
from pathlib import Path


# Segment definitions keyed by (income_level, spending_level)
SEGMENT_DEFINITIONS = {
    ('low', 'low'): {
        'name': 'Careful Customers',
        'description': 'Low Income, Low Spending',
        'strategy': 'Budget-friendly products, discount offers',
        'color': '#2ecc71'
    },
    ('moderate', 'moderate'): {
        'name': 'Standard Customers',
        'description': 'Moderate Income, Moderate Spending',
        'strategy': 'Balanced product range, loyalty programs',
        'color': '#e74c3c'
    },
    ('high', 'high'): {
        'name': 'Target Customers',
        'description': 'High Income, High Spending',
        'strategy': 'Premium products, VIP services',
        'color': '#f39c12'
    },
    ('low', 'high'): {
        'name': 'Careless Customers',
        'description': 'Low Income, High Spending',
        'strategy': 'Installment plans, credit options',
        'color': '#9b59b6'
    },
    ('high', 'low'): {
        'name': 'Sensible Customers',
        'description': 'High Income, Low Spending',
        'strategy': 'Quality emphasis, investment products',
        'color': '#3498db'
    },
}

DEFAULT_SEGMENT = {
    'name': 'Unknown Segment',
    'description': 'Unclassified',
    'strategy': 'Further analysis needed',
    'color': '#95a5a6'
}


def _classify_level(value, low_thresh, high_thresh):
    """Classify a value as low, moderate, or high."""
    if value < low_thresh:
        return 'low'
    elif value > high_thresh:
        return 'high'
    return 'moderate'


def _build_cluster_map(model, scaler):
    """Dynamically map cluster IDs to segment definitions
    based on cluster center positions in original scale.

    This solves the KMeans arbitrary cluster ID problem —
    cluster 0 might be 'Target' in one run and 'Careful' in another.
    """
    centers_original = scaler.inverse_transform(model.cluster_centers_)
    cluster_map = {}

    for cluster_id, (income, spending) in enumerate(centers_original):
        inc_level = _classify_level(income, low_thresh=40, high_thresh=70)
        spend_level = _classify_level(spending, low_thresh=40, high_thresh=60)
        key = (inc_level, spend_level)
        segment = SEGMENT_DEFINITIONS.get(key, DEFAULT_SEGMENT)
        cluster_map[cluster_id] = segment

    return cluster_map


class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        self.scaler = joblib.load(Path('artifacts/data_transformation/scaler.joblib'))
        self.cluster_map = _build_cluster_map(self.model, self.scaler)


    def predict(self, data):
        """Predict customer segment.

        Args:
            data: array-like of shape (1, 2) - [annual_income, spending_score]

        Returns:
            dict: Prediction result with cluster ID and business context.
        """
        data = np.array(data).reshape(1, -1)
        scaled_data = self.scaler.transform(data)
        cluster_id = int(self.model.predict(scaled_data)[0])
        info = self.cluster_map.get(cluster_id, DEFAULT_SEGMENT)

        return {
            'cluster_id': cluster_id,
            'name': info['name'],
            'description': info['description'],
            'strategy': info['strategy'],
            'color': info['color']
        }
