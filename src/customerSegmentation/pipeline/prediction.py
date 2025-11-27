import joblib
import numpy as np
from pathlib import Path


# Cluster information for business context
CLUSTER_INFO = {
    0: {
        'name': 'Careful Customers',
        'description': 'Low Income, Low Spending',
        'strategy': 'Budget-friendly products, discount offers',
        'color': '#2ecc71'
    },
    1: {
        'name': 'Standard Customers',
        'description': 'Moderate Income, Moderate Spending',
        'strategy': 'Balanced product range, loyalty programs',
        'color': '#e74c3c'
    },
    2: {
        'name': 'Target Customers',
        'description': 'High Income, High Spending',
        'strategy': 'Premium products, VIP services',
        'color': '#f39c12'
    },
    3: {
        'name': 'Careless Customers',
        'description': 'Low Income, High Spending',
        'strategy': 'Installment plans, credit options',
        'color': '#9b59b6'
    },
    4: {
        'name': 'Sensible Customers',
        'description': 'High Income, Low Spending',
        'strategy': 'Quality emphasis, investment products',
        'color': '#3498db'
    }
}


class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        self.scaler = joblib.load(Path('artifacts/data_transformation/scaler.joblib'))


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
        info = CLUSTER_INFO.get(cluster_id, {
            'name': f'Cluster {cluster_id}',
            'description': 'Unknown segment',
            'strategy': 'Further analysis needed',
            'color': '#95a5a6'
        })

        return {
            'cluster_id': cluster_id,
            'name': info['name'],
            'description': info['description'],
            'strategy': info['strategy'],
            'color': info['color']
        }
