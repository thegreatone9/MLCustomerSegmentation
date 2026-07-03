import os
import numpy as np
import joblib
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from customerSegmentation import logger
from customerSegmentation.entity.config_entity import ModelEvaluationConfig
from customerSegmentation.utils.common import save_json
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config


    def evaluate(self):
        """Evaluate the trained model and save metrics to JSON."""
        # Load model and data
        model = joblib.load(self.config.model_path)
        X_scaled = np.load(self.config.data_path)

        # Get cluster labels
        labels = model.predict(X_scaled)

        # Calculate metrics
        silhouette = silhouette_score(X_scaled, labels)
        calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
        davies_bouldin = davies_bouldin_score(X_scaled, labels)
        inertia = model.inertia_

        logger.info(f"Silhouette Score: {silhouette:.4f}")
        logger.info(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")
        logger.info(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
        logger.info(f"Inertia: {inertia:.4f}")

        # Save metrics
        metrics = {
            "silhouette_score": round(silhouette, 4),
            "calinski_harabasz_score": round(calinski_harabasz, 4),
            "davies_bouldin_index": round(davies_bouldin, 4),
            "inertia": round(inertia, 4),
            "n_clusters": model.n_clusters
        }

        save_json(path=Path(self.config.metric_file_name), data=metrics)
        logger.info("Model evaluation completed successfully")
