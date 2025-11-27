import os
import numpy as np
import joblib
from sklearn.cluster import KMeans
from customerSegmentation import logger
from customerSegmentation.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    def train(self):
        """Train KMeans model on scaled features."""
        # Load scaled features
        X_scaled = np.load(self.config.data_path)
        logger.info(f"Loaded scaled features with shape: {X_scaled.shape}")

        # Train KMeans
        kmeans = KMeans(
            n_clusters=self.config.n_clusters,
            init=self.config.init,
            n_init=self.config.n_init,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state
        )
        kmeans.fit(X_scaled)
        logger.info(f"KMeans trained with {self.config.n_clusters} clusters")
        logger.info(f"Inertia: {kmeans.inertia_:.2f}")

        # Save model
        model_path = os.path.join(self.config.root_dir, self.config.model_name)
        joblib.dump(kmeans, model_path)
        logger.info(f"Model saved to: {model_path}")
