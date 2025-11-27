import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from customerSegmentation import logger
from customerSegmentation.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config


    def transform_data(self):
        """Extract feature columns, apply StandardScaler, and save outputs."""
        data = pd.read_csv(self.config.data_path)

        # Extract the two feature columns used for clustering
        feature_columns = ['Annual Income (k$)', 'Spending Score (1-100)']
        X = data[feature_columns].values

        # Apply StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Save scaled features
        scaled_path = os.path.join(self.config.root_dir, "scaled_features.npy")
        np.save(scaled_path, X_scaled)
        logger.info(f"Scaled features saved to: {scaled_path}")
        logger.info(f"Shape: {X_scaled.shape}")

        # Save the scaler for prediction
        scaler_path = os.path.join(self.config.root_dir, "scaler.joblib")
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to: {scaler_path}")

        # Also save the original data with features for reference
        data.to_csv(os.path.join(self.config.root_dir, "original_data.csv"), index=False)
        logger.info("Original data saved for reference")
