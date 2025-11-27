import os
import shutil
from customerSegmentation import logger
from customerSegmentation.utils.common import get_size
from pathlib import Path
from customerSegmentation.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    def copy_data(self):
        """Copy the dataset from data/ to the artifacts directory."""
        if not os.path.exists(self.config.ingestion_data_file):
            shutil.copy2(
                src=self.config.source_data_file,
                dst=self.config.ingestion_data_file
            )
            logger.info(f"Copied {self.config.source_data_file} to {self.config.ingestion_data_file}")
        else:
            logger.info(f"File already exists: {get_size(Path(self.config.ingestion_data_file))}")


    def ensure_data_ready(self):
        """Ensure the dataset is available in the artifacts directory."""
        if not os.path.exists(self.config.source_data_file):
            raise FileNotFoundError(
                f"Source data not found at {self.config.source_data_file}. "
                f"Please place Mall_Customers.csv in the data/ directory."
            )
        self.copy_data()
        logger.info(f"Dataset ready at: {self.config.ingestion_data_file}")
