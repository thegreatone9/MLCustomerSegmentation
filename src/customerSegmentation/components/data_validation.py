import os
import pandas as pd
from customerSegmentation import logger
from customerSegmentation.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config


    def validate_all_columns(self) -> bool:
        """Validate that all columns in the dataset match the expected schema.

        Returns:
            bool: True if all columns are valid, False otherwise.
        """
        try:
            validation_status = None

            data = pd.read_csv(self.config.data_path)
            all_cols = list(data.columns)
            all_schema = self.config.all_schema.keys()

            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                    logger.info(f"Column '{col}' not found in schema")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

            logger.info(f"Data validation completed. Status: {validation_status}")
            return validation_status

        except Exception as e:
            raise e
