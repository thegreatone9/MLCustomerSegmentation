import os
from pathlib import Path
from customerSegmentation.utils.common import read_yaml, create_directories
from customerSegmentation.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)


CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")
SCHEMA_FILE_PATH = Path("schema.yaml")


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir=config.root_dir,
            source_data_file=config.source_data_file,
            ingestion_data_file=config.ingestion_data_file
        )


    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS
        create_directories([config.root_dir])

        return DataValidationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            STATUS_FILE=config.STATUS_FILE,
            all_schema=schema,
        )


    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])

        return DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
        )


    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.KMeans
        create_directories([config.root_dir])

        return ModelTrainerConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_name=config.model_name,
            n_clusters=params.n_clusters,
            init=params.init,
            n_init=params.n_init,
            max_iter=params.max_iter,
            random_state=params.random_state
        )


    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_directories([config.root_dir])

        return ModelEvaluationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_path=config.model_path,
            scaler_path=config.scaler_path,
            metric_file_name=config.metric_file_name
        )
