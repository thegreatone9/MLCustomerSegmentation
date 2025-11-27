from customerSegmentation import logger
from customerSegmentation.config.configuration import ConfigurationManager
from customerSegmentation.components.data_ingestion import DataIngestion
from customerSegmentation.components.data_validation import DataValidation
from customerSegmentation.components.data_transformation import DataTransformation
from customerSegmentation.components.model_trainer import ModelTrainer
from customerSegmentation.components.model_evaluation import ModelEvaluation
from pathlib import Path


def run_stage(name, fn):
    """Run a pipeline stage with logging and error handling."""
    try:
        logger.info(f">>>>>> {name} started <<<<<<")
        fn()
        logger.info(f">>>>>> {name} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise


def main():
    config = ConfigurationManager()

    # Stage 1: Data Ingestion
    run_stage("Data Ingestion", lambda: (
        DataIngestion(config.get_data_ingestion_config()).ensure_data_ready()
    ))

    # Stage 2: Data Validation
    run_stage("Data Validation", lambda: (
        DataValidation(config.get_data_validation_config()).validate_all_columns()
    ))

    # Stage 3: Data Transformation (only if validation passed)
    def transform():
        with open(Path("artifacts/data_validation/status.txt"), "r") as f:
            status = f.read().split(" ")[-1]
        if status == "True":
            DataTransformation(config.get_data_transformation_config()).transform_data()
        else:
            raise Exception("Data schema is not valid")

    run_stage("Data Transformation", transform)

    # Stage 4: Model Training
    run_stage("Model Training", lambda: (
        ModelTrainer(config.get_model_trainer_config()).train()
    ))

    # Stage 5: Model Evaluation
    run_stage("Model Evaluation", lambda: (
        ModelEvaluation(config.get_model_evaluation_config()).evaluate()
    ))


if __name__ == "__main__":
    main()
