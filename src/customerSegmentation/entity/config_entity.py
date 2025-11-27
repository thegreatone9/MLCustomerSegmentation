from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_data_file: Path
    ingestion_data_file: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    data_path: Path
    STATUS_FILE: str
    all_schema: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_name: str
    n_clusters: int
    init: str
    n_init: int
    max_iter: int
    random_state: int


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
    scaler_path: Path
    metric_file_name: Path
