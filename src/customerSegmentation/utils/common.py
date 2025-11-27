import os
import yaml
import json
from customerSegmentation import logger
from box import ConfigBox
from pathlib import Path


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns its contents as a ConfigBox.

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Raises:
        ValueError: If the YAML file is empty.

    Returns:
        ConfigBox: ConfigBox object with dot-access to YAML contents.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except Exception as e:
        raise e


def create_directories(path_to_directories: list, verbose=True):
    """Create a list of directories.

    Args:
        path_to_directories (list): List of directory paths to create.
        verbose (bool, optional): Log creation. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


def save_json(path: Path, data: dict):
    """Save data as a JSON file.

    Args:
        path (Path): Path to the JSON file.
        data (dict): Data to save.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")


def load_json(path: Path) -> dict:
    """Load data from a JSON file.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        dict: Data loaded from the JSON file.
    """
    with open(path) as f:
        data = json.load(f)

    logger.info(f"json file loaded from: {path}")
    return data


def get_size(path: Path) -> str:
    """Get file size in KB.

    Args:
        path (Path): Path to the file.

    Returns:
        str: Size in KB.
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"
