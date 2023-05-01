import logging
import os
from pathlib import Path

import pandas as pd


def get_root_dir():
    """
    Get the absolute path to the virgin_media_test directory.

    Returns:
        pathlib.Path: The path to the virgin_media_test directory.
    """
    script_dir = Path(os.path.abspath(__file__)).parent
    virgin_media_test_dir = script_dir.parent
    return virgin_media_test_dir.resolve()


def create_output_dir(root_dir):
    """
    Create the 'output' directory in the root directory if it doesn't exist.

    Args:
        root_dir (pathlib.Path): The path to the root directory.

    Returns:
        pathlib.Path: The path to the created output directory.
    """
    output_dir = root_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_results(data, output_path):
    """
    Save the given data to a CSV file at the specified output path.

    Args:
        data (pandas.DataFrame): The data to be saved.
        output_path (pathlib.Path): The path to the output file.
    """
    data.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


# def load_data(filepath):
#     """
#     Load data from the specified file path.

#     Args:
#         filepath (str): The file path to the data file.

#     Returns:
#         pandas.DataFrame: The loaded data as a DataFrame.
#     """
#     return pd.read_csv(filepath)
