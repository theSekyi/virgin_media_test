from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from scripts import get_data, preprocessing, train
from scripts.config import CONFIG
from scripts.prediction import convert_prob_to_target, load_xgboost_model, predict


# Mock functions and data for testing
def mock_load_data():
    data = {
        "Type": ["Cat", "Cat", "Dog", "Dog", "Dog"],
        "Age": [3, 1, 1, 4, 1],
        "Breed1": ["Tabby", "Domestic Medium Hair", "Mixed Breed", "Mixed Breed", "Mixed Breed"],
        "Gender": ["Male", "Male", "Male", "Female", "Male"],
        "Color1": ["Black", "Black", "Brown", "Black", "Black"],
        "Color2": ["White", "Brown", "White", "Brown", "No Color"],
        "MaturitySize": ["Small", "Medium", "Medium", "Medium", "Medium"],
        "FurLength": ["Short", "Medium", "Medium", "Short", "Short"],
        "Vaccinated": ["No", "Not Sure", "Yes", "Yes", "No"],
        "Sterilized": ["No", "Not Sure", "No", "No", "No"],
        "Health": ["Healthy", "Healthy", "Healthy", "Healthy", "Healthy"],
        "Fee": [100, 0, 0, 150, 0],
        "PhotoAmt": [1, 2, 7, 8, 3],
        "Adopted": ["Yes", "Yes", "Yes", "Yes", "Yes"],
    }
    return pd.DataFrame(data)


def mock_preprocess_dataframe(df, config=CONFIG):
    df = preprocessing.preprocess_dataframe(df, config)
    df = df.drop(columns=[config["target_column"]])
    return df


def mock_predict_with_xgboost_classifier(model, df):
    return np.array([0.8, 0.2, 0.9, 0.1, 0.7])


@pytest.fixture
def model_path():
    return Path("artifacts/model/xgboost_classifier.model")


# Test cases
def test_load_xgboost_model(model_path):
    """
    Test the load_xgboost_model function by checking if it loads the correct model type.
    """
    model = load_xgboost_model(model_path)
    assert isinstance(model, xgb.Booster), "Model should be an instance of xgb.Booster"


def test_convert_prob_to_target():
    """
    Test the convert_prob_to_target function by providing a sample list of probabilities
    and checking if the conversion to target classes is done correctly.
    """
    y_pred = [0.8, 0.3, 0.6, 0.2, 0.9]
    threshold = 0.5
    expected_output = ["Yes", "No", "Yes", "No", "Yes"]
    assert (
        convert_prob_to_target(y_pred, threshold) == expected_output
    ), "Conversion of probabilities to target classes is incorrect"


@pytest.mark.parametrize(
    "y_pred,threshold,expected_output",
    [
        ([0.8, 0.3, 0.6, 0.2, 0.9], 0.5, ["Yes", "No", "Yes", "No", "Yes"]),
        ([0.8, 0.3, 0.6, 0.2, 0.9], 0.7, ["Yes", "No", "No", "No", "Yes"]),
    ],
)
def test_convert_prob_to_target_parametrized(y_pred, threshold, expected_output):
    """
    Test the convert_prob_to_target function using pytest's parametrize feature.
    This allows testing the function with multiple sets of input parameters and expected outputs.
    
    Args:
        y_pred (list): A list of probabilities to be converted.
        threshold (float): The threshold to determine the target class.
        expected_output (list): The expected output after converting probabilities to target classes.
    """
    assert (
        convert_prob_to_target(y_pred, threshold) == expected_output
    ), "Conversion of probabilities to target classes is incorrect"


def test_predict(monkeypatch):
    """
    Test the predict function by mocking its dependencies and checking if the output is as expected.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture used for safely patching functions and attributes.
    """
    monkeypatch.setattr(get_data, "load_data", mock_load_data)
    # monkeypatch.setattr(preprocessing, "preprocess_dataframe", mock_preprocess_dataframe)
    monkeypatch.setattr(train, "predict_with_xgboost_classifier", mock_predict_with_xgboost_classifier)

    df = mock_load_data()

    model_path = Path("artifacts/model/xgboost_classifier.model")
    model = load_xgboost_model(model_path)
    preprocess_df = mock_preprocess_dataframe(df, CONFIG)

    convert_y_pred_to_target = predict(model, preprocess_df)
    expected_output = ["No", "Yes", "Yes", "Yes", "Yes"]

    assert convert_y_pred_to_target == expected_output, "Predict function output is incorrect"

