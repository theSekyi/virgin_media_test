import sys
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

import logging

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

from scripts.config import CONFIG
from scripts.file_utilities import get_root_dir
from scripts.get_data import load_data
from scripts.preprocessing import convert_target_to_binary, preprocess_dataframe

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()],
)


def split_dataset(X, y):
    """
    Splits the dataset into training, validation, and test sets.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.

    Returns:
        tuple: The train, validation, and test feature matrices and target vectors.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # assert X_val.shape == X_test.shape, "X_val and X_test should have the same shape"
    # logging.info("Shape of validation and test sets are the same as expected.")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_xgboost_classifier(X_train, y_train, X_val, y_val, params=None):
    """
    Trains an XGBoost classifier on the given training set.

    Args:
        X_train (pd.DataFrame): The training feature matrix.
        y_train (pd.Series): The training target vector.
        X_val (pd.DataFrame): The validation feature matrix.
        y_val (pd.Series): The validation target vector.
        params (dict, optional): The parameters for the XGBoost classifier. Defaults to None.

    Returns:
        xgb.Booster: The trained XGBoost classifier.
    """
    logging.info("Starting training...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    xgb_params = params or {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.1,
        "max_depth": 6,
        "seed": 42,
    }

    watchlist = [(dtrain, "train"), (dval, "validation")]
    num_rounds = 1000

    model = xgb.train(xgb_params, dtrain, num_rounds, watchlist, early_stopping_rounds=10)
    logging.info("Training complete.")
    return model


def predict_with_xgboost_classifier(model, X):
    """
    Predicts the target using the given XGBoost classifier.

    Args:
        model (xgb.Booster): The trained XGBoost classifier.
        X (pd.DataFrame): The feature matrix.

    Returns:
        np.array: The predicted target values.
    """
    logging.info("Starting prediction...")
    dtest = xgb.DMatrix(X, enable_categorical=True)
    y_pred = model.predict(dtest)
    logging.info("Prediction complete.")
    return y_pred


def convert_probabilities_to_binary(y_pred, threshold=0.5):
    """
    Converts the predicted probabilities to binary predictions using the given threshold.

    Args:
        y_pred (List[float]): The list of predicted probabilities.
        threshold (float, optional): The threshold value to be used for converting probabilities to binary predictions. Defaults to 0.5.

    Returns:
        List[int]: The list of binary predictions.
    """
    return [1 if p >= threshold else 0 for p in y_pred]


def evaluate_xgboost_classifier(model, X_test, y_test):
    """
    Evaluates the XGBoost classifier on the test set.

    Args:
        model (xgb.Booster): The trained XGBoost classifier.
        X_test (pd.DataFrame): The test feature matrix.
        y_test (pd.Series): The test target vector.

    Returns:
        tuple: The accuracy, F1 score, and recall of the classifier.
    """
    logging.info("Starting evaluation...")
    y_pred = predict_with_xgboost_classifier(model, X_test)
    y_pred_binary = convert_probabilities_to_binary(y_pred)

    accuracy = accuracy_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)

    logging.info("Evaluation complete.")
    return accuracy, f1, recall


def save_model(model, model_dir="artifacts/model", model_filename="xgboost_classifier.model"):
    """
    Saves the trained model to the specified directory.

    Args:
        model (xgb.Booster): The trained XGBoost classifier.
        model_dir (str, optional): The directory where the model will be saved. Defaults to "artifacts/model".
        model_filename (str, optional): The filename for the saved model. Defaults to "xgboost_classifier.model".

    Returns:
        None
    """
    root_dir = get_root_dir()
    model_path = root_dir / model_dir
    model_path.mkdir(parents=True, exist_ok=True)

    model_file_path = model_path / model_filename
    model.save_model(str(model_file_path))
    logging.info(f"Model saved to {model_file_path}")


def train_and_evaluate(df):
    """
    Trains and evaluates an XGBoost classifier on the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the feature matrix and target vector.

    Returns:
        None
    """
    if "Adopted" not in df.columns:
        raise ValueError("The 'Adopted' column is missing in the input DataFrame")

    X = df.drop(columns=["Adopted"])
    y = df["Adopted"]

    try:
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
        classifier_model = train_xgboost_classifier(X_train, y_train, X_val, y_val)
        accuracy, f1, recall = evaluate_xgboost_classifier(classifier_model, X_test, y_test)

        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"F1 Score: {f1}")
        logging.info(f"Recall: {recall}")

        save_model(classifier_model)

    except Exception as e:
        logging.error(f"An error occurred during the training and evaluation process: {e}")


if __name__ == "__main__":
    df = load_data()
    df_target_encoded = convert_target_to_binary(df, CONFIG["target_column"])
    df_preprocessed = preprocess_dataframe(df_target_encoded, CONFIG)
    train_and_evaluate(df_preprocessed)

