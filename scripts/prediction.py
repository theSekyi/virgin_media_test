import logging
import os
from pathlib import Path

import xgboost as xgb

from scripts.file_utilities import create_output_dir, get_root_dir, save_results
from scripts.get_data import load_data
from scripts.preprocessing import preprocess_dataframe
from scripts.train import predict_with_xgboost_classifier

ADOPTION_THRESHOLD = 0.5

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()],
)


def load_xgboost_model(model_path):
    """
    Load an XGBoost model from the specified path.
    
    Args:
        model_path (str): The path to the saved XGBoost model file.
        
    Returns:
        xgb.Booster: The loaded XGBoost model.
    """
    logging.info(f"Loading XGBoost model from {model_path}")
    model = xgb.Booster()
    model.load_model(model_path)
    logging.info(f"XGBoost model loaded successfully from {model_path}")
    return model


def convert_prob_to_target(y_pred, threshold=ADOPTION_THRESHOLD):
    """
    Convert probabilities to target classes based on the specified threshold.
    
    Args:
        y_pred (list): List of predicted probabilities.
        threshold (float): The threshold value to convert probabilities to target classes.
        
    Returns:
        list: The list of target classes.
    """
    logging.info(f"Converting probabilities to target classes with threshold {threshold}")
    target_classes = ["Yes" if p >= threshold else "No" for p in y_pred]
    logging.info(f"Conversion complete. Total classes: {len(target_classes)}")
    return target_classes


def predict(model, preprocess_df):
    """
    Predict the target classes using the given model and preprocessed data.
    
    Args:
        model (xgb.Booster): The trained XGBoost model.
        preprocess_df (pd.DataFrame): The preprocessed input data.
        
    Returns:
        list: The list of predicted target classes.
    """
    y_pred = predict_with_xgboost_classifier(model, preprocess_df)
    return convert_prob_to_target(y_pred)


def main():
    df = load_data()
    X = df.drop(columns=["Adopted"])

    model_path = Path(__file__).parent / ".." / "artifacts" / "model" / "xgboost_classifier.model"
    model = load_xgboost_model(model_path)

    preprocess_df = preprocess_dataframe(X)

    convert_y_pred_to_target = predict(model, preprocess_df)

    df["Adopted_prediction"] = convert_y_pred_to_target

    logging.info(f"Output csv DataFrame shape: {df.shape}")
    logging.info(f"Output csv DataFrame columns: {df.columns}")
    logging.info("Output csv DataFrame's content:\n" + str(df.head()))

    root_dir = get_root_dir()
    create_output_dir(root_dir)
    output_path = root_dir / "output" / "results.csv"

    save_results(df, output_path)


if __name__ == "__main__":
    main()

