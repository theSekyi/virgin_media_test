import logging

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from scripts.config import CONFIG
from scripts.get_data import load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def one_hot_encode_columns(df, columns_to_be_one_hot_encoded):
    """
    One-hot encode the specified columns in the DataFrame.

    Args:
        df: The input DataFrame.
        columns_to_be_one_hot_encoded: A list of column names to be one-hot encoded.

    Returns:
        pd.DataFrame: The DataFrame with specified columns one-hot encoded.
    """
    logging.info("One-hot encoding columns: %s", columns_to_be_one_hot_encoded)
    for column in columns_to_be_one_hot_encoded:
        df = pd.get_dummies(df, columns=[column], prefix=column)
    return df


def label_encode_columns(df, columns_to_be_label_encoded):
    """
    Label encode the specified columns in the DataFrame.

    Args:
        df: The input DataFrame.
        columns_to_be_label_encoded: A list of column names to be label encoded.

    Returns:
        pd.DataFrame: The DataFrame with specified columns label encoded.
    """
    logging.info("Label encoding columns: %s", columns_to_be_label_encoded)
    for column in columns_to_be_label_encoded:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
    return df


def ordinally_encode_columns(df, columns_to_be_ordinally_encoded):
    """
    Ordinally encode the specified columns in the DataFrame.

    Args:
        df: The input DataFrame.
        columns_to_be_ordinally_encoded: A dictionary where the key is the column name and the value is a list of ordered values for that column.

    Returns:
        pd.DataFrame: The DataFrame with specified columns ordinally encoded.
    """
    logging.info("Ordinally encoding columns: %s", columns_to_be_ordinally_encoded.keys())
    for column, ordered_values in columns_to_be_ordinally_encoded.items():
        encoder = OrdinalEncoder(categories=[ordered_values])
        df[column] = encoder.fit_transform(df[[column]])
    return df


def count_encode_column(df, col="Breed1"):
    """
    Count encode the specified column in the DataFrame.

    Args:
        df: The input DataFrame.
        col: The column name to be count encoded. Default is 'Breed1'.

    Returns:
        pd.DataFrame: The DataFrame with the specified column count encoded.
    """
    logging.info("Count encoding column: %s", col)
    df[col] = df[col].map(df[col].value_counts())
    return df


def convert_target_to_binary(df, target_col="Adopted"):
    """
    Convert the target column to binary values.

    Args:
        df: The input DataFrame.
        target_col: The target column name. Default is 'Adopted'.

    Returns:
        pd.DataFrame: The DataFrame with the target column converted to binary values.
    """
    logging.info("Converting target column '%s' to binary values", target_col)
    df[target_col] = df[target_col].replace({"Yes": 1, "No": 0})
    logging.info("DataFrame's content:\n" + str(df.head()))
    return df


def preprocess_dataframe(df, config=CONFIG):
    """
    Preprocess the input DataFrame using the specified configuration.

    Args:
        df: The input DataFrame.
        config: A dictionary containing the configuration for preprocessing. This should include the following keys:
                - 'one_hot_encode_columns': list of columns to be one-hot encoded
                - 'label_encode_columns': list of columns to be label encoded
                - 'ordinal_encode_columns': dict with column names as keys and lists of ordered values as values
                - 'count_encode_column': name of the column to be count encoded (default: 'Breed1')
                - 'target_column': name of the target column to convert to binary values (default: 'Adopted')

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    logging.info("Starting preprocessing of the DataFrame")
    df = one_hot_encode_columns(df, config["one_hot_encode_columns"])
    df = label_encode_columns(df, config["label_encode_columns"])
    df = ordinally_encode_columns(df, config["ordinal_encode_columns"])
    df = count_encode_column(df, config["count_encode_column"])

    logging.info(f"DataFrame shape: {df.shape}")
    logging.info(f"DataFrame columns: {df.columns}")
    logging.info("DataFrame's content:\n" + str(df.head()))
    logging.info("Finished preprocessing of the DataFrame")
    return df


if __name__ == "__main__":
    df = load_data()
    preprocessed_df = preprocess_dataframe(df, CONFIG)
