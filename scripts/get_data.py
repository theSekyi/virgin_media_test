import io
import logging

import pandas as pd
from google.cloud import storage

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()],
)


def create_storage_client():
    """
    Creates a client to interact with the Google Cloud Storage API.

    Returns:
        storage.Client: An instance of the storage client.
    """
    logging.info("Creating Google Cloud Storage client")
    return storage.Client()


def download_blob_as_string(storage_client, bucket_name, blob_name):
    """
    Downloads a blob from Google Cloud Storage (GCS) and returns its contents as a string.

    Args:
        storage_client (storage.Client): The storage client instance.
        bucket_name (str): The name of the GCS bucket.
        blob_name (str): The name of the blob to download.

    Returns:
        str: The contents of the blob as a string.
    """
    logging.info(f"Downloading blob '{blob_name}' from bucket '{bucket_name}'")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    content = blob.download_as_string()
    logging.info(f"Downloaded blob '{blob_name}' successfully")

    return content


def read_csv_from_string(csv_string):
    """
    Reads a CSV file from a string and returns its contents as a pandas DataFrame.

    Args:
        csv_string (str): The string containing the CSV content.

    Returns:
        pd.DataFrame: The parsed CSV content as a pandas DataFrame.
    """
    logging.info("Reading CSV content from string")
    dataframe = pd.read_csv(io.StringIO(csv_string.decode("utf-8")))
    logging.info("Successfully read CSV content and created DataFrame")

    return dataframe


def load_data(
    bucket_name="cloud-samples-data",
    file_name="ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv",
):
    """
    Main function to demonstrate the usage of the above functions.
    """
    logging.info(f"Loading data from {bucket_name}/{file_name}")

    # bucket_name = "cloud-samples-data"
    # file_name = "ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"

    storage_client = create_storage_client()
    csv_string = download_blob_as_string(storage_client, bucket_name, file_name)
    df = read_csv_from_string(csv_string)

    # Perform any operations on the DataFrame here
    logging.info(f"DataFrame shape: {df.shape}")
    logging.info(f"DataFrame columns: {df.columns}")
    logging.info("DataFrame's content:\n" + str(df.head()))

    logging.info(f"Data has been successfully downloaded from {bucket_name}/{file_name}")
    return df


if __name__ == "__main__":
    load_data()
