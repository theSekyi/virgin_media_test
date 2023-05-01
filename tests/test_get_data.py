import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import pytest
from google.cloud import storage

from scripts.get_data import create_storage_client, load_data, read_csv_from_string

# Sample data for testing
sample_data = b"""Age,Breed1,Color1,Color2,MaturitySize,FurLength,Vaccinated,Sterilized,Health,Fee,PhotoAmt,Adopted,Type_Cat,Type_Dog,Gender_Female,Gender_Male
3,242,0,5,0.0,0.0,0,0,0.0,100,1,Yes,True,False,False,True
1,865,0,0,1.0,1.0,1,1,0.0,0,2,Yes,True,False,False,True"""


def test_create_storage_client():
    storage_client = create_storage_client()
    assert isinstance(storage_client, storage.Client), "Should return a valid storage.Client instance"


def test_read_csv_from_string():
    df = read_csv_from_string(sample_data)
    assert not df.empty, "Dataframe should not be empty"
    expected_columns = [
        "Age",
        "Breed1",
        "Color1",
        "Color2",
        "MaturitySize",
        "FurLength",
        "Vaccinated",
        "Sterilized",
        "Health",
        "Fee",
        "PhotoAmt",
        "Adopted",
        "Type_Cat",
        "Type_Dog",
        "Gender_Female",
        "Gender_Male",
    ]
    assert list(df.columns) == expected_columns, "Dataframe should have the expected columns"


def test_load_data(mocker):
    # Mock the GCS bucket and upload sample_data to it
    mocker.patch.object(storage, "Client")
    client = storage.Client()
    bucket_name = "cloud-samples-data"
    blob_name = "petfinder-tabular-classification.csv"
    bucket = client.create_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(sample_data)

    # Mock download_blob_as_string to return sample_data
    mocker.patch("scripts.get_data.download_blob_as_string", return_value=sample_data)

    # Test the load_data function
    df = load_data(bucket_name=bucket_name, file_name=blob_name)
    assert not df.empty, "Dataframe should not be empty"
    expected_columns = [
        "Age",
        "Breed1",
        "Color1",
        "Color2",
        "MaturitySize",
        "FurLength",
        "Vaccinated",
        "Sterilized",
        "Health",
        "Fee",
        "PhotoAmt",
        "Adopted",
        "Type_Cat",
        "Type_Dog",
        "Gender_Female",
        "Gender_Male",
    ]
    assert list(df.columns) == expected_columns, "Dataframe should have the expected columns"
