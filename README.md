This repository represents a solution sample for the problem stated in `test.md`. To run the code, follow the steps highlighted below:

0. Navigate to the the virgin_media_test repo:

1. Create a virtual environment
   `python -m venv .venv`

2. Activate the virtual environment
   `source .venv/bin/activate`

3. Install the required packages in the virtual environment
   `pip install -r requirements.txt`

4. To get data from the GCS bucket, run the following
   `python scripts/get_data.py`

5. To train a model, run
   `python scripts/train.py`

This step will train a model using xgboost and serialize it to <b>artifacts/model/xgboost_classifier.model</b>

6. To ran predictions on using a trained model ran,
   `python scripts/prediction.py`
   This loads the trained model from the directory above and outputs a resulting CSV to `output/results.csv`

7. We also have some tests written in pytest. To run the tests, use the command
   `pytest`
