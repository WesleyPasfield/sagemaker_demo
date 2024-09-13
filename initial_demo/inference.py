import argparse
import joblib
import os
import boto3
import tarfile
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    print("extracting arguments")
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # to simplify the demo we don't use all sklearn RandomForest hyperparameters

    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN")) # Using to store model
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--test-file", type=str, default="test.csv")
    parser.add_argument("--model-file-name", type=str, default="model.joblib")
    parser.add_argument("--target", type=str)  # in this script we ask user to explicitly name the target, can spec in hyperparameters

    args, _ = parser.parse_known_args()

    print("reading data")
    # .test location is the on the machine the file is loaded, and .test_file is the file name.
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    # Get target variable & set up test data
    y_test = test_df[args.target]
    X_test = test_df.drop(args.target, axis=1)

    # Load model & compute predictions
    # Train houses model in this case - trained model.
    model = joblib.load(tarfile.open(os.path.join(args.train, args.model_file_name)))
    predictions = model.predict(X_test)

    # Save predictions

    path = os.path.join(args.output_data_dir, "predictions.csv")
    predictions.to_csv(path)
    print("predictions persisted at " + path)



