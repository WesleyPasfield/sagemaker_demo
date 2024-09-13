import argparse
import joblib
import os

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


if __name__ == "__main__":
    print("extracting arguments")
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # to simplify the demo we don't use all sklearn RandomForest hyperparameters
    parser.add_argument("--n-estimators", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    parser.add_argument("--max_depth", type=int, default=3)

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="train.csv")
    parser.add_argument("--test-file", type=str, default="test.csv")
    parser.add_argument("--target", type=str)  # in this script we ask user to explicitly name the target

    args, _ = parser.parse_known_args()

    print("reading data")
    # .train location is the on the machine the file is loaded, and .train_file is the file name. Same for test data
    train_df = pd.read_csv(os.path.join(args.train, args.train_file)) 
    test_df = pd.read_csv(os.path.join(args.test, "test.csv"))

    # Get target variable & set up train/test data
    y_train = train_df[args.target]
    X_train = train_df.drop(args.target, axis=1)
    y_test = test_df[args.target]
    X_test = test_df.drop(args.target, axis=1)

    # train the model
    print("training model")
    model = RandomForestRegressor(
        n_estimators=args.n_estimators, 
        min_samples_leaf=args.min_samples_leaf, 
        max_depth=args.max_depth,
        n_jobs=-1,
        verbose=1
    )

    model.fit(X_train, y_train)

    # Compute model stats

    print('Computing model metrics on test set')
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'MSE: {mse}')
    print(f'MAE: {mae}')
    print(f'R2: {r2}')

    # persist model
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    print("model persisted at " + path)