import csv
import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from build_init_param_dataset import FEATURE_COLUMNS


def load_dataset(dataset_path: str):
    features = []
    targets = []

    with open(dataset_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            features.append([float(row[column]) for column in FEATURE_COLUMNS])
            targets.append(float(row["theta_target"]))

    return np.array(features, dtype=float), np.array(targets, dtype=float)


def train_model(results_dir: str) -> str:
    dataset_path = os.environ.get(
        "INIT_PARAM_DATASET",
        os.path.join(results_dir, "init_param_dataset.csv"),
    )
    model_path = os.environ.get(
        "INIT_PARAM_MODEL",
        os.path.join(results_dir, "init_param_model.pkl"),
    )

    x, y = load_dataset(dataset_path)
    if len(y) < 10:
        raise ValueError(f"Not enough rows to train a model: {len(y)}")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=0,
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=0,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    payload = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": {
            "rows": int(len(y)),
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "mae": float(mae),
            "mse": float(mse),
            "r2": float(r2),
        },
    }

    with open(model_path, "wb") as model_file:
        pickle.dump(payload, model_file)

    print(f"Wrote model to {model_path}")
    print("Metrics:")
    for key, value in payload["metrics"].items():
        print(f"  {key}: {value}")
    return model_path


if __name__ == "__main__":
    default_results_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "../results/init_baseline")
    )
    results_dir = os.environ.get("RESULTS_DIR", default_results_dir)
    train_model(results_dir)
