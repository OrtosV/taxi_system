import sys
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import calculate_metrics
from src.experiments.registry import save_experiment
from src.logging_config import setup_logging


logger = setup_logging()

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "modeling"


def load_datasets():
    """Загружает подготовленные выборки."""

    logger.info("Loading modeling datasets...")

    train = pd.read_parquet(DATA_DIR / "train.parquet")
    validation = pd.read_parquet(DATA_DIR / "validation.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")

    return train, validation, test


def prepare_features(train, validation, test):
    """Разделяет признаки и целевую переменную."""

    target_column = "target_next_hour"

    excluded_columns = [
        target_column,
        "datetime",
        "forecast_datetime",
    ]

    feature_columns = [
        column for column in train.columns
        if column not in excluded_columns
    ]

    categorical_columns = [
        "zone_id",
        "Borough",
        "Zone",
        "service_zone",
    ]

    numeric_columns = [
        column for column in feature_columns
        if column not in categorical_columns
    ]

    datasets = []

    for dataset in [train, validation, test]:
        X = dataset[feature_columns].copy()
        y = dataset[target_column].copy()

        for column in categorical_columns:
            X[column] = X[column].fillna("Unknown").astype(str)

        datasets.append((X, y))

    return datasets, feature_columns, numeric_columns, categorical_columns


def build_model(numeric_columns, categorical_columns):
    """Создаёт Pipeline с предобработкой и XGBoost."""

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_columns),
            ("categorical", categorical_transformer, categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    xgboost = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", xgboost),
        ]
    )

    return model


def get_dataset_info(train, validation, test):
    """Формирует информацию о размерах и временных границах выборок."""

    result = {}

    for name, dataset in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        result[name] = {
            "observations": len(dataset),
            "start": str(dataset["forecast_datetime"].min()),
            "end": str(dataset["forecast_datetime"].max()),
        }

    return result


def main():
    train, validation, test = load_datasets()

    (
        datasets,
        feature_columns,
        numeric_columns,
        categorical_columns,
    ) = prepare_features(train, validation, test)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = datasets

    logger.info("Building XGBoost pipeline...")

    model = build_model(numeric_columns, categorical_columns)

    logger.info("Starting XGBoost training...")
    logger.info("Training observations: %s", len(X_train))
    logger.info("Number of features before encoding: %s", len(feature_columns))

    model.fit(X_train, y_train)

    logger.info("Training completed. Calculating metrics...")

    metrics_by_split = {}

    for split_name, X, y in [
        ("train", X_train, y_train),
        ("validation", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        logger.info("Evaluating %s...", split_name)

        predictions = model.predict(X)

        metrics_by_split[split_name] = calculate_metrics(
            y.to_numpy(),
            predictions,
        )

    dataset_info = get_dataset_info(train, validation, test)

    model_params = model.named_steps["model"].get_params()

    params_to_save = {
        "n_estimators": model_params["n_estimators"],
        "max_depth": model_params["max_depth"],
        "learning_rate": model_params["learning_rate"],
        "subsample": model_params["subsample"],
        "colsample_bytree": model_params["colsample_bytree"],
        "reg_lambda": model_params["reg_lambda"],
        "tree_method": model_params["tree_method"],
        "objective": model_params["objective"],
        "n_jobs": model_params["n_jobs"],
    }

    logger.info("Saving experiment and trained model...")

    experiment_id = save_experiment(
        model=model,
        model_name="XGBoost",
        params=params_to_save,
        metrics_by_split=metrics_by_split,
        features=feature_columns,
        dataset_info=dataset_info,
        random_state=42,
    )

    results = []

    for split_name, metrics in metrics_by_split.items():
        row = {
            "model": "XGBoost",
            "dataset": split_name,
            "observations": dataset_info[split_name]["observations"],
            **metrics,
        }
        results.append(row)

    results_df = pd.DataFrame(results)

    print("\nXGBoost results:")
    print(results_df.to_string(index=False))

    print(f"\nExperiment ID: {experiment_id}")

    logger.info("XGBoost experiment completed successfully.")


if __name__ == "__main__":
    main()