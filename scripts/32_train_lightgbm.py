import sys
from pathlib import Path

import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import calculate_metrics
from src.experiments.registry import save_experiment
from src.logging_config import setup_logging


logger = setup_logging()

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "modeling"


def load_datasets():
    """Загружает подготовленные обучающую, валидационную и тестовую выборки."""

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
    """Создаёт Pipeline с предобработкой и LightGBM."""

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

    lightgbm = LGBMRegressor(
        objective="regression",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=12,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", lightgbm),
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

    logger.info("Building LightGBM pipeline...")

    model = build_model(numeric_columns, categorical_columns)

    logger.info("Starting LightGBM training...")
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
        "learning_rate": model_params["learning_rate"],
        "max_depth": model_params["max_depth"],
        "num_leaves": model_params["num_leaves"],
        "subsample": model_params["subsample"],
        "colsample_bytree": model_params["colsample_bytree"],
        "reg_lambda": model_params["reg_lambda"],
        "objective": model_params["objective"],
        "n_jobs": model_params["n_jobs"],
    }

    logger.info("Saving experiment and trained model...")

    experiment_id = save_experiment(
        model=model,
        model_name="LightGBM",
        params=params_to_save,
        metrics_by_split=metrics_by_split,
        features=feature_columns,
        dataset_info=dataset_info,
        random_state=42,
    )

    results = []

    for split_name, metrics in metrics_by_split.items():
        row = {
            "model": "LightGBM",
            "dataset": split_name,
            "observations": dataset_info[split_name]["observations"],
            **metrics,
        }
        results.append(row)

    results_df = pd.DataFrame(results)

    print("\nLightGBM results:")
    print(results_df.to_string(index=False))

    print(f"\nExperiment ID: {experiment_id}")

    logger.info("LightGBM experiment completed successfully.")


if __name__ == "__main__":
    main()