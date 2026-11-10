import sys
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import calculate_metrics
from src.experiments.registry import save_experiment
from src.logging_config import setup_logging


logger = setup_logging()

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "modeling"


def load_datasets():
    """Загружает обучающую, валидационную и тестовую выборки."""

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

    datasets = []

    for dataset in [train, validation, test]:
        X = dataset[feature_columns].copy()
        y = dataset[target_column].copy()

        # CatBoost ожидает категориальные значения в строковом или
        # целочисленном формате, но не в виде NaN.
        for column in categorical_columns:
            X[column] = X[column].fillna("Unknown").astype(str)

        datasets.append((X, y))

    return datasets, feature_columns, categorical_columns


def build_model(categorical_columns):
    """Создаёт CatBoost Regressor с нативной обработкой категорий."""

    model = CatBoostRegressor(
        iterations=300,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        l2_leaf_reg=3.0,
        random_strength=1.0,
        random_seed=42,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
        cat_features=categorical_columns,
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
        categorical_columns,
    ) = prepare_features(train, validation, test)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = datasets

    logger.info("Building CatBoost model...")
    model = build_model(categorical_columns)

    logger.info("Starting CatBoost training...")
    logger.info("Training observations: %s", len(X_train))
    logger.info("Number of features: %s", len(feature_columns))
    logger.info("Categorical features: %s", categorical_columns)

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

    model_params = model.get_params()

    params_to_save = {
        "iterations": model_params["iterations"],
        "depth": model_params["depth"],
        "learning_rate": model_params["learning_rate"],
        "loss_function": model_params["loss_function"],
        "l2_leaf_reg": model_params["l2_leaf_reg"],
        "random_strength": model_params["random_strength"],
        "thread_count": -1,
        "categorical_features": categorical_columns,
    }

    logger.info("Saving experiment and trained model...")

    experiment_id = save_experiment(
        model=model,
        model_name="CatBoost",
        params=params_to_save,
        metrics_by_split=metrics_by_split,
        features=feature_columns,
        dataset_info=dataset_info,
        random_state=42,
    )

    results = []

    for split_name, metrics in metrics_by_split.items():
        row = {
            "model": "CatBoost",
            "dataset": split_name,
            "observations": dataset_info[split_name]["observations"],
            **metrics,
        }
        results.append(row)

    results_df = pd.DataFrame(results)

    print("\nCatBoost results:")
    print(results_df.to_string(index=False))

    print(f"\nExperiment ID: {experiment_id}")

    logger.info("CatBoost experiment completed successfully.")


if __name__ == "__main__":
    main()