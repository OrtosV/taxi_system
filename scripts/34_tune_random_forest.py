import sys
import time
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
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
REPORTS_DIR = PROJECT_ROOT / "reports" / "tables"

RANDOM_STATE = 42
SEARCH_SAMPLE_FRACTION = 0.10

CATEGORICAL_COLUMNS = [
    "zone_id",
    "Borough",
    "Zone",
    "service_zone",
]

EXCLUDED_COLUMNS = [
    "target_next_hour",
    "datetime",
    "forecast_datetime",
]


# Исходная конфигурация + 10 альтернативных конфигураций
PARAMETER_CONFIGS = [
    # Исходная конфигурация Random Forest
    {
        "n_estimators": 30,
        "max_depth": 12,
        "min_samples_leaf": 10,
        "max_features": 0.5,
        "max_samples": 0.05,
    },
    # Альтернативные конфигурации
    {
        "n_estimators": 30,
        "max_depth": 10,
        "min_samples_leaf": 5,
        "max_features": 0.5,
        "max_samples": 0.05,
    },
    {
        "n_estimators": 30,
        "max_depth": 14,
        "min_samples_leaf": 5,
        "max_features": 0.5,
        "max_samples": 0.05,
    },
    {
        "n_estimators": 30,
        "max_depth": 18,
        "min_samples_leaf": 10,
        "max_features": 0.7,
        "max_samples": 0.05,
    },
    {
        "n_estimators": 40,
        "max_depth": 14,
        "min_samples_leaf": 5,
        "max_features": 0.5,
        "max_samples": 0.1,
    },
    {
        "n_estimators": 40,
        "max_depth": 18,
        "min_samples_leaf": 10,
        "max_features": 0.5,
        "max_samples": 0.1,
    },
    {
        "n_estimators": 30,
        "max_depth": 10,
        "min_samples_leaf": 10,
        "max_features": 0.3,
        "max_samples": 0.1,
    },
    {
        "n_estimators": 40,
        "max_depth": 14,
        "min_samples_leaf": 20,
        "max_features": 0.7,
        "max_samples": 0.1,
    },
    {
        "n_estimators": 50,
        "max_depth": 18,
        "min_samples_leaf": 5,
        "max_features": 0.5,
        "max_samples": 0.2,
    },
    {
        "n_estimators": 30,
        "max_depth": None,
        "min_samples_leaf": 10,
        "max_features": 0.5,
        "max_samples": 0.1,
    },
    {
        "n_estimators": 50,
        "max_depth": 12,
        "min_samples_leaf": 20,
        "max_features": 0.3,
        "max_samples": 0.2,
    },
]


def load_datasets():
    """Загружает обучающую и валидационную выборки."""

    logger.info("Loading modeling datasets...")

    train = pd.read_parquet(DATA_DIR / "train.parquet")
    validation = pd.read_parquet(DATA_DIR / "validation.parquet")

    return train, validation


def prepare_features(train, validation):
    """Разделяет признаки и целевую переменную."""

    feature_columns = [
        column for column in train.columns
        if column not in EXCLUDED_COLUMNS
    ]

    numeric_columns = [
        column for column in feature_columns
        if column not in CATEGORICAL_COLUMNS
    ]

    X_train = train[feature_columns].copy()
    y_train = train["target_next_hour"].copy()

    X_validation = validation[feature_columns].copy()
    y_validation = validation["target_next_hour"].copy()

    for dataset in [X_train, X_validation]:
        for column in CATEGORICAL_COLUMNS:
            dataset[column] = (
                dataset[column]
                .fillna("Unknown")
                .astype(str)
            )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        feature_columns,
        numeric_columns,
    )


def build_preprocessor(numeric_columns):
    """Создаёт преобразователь числовых и категориальных признаков."""

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

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_columns),
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


def run_parameter_search(
    X_search,
    y_search,
    X_validation,
    y_validation,
):
    """Проверяет заданные конфигурации и выбирает лучшую по MAE."""

    results = []

    best_mae = float("inf")
    best_params = None

    for trial_number, params in enumerate(PARAMETER_CONFIGS, start=1):
        logger.info(
            "Starting trial %s/%s",
            trial_number,
            len(PARAMETER_CONFIGS),
        )

        model = RandomForestRegressor(
            **params,
            bootstrap=True,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        start_time = time.perf_counter()

        model.fit(X_search, y_search)

        training_time = time.perf_counter() - start_time

        predictions = model.predict(X_validation)

        metrics = calculate_metrics(
            y_validation.to_numpy(),
            predictions,
        )

        trial_result = {
            "trial": trial_number,
            **params,
            "validation_MAE": metrics["MAE"],
            "validation_RMSE": metrics["RMSE"],
            "validation_WAPE_percent": metrics["WAPE_percent"],
            "validation_R2": metrics["R2"],
            "training_time_seconds": round(training_time, 2),
        }

        results.append(trial_result)

        logger.info(
            "Trial %s completed. Validation MAE: %.5f. Time: %.2f sec.",
            trial_number,
            metrics["MAE"],
            training_time,
        )

        if metrics["MAE"] < best_mae:
            best_mae = metrics["MAE"]
            best_params = params.copy()

    results_df = pd.DataFrame(results)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results_path = REPORTS_DIR / "random_forest_tuning_results_v2.csv"
    results_df.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info("Search results saved to: %s", results_path)

    return best_params, results_df


def main():
    train, validation = load_datasets()

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        feature_columns,
        numeric_columns,
    ) = prepare_features(train, validation)

    logger.info("Training observations: %s", len(X_train))
    logger.info("Validation observations: %s", len(X_validation))

    # Случайная подвыборка только из Train
    search_sample = X_train.sample(
        frac=SEARCH_SAMPLE_FRACTION,
        random_state=RANDOM_STATE,
    )

    y_search = y_train.loc[search_sample.index]

    logger.info(
        "Search sample size: %s (%.0f%% of training data)",
        len(search_sample),
        SEARCH_SAMPLE_FRACTION * 100,
    )

    # Кодирование признаков выполняется один раз
    logger.info("Fitting preprocessing on search sample...")

    search_preprocessor = build_preprocessor(numeric_columns)

    X_search_transformed = search_preprocessor.fit_transform(
        search_sample
    )

    X_validation_transformed = search_preprocessor.transform(
        X_validation
    )

    logger.info("Preprocessing completed.")

    # Поиск гиперпараметров
    logger.info("Starting Random Forest hyperparameter search...")

    best_params, search_results = run_parameter_search(
        X_search_transformed,
        y_search,
        X_validation_transformed,
        y_validation,
    )

    print("\nRandom Forest tuning results:")
    print(
        search_results
        .sort_values("validation_MAE")
        .to_string(index=False)
    )

    print("\nBest parameters:")
    print(best_params)

    # Финальное обучение лучшей конфигурации на всей Train
    logger.info("Preparing final Random Forest model...")

    final_preprocessor = build_preprocessor(numeric_columns)

    X_train_transformed = final_preprocessor.fit_transform(X_train)

    X_validation_final = final_preprocessor.transform(X_validation)

    final_model = RandomForestRegressor(
        **best_params,
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    logger.info("Training best configuration on full training data...")

    final_model.fit(X_train_transformed, y_train)

    logger.info("Final training completed. Calculating metrics...")

    metrics_by_split = {}

    for split_name, X, y in [
        ("train", X_train_transformed, y_train),
        ("validation", X_validation_final, y_validation),
    ]:
        logger.info("Evaluating %s...", split_name)

        predictions = final_model.predict(X)

        metrics_by_split[split_name] = calculate_metrics(
            y.to_numpy(),
            predictions,
        )

    # Объединяем предобработку и модель для сохранения
    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", final_preprocessor),
            ("model", final_model),
        ]
    )

    dataset_info = {
        "train": {
            "observations": len(train),
            "start": str(train["forecast_datetime"].min()),
            "end": str(train["forecast_datetime"].max()),
        },
        "validation": {
            "observations": len(validation),
            "start": str(validation["forecast_datetime"].min()),
            "end": str(validation["forecast_datetime"].max()),
        },
    }

    params_to_save = {
        **best_params,
        "bootstrap": True,
        "n_jobs": -1,
        "search_trials": len(PARAMETER_CONFIGS),
        "search_sample_fraction": SEARCH_SAMPLE_FRACTION,
        "selection_metric": "MAE",
        "random_state": RANDOM_STATE,
        "search_method": "manual_random_search",
        "baseline_included": True,
        "max_samples_type": "fraction",
    }

    logger.info("Saving tuned Random Forest experiment...")

    experiment_id = save_experiment(
        model=final_pipeline,
        model_name="Random Forest Tuned",
        params=params_to_save,
        metrics_by_split=metrics_by_split,
        features=feature_columns,
        dataset_info=dataset_info,
        random_state=RANDOM_STATE,
    )

    results = []

    for split_name, metrics in metrics_by_split.items():
        results.append(
            {
                "model": "Random Forest Tuned",
                "dataset": split_name,
                "observations": dataset_info[split_name]["observations"],
                **metrics,
            }
        )

    results_df = pd.DataFrame(results)

    print("\nTuned Random Forest results:")
    print(results_df.to_string(index=False))

    print(f"\nExperiment ID: {experiment_id}")

    logger.info("Random Forest tuning completed successfully.")


if __name__ == "__main__":
    main()