
import sys
import time
from pathlib import Path

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.logging_config import setup_logging
from src.evaluation.metrics import calculate_metrics
from src.experiments.registry import save_experiment


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RANDOM_STATE = 42
SEARCH_SAMPLE_FRACTION = 0.10

DATA_DIR = ROOT / "data" / "processed" / "modeling"
REPORTS_DIR = ROOT / "reports" / "tables"

TRAIN_PATH = DATA_DIR / "train.parquet"
VALIDATION_PATH = DATA_DIR / "validation.parquet"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CATEGORICAL_FEATURES = [
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

BASE_PARAMS = {
    "objective": "regression",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
}

# Исходная конфигурация + 9 альтернативных
PARAMETER_CONFIGS = [
    # 1. Исходная конфигурация LightGBM
    # subsample_freq=0 сохраняет исходное поведение:
    # subsample фактически не применяется.
    {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 12,
        "num_leaves": 31,
        "subsample": 0.8,
        "subsample_freq": 0,
        "colsample_bytree": 0.8,
        "reg_lambda": 1,
        "min_child_samples": 20,
    },
    # 2. Меньшая глубина и меньше листьев
    {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 8,
        "num_leaves": 31,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1,
        "min_child_samples": 20,
    },
    # 3. Больше листьев
    {
        "n_estimators": 250,
        "learning_rate": 0.05,
        "max_depth": 12,
        "num_leaves": 63,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1,
        "min_child_samples": 20,
    },
    # 4. Более глубокие деревья и регуляризация
    {
        "n_estimators": 250,
        "learning_rate": 0.03,
        "max_depth": 15,
        "num_leaves": 63,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 3,
        "min_child_samples": 30,
    },
    # 5. Больше деревьев и листьев
    {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 12,
        "num_leaves": 63,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_lambda": 3,
        "min_child_samples": 20,
    },
    # 6. Более сильная регуляризация
    {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 10,
        "num_leaves": 31,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 5,
        "min_child_samples": 50,
    },
    # 7. Низкая скорость обучения
    {
        "n_estimators": 350,
        "learning_rate": 0.03,
        "max_depth": 12,
        "num_leaves": 31,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 3,
        "min_child_samples": 30,
    },
    # 8. Больше листьев и объектов в листе
    {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 15,
        "num_leaves": 63,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_lambda": 5,
        "min_child_samples": 50,
    },
    # 9. Более высокая скорость обучения
    {
        "n_estimators": 250,
        "learning_rate": 0.07,
        "max_depth": 10,
        "num_leaves": 31,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_lambda": 1,
        "min_child_samples": 20,
    },
    # 10. Сложная модель с дополнительной регуляризацией
    {
        "n_estimators": 350,
        "learning_rate": 0.05,
        "max_depth": 15,
        "num_leaves": 63,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 5,
        "min_child_samples": 30,
    },
]


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

def build_preprocessor(numeric_features):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
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
            ("num", numeric_transformer, numeric_features),
            (
                "cat",
                categorical_transformer,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------
# Model creation
# ---------------------------------------------------------

def build_model(params):
    model_params = {
        **BASE_PARAMS,
        **params,
    }

    return LGBMRegressor(**model_params)


# ---------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------

def run_parameter_search(
    X_search,
    y_search,
    X_validation,
    y_validation,
    preprocessor,
    logger,
):
    results = []

    logger.info("Starting LightGBM hyperparameter search...")

    X_search_transformed = preprocessor.transform(X_search)
    X_validation_transformed = preprocessor.transform(X_validation)

    for trial_number, params in enumerate(PARAMETER_CONFIGS, start=1):
        logger.info(
            "Starting trial %s/%s",
            trial_number,
            len(PARAMETER_CONFIGS),
        )

        start_time = time.time()

        model = build_model(params)

        model.fit(
            X_search_transformed,
            y_search,
        )

        predictions = model.predict(X_validation_transformed)

        metrics = calculate_metrics(
            y_validation,
            predictions,
        )

        training_time = time.time() - start_time

        result = {
            "trial": trial_number,
            **params,
            "validation_MAE": metrics["MAE"],
            "validation_MSE": metrics["MSE"],
            "validation_RMSE": metrics["RMSE"],
            "validation_WAPE_percent": metrics["WAPE_percent"],
            "validation_R2": metrics["R2"],
            "training_time_seconds": training_time,
        }

        results.append(result)

        logger.info(
            "Trial %s completed. Validation MAE: %.5f. Time: %.2f sec.",
            trial_number,
            metrics["MAE"],
            training_time,
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "validation_MAE",
        ascending=True,
    )

    results_path = REPORTS_DIR / "lightgbm_tuning_results.csv"
    results_df.to_csv(results_path, index=False)

    logger.info("Search results saved to: %s", results_path)

    return results_df


# ---------------------------------------------------------
# Final training and evaluation
# ---------------------------------------------------------

def main():
    logger = setup_logging()

    logger.info("Loading modeling datasets...")

    train = pd.read_parquet(TRAIN_PATH)
    validation = pd.read_parquet(VALIDATION_PATH)

    logger.info("Training observations: %s", len(train))
    logger.info("Validation observations: %s", len(validation))

    feature_columns = [
        column
        for column in train.columns
        if column not in EXCLUDED_COLUMNS
    ]

    X_train = train[feature_columns].copy()
    y_train = train["target_next_hour"].copy()

    X_validation = validation[feature_columns].copy()
    y_validation = validation["target_next_hour"].copy()

    numeric_features = [
        column
        for column in feature_columns
        if column not in CATEGORICAL_FEATURES
    ]

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

    logger.info("Fitting preprocessing on search sample...")

    search_preprocessor = build_preprocessor(numeric_features)
    search_preprocessor.fit(search_sample)

    logger.info("Preprocessing completed.")

    results_df = run_parameter_search(
        X_search=search_sample,
        y_search=y_search,
        X_validation=X_validation,
        y_validation=y_validation,
        preprocessor=search_preprocessor,
        logger=logger,
    )

    print("\nLightGBM tuning results:")
    print(results_df.to_string(index=False))

    best_trial = results_df.iloc[0]

    best_params = {
        key: best_trial[key]
        for key in PARAMETER_CONFIGS[0].keys()
    }

    # Восстанавливаем типы параметров после чтения из DataFrame
    integer_params = [
        "n_estimators",
        "max_depth",
        "num_leaves",
        "subsample_freq",
        "min_child_samples",
    ]

    for param in integer_params:
        best_params[param] = int(best_params[param])

    float_params = [
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
    ]

    for param in float_params:
        best_params[param] = float(best_params[param])

    logger.info("Best parameters: %s", best_params)

    logger.info("Preparing final LightGBM model...")

    final_model = build_model(best_params)

    logger.info("Fitting final preprocessing on full training data...")

    final_preprocessor = build_preprocessor(numeric_features)
    final_preprocessor.fit(X_train)

    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", final_preprocessor),
            ("model", final_model),
        ]
    )

    logger.info("Transforming full training data...")

    X_train_transformed = final_preprocessor.transform(X_train)

    logger.info("Training best configuration on full training data...")

    start_time = time.time()

    final_model.fit(
        X_train_transformed,
        y_train,
    )

    final_training_time = time.time() - start_time

    logger.info(
        "Final training completed in %.2f seconds.",
        final_training_time,
    )

    logger.info("Evaluating train...")

    train_predictions = final_pipeline.predict(X_train)
    train_metrics = calculate_metrics(
        y_train,
        train_predictions,
    )

    logger.info("Evaluating validation...")

    validation_predictions = final_pipeline.predict(X_validation)
    validation_metrics = calculate_metrics(
        y_validation,
        validation_predictions,
    )

    metrics_by_split = {
        "train": train_metrics,
        "validation": validation_metrics,
    }

    results_summary = []

    for split_name, metrics in metrics_by_split.items():
        observations = (
            len(train)
            if split_name == "train"
            else len(validation)
        )

        results_summary.append(
            {
                "model": "LightGBM Tuned",
                "dataset": split_name,
                "observations": observations,
                **metrics,
            }
        )

    summary_df = pd.DataFrame(results_summary)

    print("\nTuned LightGBM results:")
    print(summary_df.to_string(index=False))

    dataset_info = {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "search_sample_rows": len(search_sample),
        "search_sample_fraction": SEARCH_SAMPLE_FRACTION,
        "features_count": len(feature_columns),
        "target": "target_next_hour",
        "selection_metric": "Validation MAE",
        "search_trials": len(PARAMETER_CONFIGS),
        "final_training_time_seconds": final_training_time,
        "search_method": "manual_random_search",
        "baseline_included": True,
    }

    logger.info("Saving tuned LightGBM experiment...")

    experiment_id = save_experiment(
        model=final_pipeline,
        model_name="LightGBM Tuned",
        params={
            **BASE_PARAMS,
            **best_params,
            "search_sample_fraction": SEARCH_SAMPLE_FRACTION,
            "search_trials": len(PARAMETER_CONFIGS),
        },
        metrics_by_split=metrics_by_split,
        features=feature_columns,
        dataset_info=dataset_info,
        random_state=RANDOM_STATE,
    )

    print(f"\nExperiment ID: {experiment_id}")

    logger.info("LightGBM tuning completed successfully.")


if __name__ == "__main__":
    main()
