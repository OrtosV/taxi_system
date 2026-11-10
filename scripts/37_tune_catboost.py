
import sys
import time
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor

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
    "loss_function": "RMSE",
    "random_seed": RANDOM_STATE,
    "thread_count": -1,
    "verbose": False,
    "allow_writing_files": False,
}

# Исходная конфигурация + 9 альтернатив
PARAMETER_CONFIGS = [
    # 1. Исходная конфигурация CatBoost
    {
        "iterations": 300,
        "depth": 8,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3,
        "random_strength": 1,
        "bagging_temperature": 1,
    },
    # 2. Более простые деревья
    {
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3,
        "random_strength": 1,
        "bagging_temperature": 1,
    },
    # 3. Больше деревьев при небольшой глубине
    {
        "iterations": 400,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3,
        "random_strength": 1,
        "bagging_temperature": 1,
    },
    # 4. Более глубокие деревья
    {
        "iterations": 300,
        "depth": 10,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3,
        "random_strength": 1,
        "bagging_temperature": 1,
    },
    # 5. Глубокие деревья и более сильная регуляризация
    {
        "iterations": 400,
        "depth": 10,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5,
        "random_strength": 1,
        "bagging_temperature": 1,
    },
    # 6. Низкая скорость обучения
    {
        "iterations": 500,
        "depth": 8,
        "learning_rate": 0.03,
        "l2_leaf_reg": 5,
        "random_strength": 1,
        "bagging_temperature": 1,
    },
    # 7. Повышенная случайность
    {
        "iterations": 400,
        "depth": 8,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3,
        "random_strength": 2,
        "bagging_temperature": 2,
    },
    # 8. Более сильная регуляризация
    {
        "iterations": 400,
        "depth": 8,
        "learning_rate": 0.05,
        "l2_leaf_reg": 7,
        "random_strength": 1,
        "bagging_temperature": 1,
    },
    # 9. Более высокая скорость обучения
    {
        "iterations": 300,
        "depth": 8,
        "learning_rate": 0.07,
        "l2_leaf_reg": 3,
        "random_strength": 1,
        "bagging_temperature": 1,
    },
    # 10. Глубокие деревья с дополнительной регуляризацией
    {
        "iterations": 500,
        "depth": 10,
        "learning_rate": 0.03,
        "l2_leaf_reg": 7,
        "random_strength": 2,
        "bagging_temperature": 2,
    },
]


# ---------------------------------------------------------
# Data preparation
# ---------------------------------------------------------

def prepare_features(data):
    """
    Подготавливает признаки для CatBoost.

    Числовые пропуски CatBoost обрабатывает самостоятельно.
    Категориальные признаки преобразуются в строки.
    """

    prepared = data.copy()

    for column in CATEGORICAL_FEATURES:
        prepared[column] = (
            prepared[column]
            .fillna("Unknown")
            .astype(str)
        )

    return prepared


# ---------------------------------------------------------
# Model creation
# ---------------------------------------------------------

def build_model(params):
    model_params = {
        **BASE_PARAMS,
        **params,
    }

    return CatBoostRegressor(**model_params)


# ---------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------

def run_parameter_search(
    X_search,
    y_search,
    X_validation,
    y_validation,
    logger,
):
    results = []

    logger.info("Starting CatBoost hyperparameter search...")

    for trial_number, params in enumerate(PARAMETER_CONFIGS, start=1):
        logger.info(
            "Starting trial %s/%s",
            trial_number,
            len(PARAMETER_CONFIGS),
        )

        start_time = time.time()

        model = build_model(params)

        model.fit(
            X_search,
            y_search,
            cat_features=CATEGORICAL_FEATURES,
        )

        predictions = model.predict(X_validation)

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

    results_path = REPORTS_DIR / "catboost_tuning_results.csv"
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

    # Подготовка категориальных признаков
    X_train = prepare_features(X_train)
    X_validation = prepare_features(X_validation)

    # Случайная выборка 10% обучающих данных
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

    results_df = run_parameter_search(
        X_search=search_sample,
        y_search=y_search,
        X_validation=X_validation,
        y_validation=y_validation,
        logger=logger,
    )

    print("\nCatBoost tuning results:")
    print(results_df.to_string(index=False))

    # Выбор лучшей конфигурации
    best_trial = results_df.iloc[0]

    best_params = {
        key: best_trial[key]
        for key in PARAMETER_CONFIGS[0].keys()
    }

    # Восстанавливаем типы параметров после чтения из DataFrame
    integer_params = [
        "iterations",
        "depth",
    ]

    for param in integer_params:
        best_params[param] = int(best_params[param])

    float_params = [
        "learning_rate",
        "l2_leaf_reg",
        "random_strength",
        "bagging_temperature",
    ]

    for param in float_params:
        best_params[param] = float(best_params[param])

    logger.info("Best parameters: %s", best_params)

    # Финальное обучение на полном Train
    logger.info("Preparing final CatBoost model...")

    final_model = build_model(best_params)

    logger.info("Training best configuration on full training data...")

    start_time = time.time()

    final_model.fit(
        X_train,
        y_train,
        cat_features=CATEGORICAL_FEATURES,
    )

    final_training_time = time.time() - start_time

    logger.info(
        "Final training completed in %.2f seconds.",
        final_training_time,
    )

    # Оценка на Train
    logger.info("Evaluating train...")

    train_predictions = final_model.predict(X_train)

    train_metrics = calculate_metrics(
        y_train,
        train_predictions,
    )

    # Оценка на Validation
    logger.info("Evaluating validation...")

    validation_predictions = final_model.predict(X_validation)

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
                "model": "CatBoost Tuned",
                "dataset": split_name,
                "observations": observations,
                **metrics,
            }
        )

    summary_df = pd.DataFrame(results_summary)

    print("\nTuned CatBoost results:")
    print(summary_df.to_string(index=False))

    dataset_info = {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "search_sample_rows": len(search_sample),
        "search_sample_fraction": SEARCH_SAMPLE_FRACTION,
        "features_count": len(feature_columns),
        "target": "target_next_hour",
        "categorical_features": CATEGORICAL_FEATURES,
        "selection_metric": "Validation MAE",
        "search_trials": len(PARAMETER_CONFIGS),
        "final_training_time_seconds": final_training_time,
        "search_method": "manual_random_search",
        "baseline_included": True,
    }

    # Регистрация эксперимента
    logger.info("Saving tuned CatBoost experiment...")

    experiment_id = save_experiment(
        model=final_model,
        model_name="CatBoost Tuned",
        params={
            **BASE_PARAMS,
            **best_params,
            "categorical_features": CATEGORICAL_FEATURES,
            "search_sample_fraction": SEARCH_SAMPLE_FRACTION,
            "search_trials": len(PARAMETER_CONFIGS),
        },
        metrics_by_split=metrics_by_split,
        features=feature_columns,
        dataset_info=dataset_info,
        random_state=RANDOM_STATE,
    )

    print(f"\nExperiment ID: {experiment_id}")

    logger.info("CatBoost tuning completed successfully.")


if __name__ == "__main__":
    main()
