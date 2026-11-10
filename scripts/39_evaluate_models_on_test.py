
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import calculate_metrics


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


MODELS_DIR = PROJECT_ROOT / "models" / "trained"
EXPERIMENTS_DIR = PROJECT_ROOT / "models" / "registry" / "experiments"

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling"
    / "test.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "final_test_metrics.csv"


MODEL_EXPERIMENTS = [
    {
        "name": "Random Forest",
        "experiment_id": "exp_20261110_151730",
        "model_file": "random_forest_exp_20261110_151730.joblib",
    },
    {
        "name": "Random Forest Tuned",
        "experiment_id": "exp_20261110_171542",
        "model_file": "random_forest_tuned_exp_20261110_171542.joblib",
    },
    {
        "name": "XGBoost",
        "experiment_id": "exp_20261110_153017",
        "model_file": "xgboost_exp_20261110_153017.joblib",
    },
    {
        "name": "XGBoost Tuned",
        "experiment_id": "exp_20261110_173122",
        "model_file": "xgboost_tuned_exp_20261110_173122.joblib",
    },
    {
        "name": "LightGBM",
        "experiment_id": "exp_20261110_153442",
        "model_file": "lightgbm_exp_20261110_153442.joblib",
    },
    {
        "name": "LightGBM Tuned",
        "experiment_id": "exp_20261110_173517",
        "model_file": "lightgbm_tuned_exp_20261110_173517.joblib",
    },
    {
        "name": "CatBoost",
        "experiment_id": "exp_20261110_155122",
        "model_file": "catboost_exp_20261110_155122.joblib",
    },
    {
        "name": "CatBoost Tuned",
        "experiment_id": "exp_20261110_180422",
        "model_file": "catboost_tuned_exp_20261110_180422.joblib",
    },
]


def load_experiment(experiment_id):
    """Загружает метаданные эксперимента из JSON."""

    experiment_path = (
        EXPERIMENTS_DIR / f"{experiment_id}.json"
    )

    if not experiment_path.exists():
        raise FileNotFoundError(
            f"Не найден JSON эксперимента: {experiment_path}"
        )

    with open(
        experiment_path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def evaluate_model(model_info, X_test, y_test):
    """Загружает Pipeline и оценивает его на Test."""

    model_path = MODELS_DIR / model_info["model_file"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Не найдена модель: {model_path}"
        )

    logger.info(
        "Загрузка модели: %s",
        model_info["name"],
    )

    model = joblib.load(model_path)

    # Для CatBoost восстанавливаем ту же обработку
    # пропусков в категориальных признаках, что использовалась
    # при обучении модели.
    X_model = X_test.copy()

    if "CatBoost" in model_info["name"]:
        categorical_features = [
            "zone_id",
            "Borough",
            "Zone",
            "service_zone",
        ]

        for column in categorical_features:
            if column in X_model.columns:
                X_model[column] = (
                    X_model[column]
                    .fillna("Unknown")
                    .astype(str)
                )

    start_time = time.perf_counter()

    predictions = model.predict(X_model)

    prediction_time = time.perf_counter() - start_time

    metrics = calculate_metrics(
        y_test,
        predictions,
    )

    result = {
        "model_name": model_info["name"],
        "experiment_id": model_info["experiment_id"],
        **metrics,
        "prediction_time_seconds": prediction_time,
    }

    logger.info(
        "%s | MAE: %.4f | RMSE: %.4f | WAPE: %.4f%%",
        model_info["name"],
        metrics["MAE"],
        metrics["RMSE"],
        metrics["WAPE_percent"],
    )

    return result


def main():
    logger.info("Загрузка тестовой выборки.")

    test_df = pd.read_parquet(TEST_PATH)

    if "target_next_hour" not in test_df.columns:
        raise ValueError(
            "В тестовой выборке отсутствует target_next_hour."
        )

    # Загружаем список признаков из JSON первого эксперимента.
    reference_experiment = load_experiment(
        MODEL_EXPERIMENTS[0]["experiment_id"]
    )

    features = reference_experiment["features"]

    missing_features = [
        feature
        for feature in features
        if feature not in test_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"В тестовой выборке отсутствуют признаки: "
            f"{missing_features}"
        )

    X_test = test_df[features].copy()
    y_test = test_df["target_next_hour"]

    # Приводим категориальные признаки к типам,
    # использованным при обучении моделей.
    categorical_features = [
        "zone_id",
        "Borough",
        "Zone",
        "service_zone",
    ]

    for column in categorical_features:
        if column in X_test.columns:
            X_test[column] = X_test[column].astype(str)

    logger.info(
        "Тестовая выборка: %s строк, %s признаков.",
        len(test_df),
        len(features),
    )

    results = []

    for model_info in MODEL_EXPERIMENTS:
        experiment = load_experiment(
            model_info["experiment_id"]
        )

        # Проверяем, что все модели используют одинаковые признаки.
        if experiment["features"] != features:
            raise ValueError(
                f"Набор признаков модели {model_info['name']} "
                "отличается от эталонного."
            )

        result = evaluate_model(
            model_info,
            X_test,
            y_test,
        )

        results.append(result)

    # Добавляем Seasonal Naive.
    seasonal_experiment = load_experiment(
        "exp_20261110_184505_seasonal_naive"
    )

    seasonal_metrics = seasonal_experiment["metrics"]["test"]

    results.append({
        "model_name": "Seasonal Naive",
        "experiment_id": seasonal_experiment["experiment_id"],
        **seasonal_metrics,
        "prediction_time_seconds": None,
    })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="MAE",
        ascending=True,
    ).reset_index(drop=True)

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ НА TEST")
    print("=" * 90)

    print(
        results_df[
            [
                "model_name",
                "MAE",
                "MSE",
                "RMSE",
                "WAPE_percent",
                "R2",
            ]
        ].to_string(index=False)
    )

    print("\nЛучшая модель по MAE:")
    print(results_df.iloc[0]["model_name"])

    print(
        f"\nРезультаты сохранены: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
