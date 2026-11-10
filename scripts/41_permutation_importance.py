
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# Финальная модель
EXPERIMENT_ID = "exp_20261110_153017"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "trained"
    / f"xgboost_{EXPERIMENT_ID}.joblib"
)

EXPERIMENT_PATH = (
    PROJECT_ROOT
    / "models"
    / "registry"
    / "experiments"
    / f"{EXPERIMENT_ID}.json"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling"
    / "validation.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = (
    OUTPUT_DIR / "permutation_importance.csv"
)

SAMPLE_SIZE = 30_000
N_REPEATS = 5
RANDOM_STATE = 42

CATEGORICAL_FEATURES = [
    "zone_id",
    "Borough",
    "Zone",
    "service_zone",
]


def main():
    logger.info("Загрузка метаданных эксперимента.")

    with open(
        EXPERIMENT_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        experiment = json.load(file)

    features = experiment["features"]

    logger.info("Загрузка финальной модели.")

    model = joblib.load(MODEL_PATH)

    logger.info("Загрузка Validation.")

    validation_df = pd.read_parquet(
        VALIDATION_PATH,
        columns=features + ["target_next_hour"],
    )

    logger.info(
        "Размер исходной Validation: %s строк.",
        len(validation_df),
    )

    # Фиксированная случайная выборка
    if len(validation_df) > SAMPLE_SIZE:
        validation_sample = validation_df.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        )
    else:
        validation_sample = validation_df.copy()

    validation_sample = validation_sample.reset_index(
        drop=True
    )

    X_validation = validation_sample[features].copy()
    y_validation = validation_sample["target_next_hour"]

    # Воспроизводим типы категориальных признаков,
    # использованные при обучении модели.
    for column in CATEGORICAL_FEATURES:
        if column in X_validation.columns:
            X_validation[column] = (
                X_validation[column].astype(str)
            )

    logger.info(
        "Выборка для интерпретации: %s строк, %s признаков.",
        len(X_validation),
        len(features),
    )

    # Базовое качество на выбранной выборке
    logger.info("Расчёт исходного MAE.")

    baseline_predictions = model.predict(
        X_validation
    )

    baseline_mae = mean_absolute_error(
        y_validation,
        baseline_predictions,
    )

    logger.info(
        "Исходный MAE: %.6f",
        baseline_mae,
    )

    # Permutation Importance
    logger.info(
        "Запуск Permutation Importance: %s повторов.",
        N_REPEATS,
    )

    start_time = time.perf_counter()

    importance_result = permutation_importance(
        estimator=model,
        X=X_validation,
        y=y_validation,
        scoring="neg_mean_absolute_error",
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    elapsed_time = time.perf_counter() - start_time

    # Формируем таблицу результатов
    importance_df = pd.DataFrame({
        "feature": features,
        "importance_mean": importance_result.importances_mean,
        "importance_std": importance_result.importances_std,
    })

    importance_df = importance_df.sort_values(
        by="importance_mean",
        ascending=False,
    ).reset_index(drop=True)

    importance_df["importance_rank"] = (
        importance_df.index + 1
    )

    importance_df["baseline_mae"] = baseline_mae
    importance_df["sample_size"] = len(X_validation)
    importance_df["n_repeats"] = N_REPEATS

    importance_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(
        "Permutation Importance завершён за %.2f секунд.",
        elapsed_time,
    )

    print("\nРЕЗУЛЬТАТЫ PERMUTATION IMPORTANCE")
    print("=" * 90)

    print(
        importance_df[
            [
                "importance_rank",
                "feature",
                "importance_mean",
                "importance_std",
            ]
        ].to_string(index=False)
    )

    print("\nИсходный MAE:", round(baseline_mae, 6))
    print("Время выполнения:", round(elapsed_time, 2), "сек.")

    print(
        "\nРезультаты сохранены:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
