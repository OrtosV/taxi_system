
from pathlib import Path
import sys

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import calculate_metrics
from src.logging_config import setup_logging


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "trained"
    / "xgboost_exp_20261110_153017.joblib"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling"
    / "validation.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "monitoring"
    / "performance_drift.csv"
)

CATEGORICAL_FEATURES = [
    "zone_id",
    "Borough",
    "Zone",
    "service_zone",
]


def main():
    logger = setup_logging()
    logger.info("Начало анализа Performance Drift")

    model = joblib.load(MODEL_PATH)
    validation = pd.read_parquet(VALIDATION_PATH)

    validation["forecast_datetime"] = pd.to_datetime(
        validation["forecast_datetime"]
    )

    # Получаем список признаков непосредственно из обученного Pipeline.
    features = list(model.feature_names_in_)

    X = validation[features].copy()
    y_true = validation["target_next_hour"]

    # Приводим категориальные признаки к тому же формату,
    # который использовался при обучении модели.
    for column in CATEGORICAL_FEATURES:
        if column in X.columns:
            X[column] = X[column].fillna("Unknown").astype(str)

    logger.info("Формирование прогнозов на Validation")

    y_pred = model.predict(X)

    # Собираем результаты прогнозирования.
    results = pd.DataFrame({
        "forecast_datetime": validation["forecast_datetime"],
        "y_true": y_true.to_numpy(),
        "y_pred": y_pred,
    })

    # Разбиваем Validation по календарным месяцам.
    results["period"] = (
        results["forecast_datetime"].dt.to_period("M").astype(str)
    )

    monthly_metrics = []

    for period, group in results.groupby("period", sort=True):
        metrics = calculate_metrics(
            group["y_true"],
            group["y_pred"]
        )

        monthly_metrics.append({
            "period": period,
            "rows": len(group),
            **metrics,
        })

    metrics_df = pd.DataFrame(monthly_metrics)

    # Рассчитываем общие метрики для проверки с предыдущей оценкой.
    overall_metrics = calculate_metrics(
        results["y_true"],
        results["y_pred"]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    logger.info(
        "Анализ Performance Drift завершён. Результаты: %s",
        OUTPUT_PATH
    )

    print("\nМетрики качества по месяцам:")
    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    print("\nОбщие метрики на Validation:")
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value:.6f}")

    print(f"\nРезультаты сохранены: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
