from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.logging_config import setup_logging

TRAIN_PATH = (
    PROJECT_ROOT / "data" / "processed" / "modeling" / "train.parquet"
)
VALIDATION_PATH = (
    PROJECT_ROOT / "data" / "processed" / "modeling" / "validation.parquet"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "monitoring" / "data_drift.csv"
)

RANDOM_STATE = 42
SAMPLE_SIZE = 100_000
N_BINS = 10
EPSILON = 1e-6

FEATURES_TO_MONITOR = [
    "demand_lag_1h",
    "demand_lag_168h",
    "temperature_2m_prev_hour",
    "precipitation_prev_hour",
    "wind_speed_10m_prev_hour",
    "total_311_count_prev_hour",
]


def calculate_psi(reference, current):
    """
    Рассчитывает Population Stability Index
    между эталонным и текущим распределениями.
    """
    reference = np.asarray(reference, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) == 0 or len(current) == 0:
        return np.nan

    quantiles = np.linspace(0, 1, N_BINS + 1)
    edges = np.quantile(reference, quantiles)

    edges = np.unique(edges)

    if len(edges) == 1:
        edges = np.array([-np.inf, np.inf])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf

    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)

    reference_share = reference_counts / len(reference)
    current_share = current_counts / len(current)

    reference_share = np.clip(reference_share, EPSILON, None)
    current_share = np.clip(current_share, EPSILON, None)

    psi = np.sum(
        (current_share - reference_share)
        * np.log(current_share / reference_share)
    )

    return float(psi)


def get_drift_status(psi):
    if pd.isna(psi):
        return "Недостаточно данных"

    if psi < 0.1:
        return "Незначительное изменение"

    if psi < 0.2:
        return "Умеренное изменение"

    return "Существенное изменение"


def split_validation_by_time(validation):
    validation = validation.copy()

    validation["forecast_datetime"] = pd.to_datetime(
        validation["forecast_datetime"]
    )

    timestamps = np.sort(
        validation["forecast_datetime"].unique()
    )

    split_index = len(timestamps) // 2
    split_datetime = timestamps[split_index]

    first_period = validation[
        validation["forecast_datetime"] < split_datetime
    ]

    second_period = validation[
        validation["forecast_datetime"] >= split_datetime
    ]

    return first_period, second_period


def sample_data(data, features):
    sample_size = min(SAMPLE_SIZE, len(data))

    return data[features].sample(
        n=sample_size,
        random_state=RANDOM_STATE
    )


def main():
    logger = setup_logging()

    logger.info("Начало анализа Data Drift")

    train = pd.read_parquet(TRAIN_PATH)
    validation = pd.read_parquet(VALIDATION_PATH)

    required_columns = FEATURES_TO_MONITOR + ["forecast_datetime"]

    for column in required_columns:
        if column not in train.columns:
            raise ValueError(
                f"В Train отсутствует столбец: {column}"
            )

        if column not in validation.columns:
            raise ValueError(
                f"В Validation отсутствует столбец: {column}"
            )

    first_period, second_period = split_validation_by_time(
        validation
    )

    reference_sample = sample_data(train, FEATURES_TO_MONITOR)
    first_sample = sample_data(first_period, FEATURES_TO_MONITOR)
    second_sample = sample_data(second_period, FEATURES_TO_MONITOR)

    periods = {
        "validation_period_1": first_sample,
        "validation_period_2": second_sample,
    }

    results = []

    for feature in FEATURES_TO_MONITOR:
        reference_values = reference_sample[feature].dropna()

        for period_name, current_data in periods.items():
            current_values = current_data[feature].dropna()

            psi = calculate_psi(
                reference_values,
                current_values
            )

            results.append({
                "feature": feature,
                "period": period_name,
                "PSI": psi,
                "status": get_drift_status(psi),
                "reference_rows": len(reference_values),
                "current_rows": len(current_values),
            })

    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    logger.info(
        "Анализ Data Drift завершён. Результаты: %s",
        OUTPUT_PATH
    )

    print("\nРезультаты Data Drift:")
    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    print(f"\nФайл сохранён: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
