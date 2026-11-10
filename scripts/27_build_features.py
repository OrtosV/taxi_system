
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_project_root
from src.logging_config import setup_logging


def main():
    logger = setup_logging()
    project_root = get_project_root()

    input_path = (
        project_root
        / "data"
        / "processed"
        / "unified_hourly_dataset.parquet"
    )

    output_path = (
        project_root
        / "data"
        / "processed"
        / "feature_dataset.parquet"
    )

    logger.info("Loading unified dataset...")
    df = pd.read_parquet(input_path)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["zone_id", "datetime"]).reset_index(drop=True)

    # 1. Calendar features
    logger.info("Creating calendar features...")

    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.day_of_week
    df["day_of_month"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["day_of_week_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )
    df["day_of_week_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    # 2. Lag features of taxi demand
    logger.info("Creating demand lag features...")

    demand_lags = [1, 2, 3, 6, 24, 48, 168]

    grouped_demand = df.groupby("zone_id")["trip_count"]

    for lag in demand_lags:
        df[f"demand_lag_{lag}h"] = grouped_demand.shift(lag)

    # 3. Rolling demand statistics
    logger.info("Creating rolling demand features...")

    rolling_windows = [3, 6, 24, 168]

    for window in rolling_windows:
        df[f"demand_rolling_mean_{window}h"] = (
            df.groupby("zone_id")["trip_count"]
            .transform(
                lambda s: s.shift(1).rolling(
                    window=window,
                    min_periods=window
                ).mean()
            )
        )

    for window in [24, 168]:
        df[f"demand_rolling_std_{window}h"] = (
            df.groupby("zone_id")["trip_count"]
            .transform(
                lambda s: s.shift(1).rolling(
                    window=window,
                    min_periods=window
                ).std()
            )
        )

    # 4. External features: use only the previous known hour
    logger.info("Shifting external features to prevent data leakage...")

    external_features = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
        "weather_code",
        "weather_data_available",
        "collision_count",
        "injured_count",
        "killed_count",
        "pedestrians_injured",
        "pedestrians_killed",
        "cyclists_injured",
        "cyclists_killed",
        "motorists_injured",
        "motorists_killed",
        "total_311_count",
        "traffic_related_311_count",
        "road_condition_311_count",
        "urban_activity_311_count",
        "residential_environment_311_count",
        "described_311_count",
    ]

    for column in external_features:
        if column in df.columns:
            df[f"{column}_prev_hour"] = (
                df.groupby("zone_id")[column].shift(1)
            )

    # 5. Save feature dataset
    logger.info("Saving feature dataset...")

    df.to_parquet(output_path, index=False)

    logger.info(
        "Feature engineering completed. "
        f"Rows: {len(df):,}, "
        f"Columns: {len(df.columns)}, "
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()
