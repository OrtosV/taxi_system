
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
        / "feature_dataset.parquet"
    )

    output_dir = (
        project_root
        / "data"
        / "processed"
        / "modeling"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading feature dataset...")
    df = pd.read_parquet(input_path)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(
        ["zone_id", "datetime"]
    ).reset_index(drop=True)

    # 1. Target: demand in the next hour
    logger.info("Creating next-hour target...")

    df["forecast_datetime"] = (
        df["datetime"] + pd.Timedelta(hours=1)
    )

    df["target_next_hour"] = (
        df.groupby("zone_id")["trip_count"].shift(-1)
    )

    # 2. Calendar features for the forecast hour
    logger.info("Creating forecast calendar features...")

    forecast_time = df["forecast_datetime"]

    df["forecast_hour"] = forecast_time.dt.hour
    df["forecast_day_of_week"] = forecast_time.dt.day_of_week
    df["forecast_day_of_month"] = forecast_time.dt.day
    df["forecast_month"] = forecast_time.dt.month

    df["forecast_is_weekend"] = (
        df["forecast_day_of_week"] >= 5
    ).astype("int8")

    df["forecast_hour_sin"] = np.sin(
        2 * np.pi * df["forecast_hour"] / 24
    )
    df["forecast_hour_cos"] = np.cos(
        2 * np.pi * df["forecast_hour"] / 24
    )

    df["forecast_day_of_week_sin"] = np.sin(
        2 * np.pi * df["forecast_day_of_week"] / 7
    )
    df["forecast_day_of_week_cos"] = np.cos(
        2 * np.pi * df["forecast_day_of_week"] / 7
    )

    # 3. Select model features
    logger.info("Selecting model features...")

    calendar_features = [
        "forecast_hour",
        "forecast_day_of_week",
        "forecast_day_of_month",
        "forecast_month",
        "forecast_is_weekend",
        "forecast_hour_sin",
        "forecast_hour_cos",
        "forecast_day_of_week_sin",
        "forecast_day_of_week_cos",
    ]

    demand_features = [
        f"demand_lag_{lag}h"
        for lag in [1, 2, 3, 6, 24, 48, 168]
    ]

    rolling_features = [
        f"demand_rolling_mean_{window}h"
        for window in [3, 6, 24, 168]
    ] + [
        f"demand_rolling_std_{window}h"
        for window in [24, 168]
    ]

    external_features = [
        "temperature_2m_prev_hour",
        "relative_humidity_2m_prev_hour",
        "precipitation_prev_hour",
        "wind_speed_10m_prev_hour",
        "weather_code_prev_hour",
        "weather_data_available_prev_hour",
        "collision_count_prev_hour",
        "injured_count_prev_hour",
        "killed_count_prev_hour",
        "pedestrians_injured_prev_hour",
        "pedestrians_killed_prev_hour",
        "cyclists_injured_prev_hour",
        "cyclists_killed_prev_hour",
        "motorists_injured_prev_hour",
        "motorists_killed_prev_hour",
        "total_311_count_prev_hour",
        "traffic_related_311_count_prev_hour",
        "road_condition_311_count_prev_hour",
        "urban_activity_311_count_prev_hour",
        "residential_environment_311_count_prev_hour",
        "described_311_count_prev_hour",
    ]

    spatial_features = [
        "zone_id",
        "Borough",
        "Zone",
        "service_zone",
    ]

    feature_columns = (
        calendar_features
        + demand_features
        + rolling_features
        + external_features
        + spatial_features
    )

    feature_columns = [
        column
        for column in feature_columns
        if column in df.columns
    ]

    selected_columns = (
        ["forecast_datetime", "target_next_hour"]
        + feature_columns
    )

    model_df = df[selected_columns].copy()

    # 4. Remove rows without target or required demand history
    logger.info("Preparing modeling dataset...")

    required_history = demand_features + rolling_features

    model_df = model_df.dropna(
        subset=["target_next_hour"] + required_history
    )

    model_df["target_next_hour"] = (
        model_df["target_next_hour"].astype("int64")
    )

    model_df = model_df.sort_values(
        "forecast_datetime"
    ).reset_index(drop=True)

    # 5. Chronological train / validation / test split
    logger.info("Splitting data chronologically...")

    unique_times = np.sort(
        model_df["forecast_datetime"].unique()
    )

    train_cut = int(len(unique_times) * 0.70)
    validation_cut = int(len(unique_times) * 0.85)

    train_end = unique_times[train_cut - 1]
    validation_end = unique_times[validation_cut - 1]

    train_df = model_df[
        model_df["forecast_datetime"] <= train_end
    ]

    validation_df = model_df[
        (model_df["forecast_datetime"] > train_end)
        & (model_df["forecast_datetime"] <= validation_end)
    ]

    test_df = model_df[
        model_df["forecast_datetime"] > validation_end
    ]

    # 6. Save datasets
    logger.info("Saving train, validation and test datasets...")

    train_df.to_parquet(
        output_dir / "train.parquet",
        index=False
    )

    validation_df.to_parquet(
        output_dir / "validation.parquet",
        index=False
    )

    test_df.to_parquet(
        output_dir / "test.parquet",
        index=False
    )

    logger.info(
        f"Features: {len(feature_columns)}"
    )
    logger.info(
        f"Train: {len(train_df):,} rows, "
        f"{train_df['forecast_datetime'].min()} - "
        f"{train_df['forecast_datetime'].max()}"
    )
    logger.info(
        f"Validation: {len(validation_df):,} rows, "
        f"{validation_df['forecast_datetime'].min()} - "
        f"{validation_df['forecast_datetime'].max()}"
    )
    logger.info(
        f"Test: {len(test_df):,} rows, "
        f"{test_df['forecast_datetime'].min()} - "
        f"{test_df['forecast_datetime'].max()}"
    )

    logger.info(
        f"Modeling datasets saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()
