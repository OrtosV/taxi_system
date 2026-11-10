
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_project_root
from src.logging_config import setup_logging
from src.evaluation.metrics import calculate_metrics


def main():
    logger = setup_logging()
    project_root = get_project_root()

    feature_path = (
        project_root
        / "data"
        / "processed"
        / "feature_dataset.parquet"
    )

    modeling_dir = (
        project_root
        / "data"
        / "processed"
        / "modeling"
    )

    output_dir = (
        project_root
        / "reports"
        / "tables"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading feature dataset...")

    df = pd.read_parquet(
        feature_path,
        columns=["zone_id", "datetime", "trip_count"]
    )

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(
        ["zone_id", "datetime"]
    ).reset_index(drop=True)

    logger.info("Calculating seasonal naive predictions...")

    # Forecast for t+1 uses actual demand from t-23,
    # which is the same clock hour one day before t+1.
    df["forecast_datetime"] = (
        df["datetime"] + pd.Timedelta(hours=1)
    )

    df["prediction"] = (
        df.groupby("zone_id")["trip_count"].shift(23)
    )

    df["target"] = (
        df.groupby("zone_id")["trip_count"].shift(-1)
    )

    df = df.dropna(
        subset=["prediction", "target"]
    )

    results = []

    for split_name in ["train", "validation", "test"]:
        logger.info(f"Evaluating {split_name}...")

        split = pd.read_parquet(
            modeling_dir / f"{split_name}.parquet",
            columns=["forecast_datetime", "zone_id"]
        )

        split = split.merge(
            df[
                [
                    "zone_id",
                    "forecast_datetime",
                    "target",
                    "prediction",
                ]
            ],
            on=["zone_id", "forecast_datetime"],
            how="inner",
            validate="one_to_one",
        )

        y_true = split["target"].to_numpy()
        y_pred = split["prediction"].to_numpy()

        metrics = calculate_metrics(y_true, y_pred)

        results.append({
            "model": "Seasonal Naive",
            "dataset": split_name,
            "observations": len(split),
            **metrics,
        })

    results_df = pd.DataFrame(results)

    output_path = (
        output_dir / "seasonal_naive_metrics.csv"
    )

    results_df.to_csv(output_path, index=False)

    print("\nSeasonal Naive results:")
    print(results_df.to_string(index=False))

    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
