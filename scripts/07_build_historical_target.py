from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "taxi"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

MONTHLY_DIR = INTERIM_DIR / "monthly_demand"

OUTPUT_PATH = INTERIM_DIR / "historical_hourly_demand.parquet"

BATCH_SIZE = 250_000

START_YEAR = 2024
START_MONTH = 1
END_YEAR = 2025
END_MONTH = 12


def get_months():
    months = []

    current = pd.Timestamp(
        year=START_YEAR,
        month=START_MONTH,
        day=1,
    )

    end = pd.Timestamp(
        year=END_YEAR,
        month=END_MONTH,
        day=1,
    )

    while current <= end:
        months.append((current.year, current.month))
        current += pd.DateOffset(months=1)

    return months


def aggregate_month(year: int, month: int) -> Path:
    """Агрегирует один месяц по зоне и часу."""

    file_name = f"yellow_tripdata_{year}-{month:02d}.parquet"
    source_path = RAW_DIR / file_name

    output_path = MONTHLY_DIR / (
        f"hourly_demand_{year}-{month:02d}.parquet"
    )

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.DateOffset(months=1)

    parquet_file = pq.ParquetFile(source_path)

    monthly_parts = []

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=[
            "tpep_pickup_datetime",
            "PULocationID",
        ],
    ):
        df = batch.to_pandas()

        df = df.dropna(
            subset=["tpep_pickup_datetime", "PULocationID"]
        )

        # Оставляем только записи соответствующего месяца.
        df = df[
            (df["tpep_pickup_datetime"] >= start)
            & (df["tpep_pickup_datetime"] < end)
        ]

        if df.empty:
            continue

        df["datetime"] = (
            df["tpep_pickup_datetime"].dt.floor("h")
        )

        part = (
            df.groupby(
                ["PULocationID", "datetime"]
            )
            .size()
            .reset_index(name="trip_count")
        )

        monthly_parts.append(part)

    if not monthly_parts:
        raise ValueError(f"No valid records in {file_name}")

    result = (
        pd.concat(monthly_parts, ignore_index=True)
        .groupby(
            ["PULocationID", "datetime"],
            as_index=False,
        )["trip_count"]
        .sum()
    )

    result = result.rename(
        columns={"PULocationID": "zone_id"}
    )

    result["trip_count"] = result["trip_count"].astype("int32")

    MONTHLY_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)

    return output_path


def main():
    months = get_months()

    print("=" * 65)
    print("HISTORICAL TAXI DEMAND AGGREGATION")
    print("=" * 65)

    monthly_files = []

    for year, month in tqdm(months, desc="Processing months"):
        result_path = aggregate_month(year, month)
        monthly_files.append(result_path)

        result = pd.read_parquet(result_path)

        print(
            f"{year}-{month:02d}: "
            f"{result['trip_count'].sum():,} trips, "
            f"{len(result):,} zone-hour combinations"
        )

    # Объединяем компактные месячные агрегации.
    print("\nCombining monthly aggregations...")

    parts = [
        pd.read_parquet(path)
        for path in monthly_files
    ]

    demand = pd.concat(parts, ignore_index=True)

    demand = (
        demand.groupby(
            ["zone_id", "datetime"],
            as_index=False,
        )["trip_count"]
        .sum()
    )

    # Формируем полный временной диапазон.
    start = pd.Timestamp(
        year=START_YEAR,
        month=START_MONTH,
        day=1,
    )

    end = pd.Timestamp(
        year=END_YEAR + (END_MONTH // 12),
        month=(END_MONTH % 12) + 1,
        day=1,
    )

    hours = pd.date_range(
        start=start,
        end=end,
        freq="h",
        inclusive="left",
    )

    zones = sorted(demand["zone_id"].unique())

    full_index = pd.MultiIndex.from_product(
        [zones, hours],
        names=["zone_id", "datetime"],
    )

    demand = (
        demand.set_index(["zone_id", "datetime"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    demand["trip_count"] = demand["trip_count"].astype("int32")

    demand = demand.sort_values(
        ["zone_id", "datetime"]
    ).reset_index(drop=True)

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    demand.to_parquet(OUTPUT_PATH, index=False)

    print("\n" + "=" * 65)
    print("HISTORICAL DATASET SUMMARY")
    print("=" * 65)

    print(f"Months: {len(months)}")
    print(f"Zones: {demand['zone_id'].nunique()}")
    print(f"Hours: {demand['datetime'].nunique():,}")
    print(f"Rows: {len(demand):,}")
    print(f"Total trips: {demand['trip_count'].sum():,}")
    print(f"Zero-demand rows: {(demand['trip_count'] == 0).sum():,}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()