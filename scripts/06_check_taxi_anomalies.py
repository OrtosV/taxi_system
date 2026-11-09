from pathlib import Path
from collections import Counter

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "taxi"

BATCH_SIZE = 100_000

FILES = sorted(DATA_DIR.glob("yellow_tripdata_*.parquet"))

REQUIRED_COLUMNS = [
    "tpep_pickup_datetime",
    "PULocationID",
]


def get_expected_period(file_path: Path):
    """Получает ожидаемые границы месяца из имени файла."""

    parts = file_path.stem.split("_")
    year, month = map(int, parts[-1].split("-"))

    start = pd.Timestamp(year=year, month=month, day=1)

    if month == 12:
        end = pd.Timestamp(year=year + 1, month=1, day=1)
    else:
        end = pd.Timestamp(year=year, month=month + 1, day=1)

    return start, end


def inspect_temporal_anomalies(file_path: Path) -> dict:
    parquet_file = pq.ParquetFile(file_path)

    start, end = get_expected_period(file_path)

    total_rows = 0
    before_period = 0
    after_period = 0
    missing_dates = 0

    min_date = None
    max_date = None

    abnormal_years = Counter()

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["tpep_pickup_datetime"],
    ):
        df = batch.to_pandas()
        dates = df["tpep_pickup_datetime"]

        total_rows += len(df)
        missing_dates += dates.isna().sum()

        valid_dates = dates.dropna()

        if valid_dates.empty:
            continue

        batch_min = valid_dates.min()
        batch_max = valid_dates.max()

        if min_date is None or batch_min < min_date:
            min_date = batch_min

        if max_date is None or batch_max > max_date:
            max_date = batch_max

        before_mask = valid_dates < start
        after_mask = valid_dates >= end

        before_period += before_mask.sum()
        after_period += after_mask.sum()

        abnormal_dates = valid_dates[before_mask | after_mask]

        abnormal_years.update(
            abnormal_dates.dt.year.value_counts().to_dict()
        )

    outside = before_period + after_period

    return {
        "file": file_path.name,
        "total_rows": total_rows,
        "min_date": min_date,
        "max_date": max_date,
        "before_period": before_period,
        "after_period": after_period,
        "outside_period": outside,
        "outside_percent": outside / total_rows * 100,
        "missing_dates": missing_dates,
        "abnormal_years": abnormal_years,
    }


def compare_schemas(files: list[Path]) -> None:
    """Сравнивает состав и типы столбцов Parquet-файлов."""

    schemas = {}

    for file_path in files:
        parquet_file = pq.ParquetFile(file_path)

        schemas[file_path.name] = {
            field.name: str(field.type)
            for field in parquet_file.schema_arrow
        }

    reference_name = files[0].name
    reference_schema = schemas[reference_name]

    print("\n" + "=" * 70)
    print("SCHEMA COMPARISON")
    print("=" * 70)

    print(f"Reference file: {reference_name}")

    for file_name, schema in schemas.items():
        if schema == reference_schema:
            continue

        print(f"\nDifferences in {file_name}:")

        reference_columns = set(reference_schema)
        current_columns = set(schema)

        added = current_columns - reference_columns
        removed = reference_columns - current_columns

        changed = {
            column
            for column in reference_columns & current_columns
            if reference_schema[column] != schema[column]
        }

        if added:
            print("  Added columns:")
            for column in sorted(added):
                print(f"    {column}: {schema[column]}")

        if removed:
            print("  Removed columns:")
            for column in sorted(removed):
                print(f"    {column}: {reference_schema[column]}")

        if changed:
            print("  Changed column types:")
            for column in sorted(changed):
                print(
                    f"    {column}: "
                    f"{reference_schema[column]} -> {schema[column]}"
                )


def main() -> None:
    print("=" * 70)
    print("TAXI DATA TEMPORAL ANOMALY CHECK")
    print("=" * 70)

    results = []

    for file_path in FILES:
        print(f"\n[CHECK] {file_path.name}")

        result = inspect_temporal_anomalies(file_path)
        results.append(result)

        print(f"Total rows: {result['total_rows']:,}")
        print(
            f"Expected period violations: "
            f"{result['outside_period']:,} "
            f"({result['outside_percent']:.4f}%)"
        )
        print(f"  Before month: {result['before_period']:,}")
        print(f"  After month: {result['after_period']:,}")
        print(f"Missing timestamps: {result['missing_dates']:,}")

        print(
            f"Actual min/max: "
            f"{result['min_date']} — {result['max_date']}"
        )

        if result["abnormal_years"]:
            print("Abnormal years:")
            for year, count in sorted(
                result["abnormal_years"].items()
            ):
                print(f"  {year}: {count:,}")

    print("\n" + "=" * 70)
    print("OVERALL TEMPORAL SUMMARY")
    print("=" * 70)

    total_rows = sum(item["total_rows"] for item in results)
    total_outside = sum(
        item["outside_period"] for item in results
    )

    print(f"Total rows: {total_rows:,}")
    print(f"Rows outside their expected month: {total_outside:,}")
    print(
        f"Overall share: "
        f"{total_outside / total_rows * 100:.4f}%"
    )

    compare_schemas(FILES)

    print("\nANOMALY CHECK COMPLETED")


if __name__ == "__main__":
    main()