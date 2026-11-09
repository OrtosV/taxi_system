from pathlib import Path
from collections import Counter

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "taxi"
REPORT_DIR = PROJECT_ROOT / "reports"

EXPECTED_FILES = [
    f"yellow_tripdata_{year}-{month:02d}.parquet"
    for year in range(2024, 2026)
    for month in range(1, 13)
]

BATCH_SIZE = 100_000

REQUIRED_COLUMNS = [
    "tpep_pickup_datetime",
    "PULocationID",
]


def inspect_file(file_path: Path) -> dict:
    """Проверяет один Parquet-файл."""

    parquet_file = pq.ParquetFile(file_path)

    schema = parquet_file.schema_arrow
    columns = schema.names

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    total_rows = parquet_file.metadata.num_rows

    pickup_min = None
    pickup_max = None

    missing_pickup = 0
    missing_zone = 0

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=REQUIRED_COLUMNS,
    ):
        df = batch.to_pandas()

        pickup = df["tpep_pickup_datetime"]
        zone = df["PULocationID"]

        missing_pickup += pickup.isna().sum()
        missing_zone += zone.isna().sum()

        batch_min = pickup.min()
        batch_max = pickup.max()

        if pd.notna(batch_min):
            if pickup_min is None or batch_min < pickup_min:
                pickup_min = batch_min

        if pd.notna(batch_max):
            if pickup_max is None or batch_max > pickup_max:
                pickup_max = batch_max

    return {
        "file": file_path.name,
        "rows": total_rows,
        "size_mb": round(file_path.stat().st_size / (1024 ** 2), 2),
        "columns": columns,
        "schema": {
            field.name: str(field.type)
            for field in schema
        },
        "pickup_min": pickup_min,
        "pickup_max": pickup_max,
        "missing_pickup": missing_pickup,
        "missing_zone": missing_zone,
    }


def main() -> None:
    print("=" * 70)
    print("NYC TLC DATA INVENTORY")
    print("=" * 70)

    results = []
    missing_files = []
    errors = []

    for file_name in EXPECTED_FILES:
        file_path = DATA_DIR / file_name

        if not file_path.exists():
            missing_files.append(file_name)
            print(f"[MISSING] {file_name}")
            continue

        print(f"[CHECK] {file_name}")

        try:
            result = inspect_file(file_path)
            results.append(result)

            print(
                f"  Rows: {result['rows']:,} | "
                f"Size: {result['size_mb']:.2f} MB | "
                f"Pickup: {result['pickup_min']} — "
                f"{result['pickup_max']}"
            )

        except Exception as error:
            errors.append((file_name, str(error)))
            print(f"[ERROR] {file_name}: {error}")

    print("\n" + "=" * 70)
    print("INVENTORY SUMMARY")
    print("=" * 70)

    total_rows = sum(item["rows"] for item in results)
    total_size = sum(item["size_mb"] for item in results)

    print(f"Expected files: {len(EXPECTED_FILES)}")
    print(f"Successfully inspected: {len(results)}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Files with errors: {len(errors)}")
    print(f"Total rows: {total_rows:,}")
    print(f"Total size: {total_size / 1024:.2f} GB")

    if results:
        print("\nMONTHLY RECORD COUNTS")
        print("-" * 70)

        for item in results:
            print(
                f"{item['file']}: "
                f"{item['rows']:,} rows"
            )

        print("\nTEMPORAL COVERAGE")
        print("-" * 70)

        print(
            "Earliest pickup:",
            min(item["pickup_min"] for item in results
                if item["pickup_min"] is not None),
        )

        print(
            "Latest pickup:",
            max(item["pickup_max"] for item in results
                if item["pickup_max"] is not None),
        )

        print("\nMISSING VALUES IN KEY COLUMNS")
        print("-" * 70)

        print(
            "Missing pickup timestamps:",
            sum(item["missing_pickup"] for item in results),
        )

        print(
            "Missing pickup zone IDs:",
            sum(item["missing_zone"] for item in results),
        )

        # Сравниваем схемы всех файлов.
        print("\nSCHEMA CONSISTENCY")
        print("-" * 70)

        reference_schema = results[0]["schema"]
        inconsistent_files = []

        for item in results[1:]:
            if item["schema"] != reference_schema:
                inconsistent_files.append(item["file"])

        if inconsistent_files:
            print("Different schemas detected:")
            for file_name in inconsistent_files:
                print(f"  {file_name}")
        else:
            print("All inspected files have identical schemas.")

        # Сохраняем краткий отчёт.
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        report = pd.DataFrame([
            {
                "file": item["file"],
                "rows": item["rows"],
                "size_mb": item["size_mb"],
                "pickup_min": item["pickup_min"],
                "pickup_max": item["pickup_max"],
                "missing_pickup": item["missing_pickup"],
                "missing_zone": item["missing_zone"],
            }
            for item in results
        ])

        report_path = REPORT_DIR / "taxi_data_inventory.csv"
        report.to_csv(report_path, index=False)

        print(f"\nInventory report saved to: {report_path}")

    if missing_files or errors:
        print("\nATTENTION: Some files require investigation.")


if __name__ == "__main__":
    main()