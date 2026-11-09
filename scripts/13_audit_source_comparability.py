
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "taxi"
LOOKUP_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "reference"
    / "taxi_zone_lookup.csv"
)

TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
BATCH_SIZE = 250_000

YEARS = [2024, 2025]


def get_month_files() -> list[Path]:
    files = []

    for year in YEARS:
        for month in range(1, 13):
            path = RAW_DIR / f"yellow_tripdata_{year}-{month:02d}.parquet"

            if not path.exists():
                raise FileNotFoundError(f"Missing source file: {path}")

            files.append(path)

    return files


def load_zone_lookup() -> dict:
    lookup = pd.read_csv(LOOKUP_PATH)

    return (
        lookup.drop_duplicates("LocationID")
        .set_index("LocationID")["Borough"]
        .to_dict()
    )


def inspect_file(path: Path, zone_mapping: dict) -> tuple[dict, list, dict]:
    year = int(path.stem.split("_")[-1][:4])
    month = int(path.stem.split("-")[-1])

    month_start = pd.Timestamp(year=year, month=month, day=1)
    next_month = month_start + pd.DateOffset(months=1)

    parquet_file = pq.ParquetFile(path)

    result = {
        "year": year,
        "month": month,
        "file": path.name,
        "raw_rows": parquet_file.metadata.num_rows,
        "valid_month_rows": 0,
        "excluded_date_rows": 0,
        "missing_pickup_datetime": 0,
        "missing_pickup_zone": 0,
        "unique_zones": set(),
    }

    borough_counts = defaultdict(int)

    schema = {
        field.name: str(field.type)
        for field in parquet_file.schema_arrow
    }

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["tpep_pickup_datetime", "PULocationID"],
    ):
        df = batch.to_pandas()

        pickup = pd.to_datetime(
            df["tpep_pickup_datetime"],
            errors="coerce",
        )

        zone = pd.to_numeric(
            df["PULocationID"],
            errors="coerce",
        )

        missing_datetime = pickup.isna()
        missing_zone = zone.isna()

        result["missing_pickup_datetime"] += int(missing_datetime.sum())
        result["missing_pickup_zone"] += int(missing_zone.sum())

        in_month = (
            pickup.ge(month_start)
            & pickup.lt(next_month)
            & pickup.notna()
        )

        result["valid_month_rows"] += int(in_month.sum())
        result["excluded_date_rows"] += int((~in_month).sum())

        valid = df.loc[in_month].copy()
        valid_zones = zone.loc[in_month]

        result["unique_zones"].update(
            valid_zones.dropna().astype(int).unique().tolist()
        )

        for zone_id, count in valid_zones.dropna().value_counts().items():
            borough = zone_mapping.get(int(zone_id), "NOT_IN_LOOKUP")
            borough_counts[borough] += int(count)

    result["unique_zones"] = len(result["unique_zones"])

    return result, schema, dict(borough_counts)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TLC SOURCE DATA COMPARABILITY AUDIT")
    print("=" * 70)

    zone_mapping = load_zone_lookup()
    files = get_month_files()

    monthly_results = []
    schemas = {}
    borough_results = []

    for index, path in enumerate(files, start=1):
        print(f"\n[{index}/{len(files)}] Processing {path.name}")

        result, schema, borough_counts = inspect_file(
            path,
            zone_mapping,
        )

        monthly_results.append(result)
        schemas[path.name] = schema

        for borough, trips in borough_counts.items():
            borough_results.append({
                "year": result["year"],
                "month": result["month"],
                "Borough": borough,
                "trip_count": trips,
            })

        print(
            f"Raw rows: {result['raw_rows']:,} | "
            f"Valid month rows: {result['valid_month_rows']:,} | "
            f"Excluded: {result['excluded_date_rows']:,}"
        )

    monthly = pd.DataFrame(monthly_results)

    monthly["excluded_pct"] = (
        monthly["excluded_date_rows"]
        / monthly["raw_rows"]
        * 100
    )

    monthly["year_month"] = (
        monthly["year"].astype(str)
        + "-"
        + monthly["month"].astype(str).str.zfill(2)
    )

    boroughs = pd.DataFrame(borough_results)

    # ---------------------------------------------------------
    # 1. MONTHLY SOURCE COUNTS
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("1. MONTHLY SOURCE COUNTS")
    print("=" * 70)

    print(
        monthly[
            [
                "year_month",
                "raw_rows",
                "valid_month_rows",
                "excluded_date_rows",
                "excluded_pct",
                "unique_zones",
            ]
        ].to_string(index=False)
    )

    monthly.to_csv(
        TABLES_DIR / "source_monthly_audit.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 2. YEARLY COMPARISON
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("2. YEARLY COMPARISON")
    print("=" * 70)

    yearly = (
        monthly.groupby("year")
        .agg(
            raw_rows=("raw_rows", "sum"),
            valid_month_rows=("valid_month_rows", "sum"),
            excluded_rows=("excluded_date_rows", "sum"),
        )
        .reset_index()
    )

    yearly["excluded_pct"] = (
        yearly["excluded_rows"] / yearly["raw_rows"] * 100
    )

    print(yearly.to_string(index=False))

    yearly.to_csv(
        TABLES_DIR / "source_yearly_audit.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 3. MONTHLY YEAR-OVER-YEAR COMPARISON
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("3. MONTHLY YEAR-OVER-YEAR COMPARISON")
    print("=" * 70)

    monthly_comparison = monthly.pivot(
        index="month",
        columns="year",
        values="valid_month_rows",
    ).reset_index()

    monthly_comparison["growth_pct"] = (
        (
            monthly_comparison[2025]
            - monthly_comparison[2024]
        )
        / monthly_comparison[2024]
        * 100
    )

    print(monthly_comparison.to_string(index=False))

    monthly_comparison.to_csv(
        TABLES_DIR / "source_monthly_yoy_comparison.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 4. BOROUGH COMPARISON
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("4. BOROUGH DISTRIBUTION")
    print("=" * 70)

    borough_yearly = (
        boroughs.groupby(["year", "Borough"])["trip_count"]
        .sum()
        .reset_index()
    )

    borough_yearly["year_total"] = (
        borough_yearly.groupby("year")["trip_count"]
        .transform("sum")
    )

    borough_yearly["share_pct"] = (
        borough_yearly["trip_count"]
        / borough_yearly["year_total"]
        * 100
    )

    print(borough_yearly.to_string(index=False))

    borough_yearly.to_csv(
        TABLES_DIR / "source_borough_yearly_audit.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 5. SCHEMA COMPARISON
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("5. SCHEMA COMPARISON")
    print("=" * 70)

    schema_2024 = schemas["yellow_tripdata_2024-01.parquet"]
    schema_2025 = schemas["yellow_tripdata_2025-01.parquet"]

    all_columns = sorted(set(schema_2024) | set(schema_2025))

    schema_comparison = []

    for column in all_columns:
        type_2024 = schema_2024.get(column, "NOT_PRESENT")
        type_2025 = schema_2025.get(column, "NOT_PRESENT")

        status = (
            "SAME"
            if type_2024 == type_2025
            else "CHANGED"
        )

        schema_comparison.append({
            "column": column,
            "2024_type": type_2024,
            "2025_type": type_2025,
            "status": status,
        })

    schema_df = pd.DataFrame(schema_comparison)

    print(schema_df.to_string(index=False))

    schema_df.to_csv(
        TABLES_DIR / "source_schema_comparison.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 6. FINAL CHECKS
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("6. FINAL INTEGRITY CHECKS")
    print("=" * 70)

    print(
        "Total valid source records:",
        f"{monthly['valid_month_rows'].sum():,}",
    )

    print(
        "Total excluded records:",
        f"{monthly['excluded_date_rows'].sum():,}",
    )

    print(
        "Missing pickup timestamps:",
        f"{monthly['missing_pickup_datetime'].sum():,}",
    )

    print(
        "Missing pickup zone IDs:",
        f"{monthly['missing_pickup_zone'].sum():,}",
    )

    print(
        "Total unique zones across files:",
        len(
            set().union(
                *[
                    set()
                    for _ in []
                ]
            )
        ) if False else monthly["unique_zones"].max(),
    )

    print("\nAudit completed.")
    print(f"Results saved to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
