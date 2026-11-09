
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGET_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "historical_hourly_demand.parquet"
)

LOOKUP_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "reference"
    / "taxi_zone_lookup.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_PATH = (
    OUTPUT_DIR
    / "historical_demand_with_zones.parquet"
)


def main() -> None:
    print("=" * 70)
    print("HISTORICAL DEMAND DATA ENRICHMENT")
    print("=" * 70)

    # 1. Загрузка данных
    print("\n1. LOADING DATA")

    target = pd.read_parquet(TARGET_PATH)
    lookup = pd.read_csv(LOOKUP_PATH)

    print(f"Target rows: {len(target):,}")
    print(f"Lookup rows: {len(lookup):,}")

    # 2. Проверка структуры
    print("\n2. VALIDATING COLUMNS")

    required_target_columns = {
        "zone_id",
        "datetime",
        "trip_count",
    }

    required_lookup_columns = {
        "LocationID",
        "Borough",
        "Zone",
        "service_zone",
    }

    missing_target = required_target_columns - set(target.columns)
    missing_lookup = required_lookup_columns - set(lookup.columns)

    if missing_target:
        raise ValueError(
            f"Missing target columns: {sorted(missing_target)}"
        )

    if missing_lookup:
        raise ValueError(
            f"Missing lookup columns: {sorted(missing_lookup)}"
        )

    if lookup["LocationID"].duplicated().any():
        raise ValueError("Duplicate LocationID values in lookup.")

    # 3. Объединение таблиц
    print("\n3. MERGING DATASETS")

    enriched = target.merge(
        lookup,
        left_on="zone_id",
        right_on="LocationID",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    unmatched = (enriched["_merge"] != "both").sum()

    if unmatched > 0:
        raise ValueError(
            f"Found {unmatched} rows without a matching taxi zone."
        )

    # Удаляем технические столбцы объединения
    enriched = enriched.drop(
        columns=["LocationID", "_merge"]
    )

    print(f"Rows after merge: {len(enriched):,}")
    print(f"Unmatched rows: {unmatched}")

    # 4. Проверка целостности
    print("\n4. DATA INTEGRITY CHECK")

    if len(enriched) != len(target):
        raise ValueError(
            "Row count changed after enrichment."
        )

    if enriched["trip_count"].sum() != target["trip_count"].sum():
        raise ValueError(
            "Total trip count changed after enrichment."
        )

    print("Row count preserved.")
    print("Total trip count preserved.")

    # 5. Анализ добавленных признаков
    print("\n5. GEOGRAPHICAL FEATURES")

    print("\nRows by borough:")
    print(
        enriched.groupby(
            "Borough",
            dropna=False
        )["zone_id"]
        .nunique()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\nMissing values:")
    print(
        enriched[
            ["Borough", "Zone", "service_zone"]
        ].isna().sum().to_string()
    )

    # 6. Сохранение результата
    print("\n6. SAVING DATASET")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    enriched.to_parquet(
        OUTPUT_PATH,
        index=False,
        engine="pyarrow",
    )

    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Final rows: {len(enriched):,}")
    print(f"Final columns: {list(enriched.columns)}")

    print("\nFirst 10 rows:")
    print(enriched.head(10).to_string(index=False))

    print("\nEnrichment completed successfully.")


if __name__ == "__main__":
    main()
