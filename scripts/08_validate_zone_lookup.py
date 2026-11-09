
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOOKUP_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "reference"
    / "taxi_zone_lookup.csv"
)

TARGET_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "historical_hourly_demand.parquet"
)


def main() -> None:
    print("=" * 70)
    print("NYC TAXI ZONE LOOKUP VALIDATION")
    print("=" * 70)

    lookup = pd.read_csv(LOOKUP_PATH)
    target = pd.read_parquet(TARGET_PATH)

    zone_column = "zone_id"

    if zone_column not in target.columns:
        raise ValueError(
            f"Column '{zone_column}' not found. "
            f"Available columns: {list(target.columns)}"
        )

    # 1. Общая информация о справочнике
    print("\n1. LOOKUP SUMMARY")
    print(f"Rows: {len(lookup)}")
    print(f"Unique LocationID: {lookup['LocationID'].nunique()}")
    print(f"Missing values:\n{lookup.isna().sum()}")
    print(
        "Duplicate LocationID: "
        f"{lookup['LocationID'].duplicated().sum()}"
    )

    # 2. Распределение зон по районам
    print("\n2. ZONES BY BOROUGH")
    print(
        lookup.groupby("Borough", dropna=False)["LocationID"]
        .count()
        .sort_values(ascending=False)
        .to_string()
    )

    # 3. Сопоставление идентификаторов
    target_zones = set(target[zone_column].unique())
    lookup_zones = set(lookup["LocationID"].unique())

    missing_zones = target_zones - lookup_zones
    unused_zones = lookup_zones - target_zones

    print("\n3. ZONE ID VALIDATION")
    print(f"Unique zones in target: {len(target_zones)}")
    print(f"Unique zones in lookup: {len(lookup_zones)}")
    print(f"Zones missing in lookup: {len(missing_zones)}")
    print(f"Lookup zones not present in target: {len(unused_zones)}")

    if missing_zones:
        print(f"Missing IDs: {sorted(missing_zones)}")

    if unused_zones:
        print(f"Unused lookup IDs: {sorted(unused_zones)}")

    # 4. Специальные зоны
    print("\n4. SPECIAL ZONES")

    special_zones = lookup[
        lookup["Borough"].isin(["Unknown", "N/A"])
        | lookup["Zone"].str.contains(
            "Unknown|Outside",
            case=False,
            na=False
        )
    ]

    print(special_zones.to_string(index=False))

    # 5. Проверка объединения
    print("\n5. MERGE VALIDATION")

    merged = target.merge(
        lookup,
        left_on=zone_column,
        right_on="LocationID",
        how="left",
        validate="many_to_one",
        indicator=True
    )

    unmatched = (merged["_merge"] != "both").sum()

    print(f"Rows before merge: {len(target)}")
    print(f"Rows after merge: {len(merged)}")
    print(f"Unmatched rows: {unmatched}")

    print("\nValidation completed.")


if __name__ == "__main__":
    main()
