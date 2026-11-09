
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "historical_demand_with_zones.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "tables"

REQUIRED_COLUMNS = [
    "zone_id",
    "datetime",
    "trip_count",
    "Borough",
    "Zone",
]


def load_data() -> pd.DataFrame:
    """Загрузка и базовая проверка исторических данных."""
    print("Loading historical demand dataset...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH)

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month

    df = df[df["year"].isin([2024, 2025])].copy()

    print(f"Rows: {len(df):,}")
    print(f"Zones: {df['zone_id'].nunique()}")
    print(
        f"Period: {df['datetime'].min()} — "
        f"{df['datetime'].max()}"
    )

    return df



def calculate_zone_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Сравнение годового спроса по каждой зоне без потери специальных зон."""

    # Суммируем поездки отдельно по зоне и году.
    yearly = (
        df.groupby(
            ["zone_id", "year"],
            as_index=False,
        )["trip_count"]
        .sum()
    )

    # Формируем таблицу: одна строка на зону,
    # отдельные столбцы для 2024 и 2025 годов.
    pivot = yearly.pivot(
        index="zone_id",
        columns="year",
        values="trip_count",
    ).fillna(0).reset_index()

    pivot.columns.name = None

    for year in [2024, 2025]:
        if year not in pivot.columns:
            pivot[year] = 0

    pivot = pivot.rename(
        columns={
            2024: "trips_2024",
            2025: "trips_2025",
        }
    )

    # Добавляем географические названия отдельно.
    # Такой способ сохраняет зоны с пропущенными Borough или Zone.
    metadata = (
        df.sort_values("datetime")
        .drop_duplicates("zone_id")
        [["zone_id", "Borough", "Zone"]]
    )

    pivot = pivot.merge(
        metadata,
        on="zone_id",
        how="left",
        validate="one_to_one",
    )

    pivot["absolute_growth"] = (
        pivot["trips_2025"] - pivot["trips_2024"]
    )

    pivot["growth_pct"] = np.where(
        pivot["trips_2024"] > 0,
        (
            pivot["absolute_growth"]
            / pivot["trips_2024"]
        ) * 100,
        np.nan,
    )

    total_net_growth = pivot["absolute_growth"].sum()

    pivot["share_of_total_growth_pct"] = np.where(
        total_net_growth != 0,
        (
            pivot["absolute_growth"]
            / total_net_growth
        ) * 100,
        0,
    )

    return pivot.sort_values(
        "absolute_growth",
        ascending=False,
    )


def analyze_boroughs(df: pd.DataFrame) -> pd.DataFrame:
    """Сравнение динамики спроса по районам."""
    yearly = (
        df.groupby(
            ["Borough", "year"],
            dropna=False,
            as_index=False,
        )["trip_count"]
        .sum()
    )

    pivot = yearly.pivot_table(
        index="Borough",
        columns="year",
        values="trip_count",
        aggfunc="sum",
        fill_value=0,
        dropna=False,
    ).reset_index()

    for year in [2024, 2025]:
        if year not in pivot.columns:
            pivot[year] = 0

    pivot = pivot.rename(
        columns={
            2024: "trips_2024",
            2025: "trips_2025",
        }
    )

    pivot["absolute_growth"] = (
        pivot["trips_2025"] - pivot["trips_2024"]
    )

    pivot["growth_pct"] = np.where(
        pivot["trips_2024"] > 0,
        (
            pivot["absolute_growth"]
            / pivot["trips_2024"]
        ) * 100,
        np.nan,
    )

    total_growth = pivot["absolute_growth"].sum()

    pivot["contribution_to_growth_pct"] = (
        pivot["absolute_growth"] / total_growth * 100
        if total_growth != 0
        else 0
    )

    return pivot.sort_values(
        "absolute_growth",
        ascending=False,
    )


def analyze_monthly_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Сравнение помесячного спроса по районам."""
    monthly = (
        df.groupby(
            ["Borough", "year", "month"],
            dropna=False,
            as_index=False,
        )["trip_count"]
        .sum()
    )

    pivot = monthly.pivot_table(
        index=["Borough", "month"],
        columns="year",
        values="trip_count",
        aggfunc="sum",
        fill_value=0,
        dropna=False,
    ).reset_index()

    for year in [2024, 2025]:
        if year not in pivot.columns:
            pivot[year] = 0

    pivot = pivot.rename(
        columns={
            2024: "trips_2024",
            2025: "trips_2025",
        }
    )

    pivot["absolute_growth"] = (
        pivot["trips_2025"] - pivot["trips_2024"]
    )

    pivot["growth_pct"] = np.where(
        pivot["trips_2024"] > 0,
        (
            pivot["absolute_growth"]
            / pivot["trips_2024"]
        ) * 100,
        np.nan,
    )

    return pivot.sort_values(["Borough", "month"])


def analyze_growth_concentration(
    zone_growth: pd.DataFrame,
) -> pd.DataFrame:
    """Оценка концентрации общего прироста по зонам."""
    positive = zone_growth[
        zone_growth["absolute_growth"] > 0
    ].copy()

    positive = positive.sort_values(
        "absolute_growth",
        ascending=False,
    )

    total_growth = positive["absolute_growth"].sum()

    positive["cumulative_growth_pct"] = (
        positive["absolute_growth"].cumsum()
        / total_growth
        * 100
    )

    thresholds = []

    for threshold in [50, 80, 90]:
        count = (
            positive["cumulative_growth_pct"] < threshold
        ).sum() + 1

        thresholds.append(
            {
                "threshold_pct": threshold,
                "zones_required": min(
                    count,
                    len(positive),
                ),
            }
        )

    return pd.DataFrame(thresholds)


def save_results(
    zone_growth: pd.DataFrame,
    borough_growth: pd.DataFrame,
    monthly_growth: pd.DataFrame,
    concentration: pd.DataFrame,
) -> None:
    """Сохранение аналитических таблиц."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    zone_growth.to_csv(
        OUTPUT_DIR / "geographical_zone_growth.csv",
        index=False,
    )

    borough_growth.to_csv(
        OUTPUT_DIR / "geographical_borough_growth.csv",
        index=False,
    )

    monthly_growth.to_csv(
        OUTPUT_DIR / "geographical_monthly_growth.csv",
        index=False,
    )

    concentration.to_csv(
        OUTPUT_DIR / "geographical_growth_concentration.csv",
        index=False,
    )

    print(f"\nTables saved to: {OUTPUT_DIR}")


def main() -> None:
    df = load_data()

    print("\n" + "=" * 70)
    print("1. GROWTH BY TAXI ZONE")
    print("=" * 70)

    zone_growth = calculate_zone_growth(df)

    print("\nTop 15 zones by absolute growth:")
    print(
        zone_growth[
            [
                "zone_id",
                "Borough",
                "Zone",
                "trips_2024",
                "trips_2025",
                "absolute_growth",
                "growth_pct",
                "share_of_total_growth_pct",
            ]
        ].head(15).to_string(index=False)
    )

    print("\nZones with the largest percentage growth:")
    print(
        zone_growth[
            zone_growth["trips_2024"] >= 1000
        ]
        .sort_values("growth_pct", ascending=False)[
            [
                "zone_id",
                "Borough",
                "Zone",
                "trips_2024",
                "trips_2025",
                "growth_pct",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("2. GROWTH BY BOROUGH")
    print("=" * 70)

    borough_growth = analyze_boroughs(df)

    print(
        borough_growth.to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("3. MONTHLY GROWTH BY BOROUGH")
    print("=" * 70)

    monthly_growth = analyze_monthly_growth(df)

    for borough in [
        "Manhattan",
        "Queens",
        "Brooklyn",
        "Bronx",
    ]:
        subset = monthly_growth[
            monthly_growth["Borough"] == borough
        ]

        print(f"\n{borough}:")
        print(
            subset[
                [
                    "month",
                    "trips_2024",
                    "trips_2025",
                    "growth_pct",
                ]
            ].to_string(index=False)
        )

    print("\n" + "=" * 70)
    print("4. GROWTH CONCENTRATION")
    print("=" * 70)

    concentration = analyze_growth_concentration(
        zone_growth
    )

    print(concentration.to_string(index=False))

    positive_zones = zone_growth[
        zone_growth["absolute_growth"] > 0
    ].sort_values(
        "absolute_growth",
        ascending=False,
    )

    total_growth = positive_zones["absolute_growth"].sum()

    for n in [5, 10, 20, 30]:
        share = (
            positive_zones.head(n)["absolute_growth"].sum()
            / total_growth
            * 100
        )

        print(
            f"Top {n} zones account for "
            f"{share:.2f}% of positive growth"
        )

    print("\n" + "=" * 70)
    print("5. DATA QUALITY CHECK")
    print("=" * 70)

    print(
        "Total trips 2024:",
        f"{df.loc[df['year'] == 2024, 'trip_count'].sum():,}",
    )

    print(
        "Total trips 2025:",
        f"{df.loc[df['year'] == 2025, 'trip_count'].sum():,}",
    )

    print(
        "Unique zones:",
        df["zone_id"].nunique(),
    )

    print(
        "Rows with missing Borough:",
        df["Borough"].isna().sum(),
    )

    print(
        "Rows with missing Zone:",
        df["Zone"].isna().sum(),
    )

    save_results(
        zone_growth,
        borough_growth,
        monthly_growth,
        concentration,
    )

    print("\nGeographical investigation completed.")


if __name__ == "__main__":
    main()
