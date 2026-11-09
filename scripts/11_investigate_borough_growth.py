
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "historical_demand_with_zones.parquet"
)

TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

TARGET_BOROUGHS = ["Brooklyn", "Bronx"]


def load_data() -> pd.DataFrame:
    print("Loading historical demand dataset...")

    df = pd.read_parquet(DATA_PATH)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["date"] = df["datetime"].dt.date

    print(f"Rows: {len(df):,}")
    print(f"Period: {df['datetime'].min()} — {df['datetime'].max()}")

    return df


def analyze_annual_growth(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("1. ANNUAL DEMAND COMPARISON")
    print("=" * 70)

    result = (
        df[df["Borough"].isin(TARGET_BOROUGHS)]
        .groupby(["Borough", "year"])
        .agg(total_trips=("trip_count", "sum"))
        .reset_index()
    )

    pivot = result.pivot(
        index="Borough",
        columns="year",
        values="total_trips",
    )

    pivot["growth_pct"] = (
        pivot[2025] / pivot[2024] - 1
    ) * 100

    print(pivot.to_string())

    pivot.to_csv(
        TABLES_DIR / "borough_annual_growth.csv"
    )


def analyze_monthly_growth(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("2. MONTHLY GROWTH COMPARISON")
    print("=" * 70)

    data = df[df["Borough"].isin(TARGET_BOROUGHS)].copy()

    monthly = (
        data.groupby(["Borough", "year", "month"])
        .agg(
            total_trips=("trip_count", "sum"),
        )
        .reset_index()
    )

    monthly["days_in_month"] = monthly.apply(
        lambda row: pd.Period(
            f"{int(row['year'])}-{int(row['month']):02d}"
        ).days_in_month,
        axis=1,
    )

    monthly["trips_per_day"] = (
        monthly["total_trips"] / monthly["days_in_month"]
    )

    pivot = monthly.pivot(
        index=["Borough", "month"],
        columns="year",
        values="trips_per_day",
    ).reset_index()

    pivot["growth_pct"] = (
        pivot[2025] / pivot[2024] - 1
    ) * 100

    print(pivot.to_string(index=False))

    pivot.to_csv(
        TABLES_DIR / "borough_monthly_growth.csv",
        index=False,
    )

    return pivot


def analyze_zone_growth(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("3. GROWTH BY INDIVIDUAL TAXI ZONE")
    print("=" * 70)

    data = df[
        df["Borough"].isin(TARGET_BOROUGHS)
    ].copy()

    yearly = (
        data.groupby(["Borough", "zone_id", "Zone", "year"])
        .agg(
            total_trips=("trip_count", "sum"),
        )
        .reset_index()
    )

    pivot = yearly.pivot(
        index=["Borough", "zone_id", "Zone"],
        columns="year",
        values="total_trips",
    ).reset_index()

    pivot = pivot.dropna(subset=[2024, 2025])

    pivot["growth_pct"] = (
        pivot[2025] / pivot[2024] - 1
    ) * 100

    pivot = pivot.sort_values(
        ["Borough", "growth_pct"],
        ascending=[True, False],
    )

    print("\nZone-level statistics:")
    print(
        pivot.groupby("Borough")["growth_pct"]
        .agg(["count", "median", "mean", "min", "max"])
        .to_string()
    )

    print("\nZones with the highest growth:")
    print(
        pivot.groupby("Borough", group_keys=False)
        .head(10)
        .to_string(index=False)
    )

    pivot.to_csv(
        TABLES_DIR / "borough_zone_growth.csv",
        index=False,
    )

    return pivot


def plot_monthly_growth(monthly: pd.DataFrame) -> None:
    plt.figure(figsize=(13, 7))

    sns.lineplot(
        data=monthly,
        x="month",
        y="growth_pct",
        hue="Borough",
        marker="o",
        errorbar=None,
    )

    plt.axhline(0, linestyle="--")

    plt.title(
        "Year-over-Year Taxi Demand Growth "
        "by Month (2025 vs 2024)"
    )
    plt.xlabel("Month")
    plt.ylabel("Growth, %")
    plt.xticks(range(1, 13))
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "borough_monthly_growth.png",
        dpi=300,
    )

    plt.close()


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    analyze_annual_growth(df)

    monthly = analyze_monthly_growth(df)

    analyze_zone_growth(df)

    plot_monthly_growth(monthly)

    print("\n" + "=" * 70)
    print("INVESTIGATION COMPLETED")
    print("=" * 70)

    print(f"Tables: {TABLES_DIR}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
