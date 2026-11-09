
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

BOROUGHS = ["Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Island"]


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["date"] = df["datetime"].dt.date

    return df


def analyze_borough_shares(df: pd.DataFrame) -> pd.DataFrame:
    print("\n1. MONTHLY BOROUGH SHARES")

    data = df[df["Borough"].isin(BOROUGHS)]

    monthly = (
        data.groupby(["year", "month", "Borough"])
        ["trip_count"]
        .sum()
        .reset_index()
    )

    monthly["city_total"] = (
        monthly.groupby(["year", "month"])["trip_count"]
        .transform("sum")
    )

    monthly["share_pct"] = (
        monthly["trip_count"] / monthly["city_total"] * 100
    )

    pivot = monthly.pivot(
        index=["month", "Borough"],
        columns="year",
        values="share_pct",
    ).reset_index()

    pivot["share_change_pp"] = pivot[2025] - pivot[2024]

    print(pivot.to_string(index=False))

    pivot.to_csv(
        TABLES_DIR / "borough_share_comparison.csv",
        index=False,
    )

    return monthly


def analyze_active_zones(df: pd.DataFrame) -> pd.DataFrame:
    print("\n2. ACTIVE ZONES BY MONTH")

    data = df[df["Borough"].isin(BOROUGHS)].copy()

    active = (
        data[data["trip_count"] > 0]
        .groupby(["year", "month", "Borough"])
        .agg(
            active_zones=("zone_id", "nunique"),
            total_trips=("trip_count", "sum"),
        )
        .reset_index()
    )

    days = (
        data[["year", "month", "datetime"]]
        .drop_duplicates()
        .groupby(["year", "month"])
        .size()
        .reset_index(name="hours")
    )

    days["days"] = days["hours"] / 24

    active = active.merge(
        days[["year", "month", "days"]],
        on=["year", "month"],
        validate="many_to_one",
    )

    active["trips_per_active_zone_per_day"] = (
        active["total_trips"]
        / active["active_zones"]
        / active["days"]
    )

    print(active.to_string(index=False))

    active.to_csv(
        TABLES_DIR / "active_zones_comparison.csv",
        index=False,
    )

    return active


def analyze_zone_distribution(df: pd.DataFrame) -> pd.DataFrame:
    print("\n3. ZONE-LEVEL ANNUAL DISTRIBUTION")

    data = df[df["Borough"].isin(["Brooklyn", "Bronx"])]

    yearly = (
        data.groupby(["Borough", "zone_id", "Zone", "year"])
        ["trip_count"]
        .sum()
        .unstack("year")
        .reset_index()
    )

    yearly["absolute_change"] = yearly[2025] - yearly[2024]

    yearly["growth_pct"] = (
        yearly["absolute_change"] / yearly[2024] * 100
    )

    print("\nBrooklyn:")
    print(
        yearly[yearly["Borough"] == "Brooklyn"]
        .sort_values("absolute_change", ascending=False)
        .head(15)
        .to_string(index=False)
    )

    print("\nBronx:")
    print(
        yearly[yearly["Borough"] == "Bronx"]
        .sort_values("absolute_change", ascending=False)
        .head(15)
        .to_string(index=False)
    )

    yearly.to_csv(
        TABLES_DIR / "zone_absolute_growth.csv",
        index=False,
    )

    return yearly


def plot_borough_shares(monthly: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 8))

    sns.lineplot(
        data=monthly,
        x="month",
        y="share_pct",
        hue="Borough",
        style="year",
        markers=True,
        dashes=True,
        errorbar=None,
    )

    plt.title("Monthly Borough Share of Yellow Taxi Trips")
    plt.xlabel("Month")
    plt.ylabel("Share of Citywide Trips, %")
    plt.xticks(range(1, 13))
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "borough_share_comparison.png",
        dpi=300,
    )

    plt.close()


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data()

    monthly = analyze_borough_shares(df)

    analyze_active_zones(df)

    analyze_zone_distribution(df)

    plot_borough_shares(monthly)

    print("\nAnalysis completed.")
    print(f"Tables: {TABLES_DIR}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
