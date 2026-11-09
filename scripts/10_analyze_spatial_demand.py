
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

OUTPUT_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"


def prepare_output_directories() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    print("Loading enriched historical demand dataset...")

    df = pd.read_parquet(DATA_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])

    print(f"Rows: {len(df):,}")
    print(f"Zones: {df['zone_id'].nunique()}")
    print(
        f"Period: {df['datetime'].min()} — "
        f"{df['datetime'].max()}"
    )

    return df


def analyze_borough_demand(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("1. DEMAND BY BOROUGH")
    print("=" * 70)

    borough = (
        df.groupby("Borough", dropna=False)
        .agg(
            total_trips=("trip_count", "sum"),
            mean_hourly_demand=("trip_count", "mean"),
            median_hourly_demand=("trip_count", "median"),
            max_hourly_demand=("trip_count", "max"),
            zones=("zone_id", "nunique"),
        )
        .sort_values("total_trips", ascending=False)
        .reset_index()
    )

    borough["share_pct"] = (
        borough["total_trips"] / borough["total_trips"].sum() * 100
    )

    print(borough.to_string(index=False))

    borough.to_csv(
        TABLES_DIR / "demand_by_borough.csv",
        index=False,
    )

    return borough


def analyze_top_zones(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("2. TOP 20 TAXI ZONES")
    print("=" * 70)

    zones = (
        df.groupby(
            ["zone_id", "Borough", "Zone"],
            dropna=False,
        )
        .agg(
            total_trips=("trip_count", "sum"),
            mean_hourly_demand=("trip_count", "mean"),
            max_hourly_demand=("trip_count", "max"),
        )
        .sort_values("total_trips", ascending=False)
        .reset_index()
    )

    total_trips = zones["total_trips"].sum()

    zones["share_pct"] = (
        zones["total_trips"] / total_trips * 100
    )

    zones["cumulative_share_pct"] = (
        zones["share_pct"].cumsum()
    )

    print(zones.head(20).to_string(index=False))

    zones.to_csv(
        TABLES_DIR / "demand_by_zone.csv",
        index=False,
    )

    return zones


def analyze_demand_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("3. MONTHLY DEMAND DYNAMICS")
    print("=" * 70)

    df = df.copy()
    df["month"] = df["datetime"].dt.to_period("M").astype(str)

    monthly = (
        df.groupby(["month", "Borough"], dropna=False)
        .agg(total_trips=("trip_count", "sum"))
        .reset_index()
    )

    print(monthly.head(20).to_string(index=False))

    monthly.to_csv(
        TABLES_DIR / "monthly_demand_by_borough.csv",
        index=False,
    )

    return monthly


def analyze_demand_concentration(zones: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("4. DEMAND CONCENTRATION")
    print("=" * 70)

    valid_zones = zones.dropna(subset=["Borough", "Zone"]).copy()
    valid_zones = valid_zones[
        ~valid_zones["Borough"].isin(["Unknown", "EWR"])
    ]

    valid_zones = valid_zones.sort_values(
        "total_trips",
        ascending=False,
    )

    total = valid_zones["total_trips"].sum()

    for n in [5, 10, 20, 50]:
        share = (
            valid_zones.head(n)["total_trips"].sum()
            / total
            * 100
        )

        print(f"Top {n} zones account for {share:.2f}% of trips.")

    # Доля зон, на которые приходится 80% спроса
    cumulative = valid_zones["total_trips"].cumsum() / total

    zones_for_80 = (cumulative < 0.80).sum() + 1

    print(
        f"Number of zones accounting for approximately "
        f"80% of demand: {zones_for_80}"
    )


def plot_borough_demand(borough: pd.DataFrame) -> None:
    plot_data = borough.dropna(subset=["Borough"])

    plt.figure(figsize=(12, 7))

    sns.barplot(
        data=plot_data,
        x="Borough",
        y="total_trips",
        hue="Borough",
        legend=False,
        errorbar=None,
    )

    plt.title("Total Taxi Demand by Borough (2024–2025)")
    plt.xlabel("Borough")
    plt.ylabel("Number of Trips")
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "demand_by_borough.png",
        dpi=300,
    )
    plt.close()


def plot_top_zones(zones: pd.DataFrame) -> None:
    plot_data = zones.head(20).sort_values("total_trips")

    plt.figure(figsize=(13, 9))

    sns.barplot(
        data=plot_data,
        x="total_trips",
        y="Zone",
        hue="Borough",
        errorbar=None,
    )

    plt.title("Top 20 Taxi Zones by Total Demand")
    plt.xlabel("Number of Trips")
    plt.ylabel("Taxi Zone")
    plt.legend(title="Borough")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "top_20_taxi_zones.png",
        dpi=300,
    )
    plt.close()


def plot_monthly_dynamics(monthly: pd.DataFrame) -> None:
    plot_data = monthly.dropna(subset=["Borough"])

    plt.figure(figsize=(15, 8))

    sns.lineplot(
        data=plot_data,
        x="month",
        y="total_trips",
        hue="Borough",
        marker="o",
        errorbar=None,
    )

    plt.title("Monthly Taxi Demand Dynamics by Borough")
    plt.xlabel("Month")
    plt.ylabel("Number of Trips")
    plt.xticks(rotation=45)
    plt.legend(title="Borough")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "monthly_demand_by_borough.png",
        dpi=300,
    )
    plt.close()


def plot_demand_concentration(zones: pd.DataFrame) -> None:
    plot_data = zones.dropna(subset=["Borough", "Zone"]).copy()

    plot_data = plot_data[
        ~plot_data["Borough"].isin(["Unknown", "EWR"])
    ]

    plot_data = plot_data.sort_values(
        "total_trips",
        ascending=False,
    )

    plot_data["cumulative_share_pct"] = (
        plot_data["total_trips"].cumsum()
        / plot_data["total_trips"].sum()
        * 100
    )

    plt.figure(figsize=(12, 7))

    plt.plot(
        range(1, len(plot_data) + 1),
        plot_data["cumulative_share_pct"],
        marker=".",
    )

    plt.axhline(
        80,
        linestyle="--",
        label="80% demand threshold",
    )

    plt.title("Cumulative Concentration of Taxi Demand")
    plt.xlabel("Number of Taxi Zones (ranked by demand)")
    plt.ylabel("Cumulative Share of Trips, %")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "demand_concentration.png",
        dpi=300,
    )
    plt.close()


def main() -> None:
    prepare_output_directories()

    df = load_data()

    borough = analyze_borough_demand(df)
    zones = analyze_top_zones(df)
    monthly = analyze_demand_dynamics(df)

    analyze_demand_concentration(zones)

    print("\n" + "=" * 70)
    print("5. BUILDING VISUALIZATIONS")
    print("=" * 70)

    plot_borough_demand(borough)
    plot_top_zones(zones)
    plot_monthly_dynamics(monthly)
    plot_demand_concentration(zones)

    print(f"\nTables saved to: {TABLES_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("\nSpatial analysis completed successfully.")


if __name__ == "__main__":
    main()
