from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "taxi_hourly_demand_jan2025.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"


def analyze_target(data_path: Path) -> None:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    df = pd.read_parquet(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])

    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.day_of_week
    df["date"] = df["datetime"].dt.date

    weekday_names = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }

    df["weekday_name"] = df["weekday"].map(weekday_names)

    # 1. Общий почасовой спрос по всему городу.
    hourly_city = (
        df.groupby("datetime")["trip_count"]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(15, 5))
    sns.lineplot(
        data=hourly_city,
        x="datetime",
        y="trip_count",
    )
    plt.title("Hourly taxi demand in NYC — January 2025")
    plt.xlabel("Date")
    plt.ylabel("Number of trips")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_city_hourly_demand.png", dpi=150)
    plt.close()

    # 2. Средний спрос по часам суток.
    hourly_profile = (
        df.groupby("hour")["trip_count"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(11, 5))
    sns.barplot(
        data=hourly_profile,
        x="hour",
        y="trip_count",
    )
    plt.title("Average demand by hour of day")
    plt.xlabel("Hour")
    plt.ylabel("Average trips per zone")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_demand_by_hour.png", dpi=150)
    plt.close()

    # 3. Тепловая карта: день недели × час суток.
    weekday_hour = (
        df.groupby(["weekday", "hour"])["trip_count"]
        .mean()
        .unstack()
    )

    weekday_hour.index = [
        weekday_names[index] for index in weekday_hour.index
    ]

    plt.figure(figsize=(15, 6))
    sns.heatmap(
        weekday_hour,
        cmap="YlOrRd",
        annot=False,
    )
    plt.title("Average taxi demand: weekday vs hour")
    plt.xlabel("Hour of day")
    plt.ylabel("Day of week")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_weekday_hour_heatmap.png", dpi=150)
    plt.close()

    # 4. Топ-15 зон по суммарному спросу.
    zone_demand = (
        df.groupby("zone_id")["trip_count"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .sort_values()
        .reset_index()
    )

    plt.figure(figsize=(11, 7))
    sns.barplot(
        data=zone_demand,
        x="trip_count",
        y="zone_id",
    )
    plt.title("Top 15 taxi zones by total demand")
    plt.xlabel("Total number of trips")
    plt.ylabel("Zone ID")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_top_zones.png", dpi=150)
    plt.close()

    # 5. Распределение количества поездок в одной зоне за час.
    positive_demand = df.loc[df["trip_count"] > 0, "trip_count"]

    plt.figure(figsize=(11, 5))
    sns.histplot(
        positive_demand,
        bins=60,
    )
    plt.title("Distribution of positive hourly demand")
    plt.xlabel("Number of trips")
    plt.ylabel("Number of zone-hour observations")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_demand_distribution.png", dpi=150)
    plt.close()

    # 6. Доля нулевых наблюдений по каждой зоне.
    zero_share = (
        df.assign(is_zero=df["trip_count"] == 0)
        .groupby("zone_id")["is_zero"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .head(15)
        .sort_values()
        .reset_index(name="zero_share")
    )

    plt.figure(figsize=(11, 7))

    sns.barplot(
        data=zero_share,
        x="zero_share",
        y="zone_id",
        errorbar=None,
    )

    plt.title("Top 15 zones by share of zero-demand hours")
    plt.xlabel("Zero-demand hours, %")
    plt.ylabel("Zone ID")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "06_zero_demand_by_zone.png",
        dpi=150,
    )
    plt.close()

    print("=" * 60)
    print("TARGET ANALYSIS")
    print("=" * 60)

    print("\nAverage city demand by hour:")
    print(
        hourly_city.groupby(hourly_city["datetime"].dt.hour)[
            "trip_count"
        ].mean().round(2).to_string()
    )

    print("\nAverage demand by weekday:")
    print(
        df.groupby("weekday_name")["trip_count"]
        .mean()
        .round(2)
        .to_string()
    )

    print("\nDemand by weekday (total trips):")
    print(
        df.groupby("weekday_name")["trip_count"]
        .sum()
        .to_string()
    )

    print("\nPlots saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    analyze_target(DATA_PATH)