from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "taxi"
    / "yellow_tripdata_2025-01.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "taxi_hourly_demand_jan2025.parquet"
)

START_DATE = pd.Timestamp("2025-01-01 00:00:00")
END_DATE = pd.Timestamp("2025-02-01 00:00:00")


def build_hourly_demand(source_path: Path, output_path: Path) -> None:
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    print("Loading pickup timestamps and zone identifiers...")

    df = pd.read_parquet(
        source_path,
        columns=["tpep_pickup_datetime", "PULocationID"],
    )

    print(f"Loaded rows: {len(df):,}")

    # Удаляем только записи, в которых невозможно определить
    # время отправления или зону отправления.
    df = df.dropna(
        subset=["tpep_pickup_datetime", "PULocationID"]
    )

    # Оставляем только январь 2025 года.
    df = df[
        (df["tpep_pickup_datetime"] >= START_DATE)
        & (df["tpep_pickup_datetime"] < END_DATE)
    ].copy()

    print(f"Rows within January: {len(df):,}")

    # Приводим время отправления к часовым интервалам.
    df["hour"] = df["tpep_pickup_datetime"].dt.floor("h")

    # Считаем количество поездок для каждой пары
    # «зона отправления — час».
    demand = (
        df.groupby(["PULocationID", "hour"])
        .size()
        .reset_index(name="trip_count")
    )

    demand = demand.rename(
        columns={
            "PULocationID": "zone_id",
            "hour": "datetime",
        }
    )

    # Создаём полный временной диапазон:
    # 744 часа января 2025 года.
    hours = pd.date_range(
        start=START_DATE,
        end=END_DATE,
        freq="h",
        inclusive="left",
    )

    # Пока используем все зоны, которые встретились
    # в январских поездках.
    zones = sorted(demand["zone_id"].unique())

    # Формируем полную сетку «зона — час».
    full_index = pd.MultiIndex.from_product(
        [zones, hours],
        names=["zone_id", "datetime"],
    )

    demand = (
        demand.set_index(["zone_id", "datetime"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    demand["trip_count"] = demand["trip_count"].astype("int32")

    # Сортируем записи для удобства последующей обработки.
    demand = demand.sort_values(
        ["zone_id", "datetime"]
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    demand.to_parquet(output_path, index=False)

    print("\n" + "=" * 60)
    print("HOURLY DEMAND DATASET")
    print("=" * 60)

    print(f"Rows: {len(demand):,}")
    print(f"Unique zones: {demand['zone_id'].nunique()}")
    print(f"Unique hours: {demand['datetime'].nunique()}")
    print(f"Total trips: {demand['trip_count'].sum():,}")

    print("\nTRIP COUNT STATISTICS")
    print("-" * 60)
    print(demand["trip_count"].describe())

    print("\nTOP 10 BUSIEST ZONE-HOUR COMBINATIONS")
    print("-" * 60)
    print(
        demand.nlargest(10, "trip_count")[
            ["zone_id", "datetime", "trip_count"]
        ].to_string(index=False)
    )

    print("\nZONES WITH THE HIGHEST TOTAL DEMAND")
    print("-" * 60)

    zone_demand = (
        demand.groupby("zone_id")["trip_count"]
        .sum()
        .sort_values(ascending=False)
    )

    print(zone_demand.head(10).to_string())

    print("\nZERO-DEMAND COMBINATIONS")
    print("-" * 60)

    zero_count = (demand["trip_count"] == 0).sum()
    zero_share = zero_count / len(demand) * 100

    print(f"Zero-demand rows: {zero_count:,}")
    print(f"Zero-demand share: {zero_share:.2f}%")

    print(f"\nSaved dataset to: {output_path}")


if __name__ == "__main__":
    build_hourly_demand(SOURCE_PATH, OUTPUT_PATH)