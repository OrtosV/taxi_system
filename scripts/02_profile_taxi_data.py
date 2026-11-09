from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "taxi"
    / "yellow_tripdata_2025-01.parquet"
)

BATCH_SIZE = 100_000


def profile_dataset(file_path: Path) -> None:
    """
    Выполняет полное профилирование TLC Parquet-файла
    пакетным чтением данных.
    """

    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    parquet_file = pq.ParquetFile(file_path)
    schema_names = parquet_file.schema_arrow.names

    # Общие счётчики
    total_rows = 0
    missing_counts = Counter()

    # Временные характеристики
    min_pickup = None
    max_pickup = None
    min_dropoff = None
    max_dropoff = None

    pickups_outside_january = 0
    invalid_durations = 0

    # Проверка числовых значений
    zero_passengers = 0
    negative_passengers = 0
    negative_distances = 0
    zero_distances = 0
    negative_fares = 0

    # Пространственные характеристики
    pickup_zones = set()
    dropoff_zones = set()

    invalid_pickup_zones = 0
    invalid_dropoff_zones = 0

    # Распределение по часам
    hourly_counts = Counter()
    passenger_counts = Counter()

    january_start = pd.Timestamp("2025-01-01")
    february_start = pd.Timestamp("2025-02-01")

    print("Starting full dataset profiling...")

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE
    ):
        total_rows += batch.num_rows

        # Считаем пропуски по всем столбцам
        for index, name in enumerate(schema_names):
            missing_counts[name] += batch.column(index).null_count

        # Для анализа используем необходимые столбцы
        df = batch.select([
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "passenger_count",
            "trip_distance",
            "PULocationID",
            "DOLocationID",
            "fare_amount"
        ]).to_pandas()

        pickup = df["tpep_pickup_datetime"]
        dropoff = df["tpep_dropoff_datetime"]

        # Временной диапазон
        if pickup.notna().any():
            batch_min = pickup.min()
            batch_max = pickup.max()

            min_pickup = (
                batch_min if min_pickup is None
                else min(min_pickup, batch_min)
            )

            max_pickup = (
                batch_max if max_pickup is None
                else max(max_pickup, batch_max)
            )

        if dropoff.notna().any():
            batch_min = dropoff.min()
            batch_max = dropoff.max()

            min_dropoff = (
                batch_min if min_dropoff is None
                else min(min_dropoff, batch_min)
            )

            max_dropoff = (
                batch_max if max_dropoff is None
                else max(max_dropoff, batch_max)
            )

        # Проверка дат за пределами января
        pickups_outside_january += (
            (pickup.notna()) &
            (
                (pickup < january_start) |
                (pickup >= february_start)
            )
        ).sum()

        # Продолжительность поездки
        duration = (
            dropoff - pickup
        ).dt.total_seconds()

        invalid_durations += (
            duration.notna() & (duration <= 0)
        ).sum()

        # Пассажиры
        zero_passengers += (
            df["passenger_count"] == 0
        ).sum()

        negative_passengers += (
            df["passenger_count"] < 0
        ).sum()

        passenger_counts.update(
            df["passenger_count"].dropna().value_counts().to_dict()
        )

        # Расстояние
        negative_distances += (
            df["trip_distance"] < 0
        ).sum()

        zero_distances += (
            df["trip_distance"] == 0
        ).sum()

        # Стоимость
        negative_fares += (
            df["fare_amount"] < 0
        ).sum()

        # Зоны
        pickup_zones.update(
            df["PULocationID"].dropna().unique()
        )

        dropoff_zones.update(
            df["DOLocationID"].dropna().unique()
        )

        invalid_pickup_zones += (
            df["PULocationID"].isna() |
            (df["PULocationID"] <= 0)
        ).sum()

        invalid_dropoff_zones += (
            df["DOLocationID"].isna() |
            (df["DOLocationID"] <= 0)
        ).sum()

        # Почасовое распределение
        valid_pickups = pickup.dropna()

        hours = valid_pickups.dt.floor("h")

        hourly_counts.update(
            hours.value_counts().to_dict()
        )

        print(
            f"Processed: {total_rows:,} rows",
            end="\r"
        )

    print("\n\n" + "=" * 65)
    print("FULL DATASET PROFILE")
    print("=" * 65)

    print(f"\nTotal rows: {total_rows:,}")

    print("\nTEMPORAL RANGE")
    print("-" * 65)
    print(f"Pickup start: {min_pickup}")
    print(f"Pickup end:   {max_pickup}")
    print(f"Dropoff start: {min_dropoff}")
    print(f"Dropoff end:   {max_dropoff}")
    print(f"Pickups outside January: {pickups_outside_january:,}")

    print("\nMISSING VALUES")
    print("-" * 65)

    for column in schema_names:
        count = missing_counts[column]
        percentage = count / total_rows * 100

        print(
            f"{column:30} "
            f"{count:12,} "
            f"({percentage:.3f}%)"
        )

    print("\nDATA QUALITY")
    print("-" * 65)
    print(f"Invalid durations: {invalid_durations:,}")
    print(f"Zero passengers: {zero_passengers:,}")
    print(f"Negative passengers: {negative_passengers:,}")
    print(f"Zero distances: {zero_distances:,}")
    print(f"Negative distances: {negative_distances:,}")
    print(f"Negative fares: {negative_fares:,}")

    print("\nSPATIAL INFORMATION")
    print("-" * 65)
    print(f"Unique pickup zones: {len(pickup_zones)}")
    print(f"Unique dropoff zones: {len(dropoff_zones)}")
    print(f"Invalid pickup zones: {invalid_pickup_zones:,}")
    print(f"Invalid dropoff zones: {invalid_dropoff_zones:,}")

    print("\nPASSENGER COUNT DISTRIBUTION")
    print("-" * 65)

    for value, count in sorted(passenger_counts.items()):
        print(f"{value}: {count:,}")

    print("\nHOURLY DISTRIBUTION")
    print("-" * 65)

    hourly_series = pd.Series(hourly_counts).sort_index()

    print(hourly_series.describe())

    print("\nTop 10 busiest hours:")
    print(hourly_series.nlargest(10).to_string())

    print("\nTop 10 quietest hours:")
    print(hourly_series.nsmallest(10).to_string())

    print("\n" + "=" * 65)
    print("PROFILE COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    profile_dataset(DATA_PATH)