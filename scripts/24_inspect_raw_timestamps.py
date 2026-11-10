
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logging_config import setup_logging


logger = setup_logging()

DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports" / "tables"

DST_DATES = [
    "2024-03-10",
    "2024-11-03",
    "2025-03-09",
    "2025-11-02",
]


def inspect_taxi():
    logger.info("=" * 60)
    logger.info("TLC: ИСХОДНЫЕ ВРЕМЕННЫЕ МЕТКИ")
    logger.info("=" * 60)

    taxi_dir = DATA_DIR / "raw" / "taxi"

    records = []

    for date in DST_DATES:
        logger.info("Проверка даты: %s", date)

        day_start = pd.Timestamp(date)
        day_end = day_start + pd.Timedelta(days=1)

        for file_path in sorted(taxi_dir.glob("*.parquet")):
            df = pd.read_parquet(
                file_path,
                columns=["tpep_pickup_datetime"],
            )

            timestamps = pd.to_datetime(
                df["tpep_pickup_datetime"],
                errors="coerce",
            )

            mask = (
                (timestamps >= day_start)
                & (timestamps < day_end)
            )

            day_data = timestamps[mask]

            if day_data.empty:
                continue

            hourly = (
                day_data.dt.floor("h")
                .value_counts()
                .sort_index()
            )

            for hour, count in hourly.items():
                records.append(
                    {
                        "source": "taxi",
                        "date": date,
                        "datetime": hour,
                        "records": int(count),
                    }
                )

            del df

    return records


def inspect_weather():
    logger.info("=" * 60)
    logger.info("OPEN-METEO: ИСХОДНЫЕ ВРЕМЕННЫЕ МЕТКИ")
    logger.info("=" * 60)

    path = (
        DATA_DIR
        / "raw"
        / "weather"
        / "open_meteo_2024-01-01_2025-12-31.csv"
    )

    df = pd.read_csv(path)

    logger.info("Столбцы: %s", list(df.columns))
    logger.info("Первые временные метки:")
    logger.info("\n%s", df.head(5).to_string(index=False))

    logger.info("Последние временные метки:")
    logger.info("\n%s", df.tail(5).to_string(index=False))

    records = []

    for date in DST_DATES:
        day_start = pd.Timestamp(date)
        day_end = day_start + pd.Timedelta(days=1)

        timestamps = pd.to_datetime(
            df["time"],
            errors="coerce",
        )

        day_data = timestamps[
            (timestamps >= day_start)
            & (timestamps < day_end)
        ]

        logger.info(
            "%s: количество наблюдений %s",
            date,
            len(day_data),
        )

        for timestamp in day_data:
            records.append(
                {
                    "source": "weather",
                    "date": date,
                    "datetime": timestamp,
                    "records": 1,
                }
            )

    return records


def inspect_event_sources():
    logger.info("=" * 60)
    logger.info("ДТП И 311: ИСХОДНЫЕ ВРЕМЕННЫЕ МЕТКИ")
    logger.info("=" * 60)

    sources = {
        "collisions": (
            DATA_DIR
            / "raw"
            / "collisions"
            / "nyc_collisions_2024-01-01_2025-12-31.csv"
        ),
        "requests_311": (
            DATA_DIR
            / "raw"
            / "requests_311"
        ),
    }

    records = []

    for source_name, path in sources.items():
        if source_name == "collisions":
            df = pd.read_csv(
                path,
                usecols=["crash_date", "crash_time"],
            )

            df["datetime"] = pd.to_datetime(
                df["crash_date"].astype(str)
                + " "
                + df["crash_time"].astype(str),
                errors="coerce",
            )

        else:
            files = sorted(path.glob("*.csv.gz"))

            for file_path in files:
                df = pd.read_csv(
                    file_path,
                    usecols=["created_date"],
                )

                df["datetime"] = pd.to_datetime(
                    df["created_date"],
                    errors="coerce",
                )

                for date in DST_DATES:
                    day_start = pd.Timestamp(date)
                    day_end = day_start + pd.Timedelta(days=1)

                    day_data = df.loc[
                        (df["datetime"] >= day_start)
                        & (df["datetime"] < day_end),
                        "datetime",
                    ]

                    for timestamp in day_data:
                        records.append(
                            {
                                "source": source_name,
                                "date": date,
                                "datetime": timestamp,
                                "records": 1,
                            }
                        )

                del df

            continue

        for date in DST_DATES:
            day_start = pd.Timestamp(date)
            day_end = day_start + pd.Timedelta(days=1)

            day_data = df.loc[
                (df["datetime"] >= day_start)
                & (df["datetime"] < day_end),
                "datetime",
            ]

            logger.info(
                "%s, %s: исходных записей %s",
                source_name,
                date,
                len(day_data),
            )

            for timestamp in day_data:
                records.append(
                    {
                        "source": source_name,
                        "date": date,
                        "datetime": timestamp,
                        "records": 1,
                    }
                )

    return records


def main():
    logger.info("Начало проверки исходных временных меток")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    records = []

    records.extend(inspect_taxi())
    records.extend(inspect_weather())
    records.extend(inspect_event_sources())

    result = pd.DataFrame(records)

    output_file = REPORT_DIR / "raw_timestamp_dst_audit.csv"

    result.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info("Количество проверенных временных меток: %s", len(result))
    logger.info("Отчёт сохранён: %s", output_file)
    logger.info("Проверка завершена")


if __name__ == "__main__":
    main()
