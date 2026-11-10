
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logging_config import setup_logging


logger = setup_logging()

DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports" / "tables"

SOURCES = {
    "taxi_demand": {
        "path": DATA_DIR / "processed" / "historical_demand_with_zones.parquet",
        "column": "datetime",
        "format": "parquet",
    },
    "collisions": {
        "path": DATA_DIR / "processed" / "hourly_collisions_by_zone.parquet",
        "column": "datetime",
        "format": "parquet",
    },
    "requests_311": {
        "path": DATA_DIR / "processed" / "hourly_311_by_zone.parquet",
        "column": "datetime",
        "format": "parquet",
    },
    "weather": {
        "path": DATA_DIR / "raw" / "weather" / "open_meteo_2024-01-01_2025-12-31.csv",
        "column": "time",
        "format": "csv",
    },
}

DST_DATES = [
    "2024-03-10",
    "2024-11-03",
    "2025-03-09",
    "2025-11-02",
]


def load_timestamps(config):
    path = config["path"]

    if config["format"] == "parquet":
        df = pd.read_parquet(path, columns=[config["column"]])
    else:
        df = pd.read_csv(path, usecols=[config["column"]])

    timestamps = pd.to_datetime(
        df[config["column"]],
        errors="coerce",
    ).dropna()

    return timestamps


def main():
    logger.info("Начало детального анализа временных интервалов")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp("2024-01-01 00:00:00")
    end = pd.Timestamp("2025-12-31 23:00:00")

    expected = pd.date_range(
        start=start,
        end=end,
        freq="h",
    )

    missing_records = []
    dst_records = []

    timestamps_by_source = {}

    for source_name, config in SOURCES.items():
        logger.info("Анализ источника: %s", source_name)

        timestamps = load_timestamps(config)

        actual = pd.DatetimeIndex(
            timestamps.drop_duplicates().sort_values()
        )

        timestamps_by_source[source_name] = actual

        missing = expected.difference(actual)

        logger.info(
            "%s: уникальных часов %s, отсутствующих часов %s",
            source_name,
            f"{len(actual):,}",
            f"{len(missing):,}",
        )

        for timestamp in missing:
            missing_records.append(
                {
                    "source": source_name,
                    "missing_datetime": timestamp,
                    "date": timestamp.date(),
                    "hour": timestamp.hour,
                }
            )

        # Анализ временных меток около переходов DST.
        for dst_date in DST_DATES:
            day_start = pd.Timestamp(dst_date)
            day_end = day_start + pd.Timedelta(days=1)

            day_timestamps = timestamps[
                (timestamps >= day_start)
                & (timestamps < day_end)
            ]

            hourly_counts = (
                day_timestamps.dt.floor("h")
                .value_counts()
                .sort_index()
            )

            for hour in range(24):
                timestamp = day_start + pd.Timedelta(hours=hour)

                dst_records.append(
                    {
                        "source": source_name,
                        "dst_date": dst_date,
                        "datetime": timestamp,
                        "hour": hour,
                        "records": int(
                            hourly_counts.get(timestamp, 0)
                        ),
                    }
                )

    missing_df = pd.DataFrame(missing_records)

    dst_df = pd.DataFrame(dst_records)

    # Сравниваем временные метки внешних источников
    # с временными метками целевого набора.
    target_hours = set(timestamps_by_source["taxi_demand"])

    comparison_records = []

    for source_name, timestamps in timestamps_by_source.items():
        source_hours = set(timestamps)

        comparison_records.append(
            {
                "source": source_name,
                "hours_in_source": len(source_hours),
                "hours_in_target": len(target_hours),
                "common_hours": len(source_hours & target_hours),
                "only_in_source": len(source_hours - target_hours),
                "missing_from_source": len(target_hours - source_hours),
            }
        )

    comparison_df = pd.DataFrame(comparison_records)

    missing_file = REPORT_DIR / "missing_hours_details.csv"
    dst_file = REPORT_DIR / "dst_transition_audit.csv"
    comparison_file = REPORT_DIR / "source_target_time_comparison.csv"

    missing_df.to_csv(
        missing_file,
        index=False,
        encoding="utf-8-sig",
    )

    dst_df.to_csv(
        dst_file,
        index=False,
        encoding="utf-8-sig",
    )

    comparison_df.to_csv(
        comparison_file,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info("=" * 60)
    logger.info("ОТСУТСТВУЮЩИЕ ЧАСЫ")
    logger.info("=" * 60)

    if missing_df.empty:
        logger.info("Отсутствующих часов нет")
    else:
        for source_name, group in missing_df.groupby("source"):
            logger.info(
                "%s: %s отсутствующих часов",
                source_name,
                len(group),
            )

            logger.info(
                "Даты: %s",
                ", ".join(
                    group["missing_datetime"]
                    .astype(str)
                    .tolist()
                ),
            )

    logger.info("=" * 60)
    logger.info("СРАВНЕНИЕ С ЦЕЛЕВЫМ НАБОРОМ")
    logger.info("=" * 60)

    for _, row in comparison_df.iterrows():
        logger.info(
            "%s: совпадает %s; отсутствует в источнике %s; "
            "дополнительных часов %s",
            row["source"],
            f"{row['common_hours']:,}",
            f"{row['missing_from_source']:,}",
            f"{row['only_in_source']:,}",
        )

    logger.info("=" * 60)
    logger.info("ПРОВЕРКА ПЕРЕХОДОВ НА ЛЕТНЕЕ/ЗИМНЕЕ ВРЕМЯ")
    logger.info("=" * 60)

    for dst_date in DST_DATES:
        logger.info("Дата перехода: %s", dst_date)

        subset = dst_df[dst_df["dst_date"] == dst_date]

        for source_name, group in subset.groupby("source"):
            nonzero = group[group["records"] > 0]

            logger.info(
                "%s: часы с данными %s из 24",
                source_name,
                len(nonzero),
            )

            logger.info(
                "Количество записей по часам: %s",
                ", ".join(
                    f"{int(row.hour):02d}:00={int(row.records)}"
                    for row in group.itertuples()
                ),
            )

    logger.info("Отчёт пропущенных часов: %s", missing_file)
    logger.info("Отчёт переходов времени: %s", dst_file)
    logger.info("Сравнение с целевым набором: %s", comparison_file)

    logger.info("Детальный временной аудит завершён")


if __name__ == "__main__":
    main()
