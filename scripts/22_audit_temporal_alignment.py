
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
        "datetime_column": "datetime",
        "zone_column": "zone_id",
        "format": "parquet",
    },
    "collisions": {
        "path": DATA_DIR / "processed" / "hourly_collisions_by_zone.parquet",
        "datetime_column": "datetime",
        "zone_column": "zone_id",
        "format": "parquet",
    },
    "requests_311": {
        "path": DATA_DIR / "processed" / "hourly_311_by_zone.parquet",
        "datetime_column": "datetime",
        "zone_column": "zone_id",
        "format": "parquet",
    },
    "weather": {
        "path": DATA_DIR / "raw" / "weather" / "open_meteo_2024-01-01_2025-12-31.csv",
        "datetime_column": "time",
        "zone_column": None,
        "format": "csv",
    },
}


def load_source(source_config):
    """Загрузка одного источника."""
    path = source_config["path"]

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    if source_config["format"] == "parquet":
        return pd.read_parquet(path)

    return pd.read_csv(path)


def audit_source(source_name, source_config):
    """Аудит временных характеристик источника."""

    logger.info("Проверка источника: %s", source_name)

    df = load_source(source_config)

    datetime_column = source_config["datetime_column"]
    zone_column = source_config["zone_column"]

    if datetime_column not in df.columns:
        raise ValueError(
            f"В источнике {source_name} отсутствует "
            f"столбец {datetime_column}"
        )

    df[datetime_column] = pd.to_datetime(
        df[datetime_column],
        errors="coerce",
    )

    invalid_datetime = int(df[datetime_column].isna().sum())

    df = df.dropna(subset=[datetime_column]).copy()

    # Проверяем временные характеристики.
    min_datetime = df[datetime_column].min()
    max_datetime = df[datetime_column].max()

    unique_hours = df[datetime_column].nunique()

    # Ожидаемая последовательность часовых интервалов.
    expected_hours = pd.date_range(
        start=min_datetime,
        end=max_datetime,
        freq="h",
    )

    actual_hours = pd.DatetimeIndex(
        df[datetime_column].drop_duplicates().sort_values()
    )

    missing_hours = expected_hours.difference(actual_hours)

    unexpected_hours = actual_hours.difference(expected_hours)

    # Проверка повторов по ключам.
    if zone_column:
        missing_zone = int(df[zone_column].isna().sum())

        duplicate_keys = int(
            df.duplicated(
                subset=[zone_column, datetime_column]
            ).sum()
        )

        unique_zones = int(df[zone_column].nunique())

    else:
        missing_zone = 0
        duplicate_keys = int(
            df.duplicated(subset=[datetime_column]).sum()
        )
        unique_zones = None

    # Определение наличия временной зоны.
    timezone = str(df[datetime_column].dt.tz)

    if timezone == "None":
        timezone_status = "naive"
    else:
        timezone_status = timezone

    profile = {
        "source": source_name,
        "rows": len(df),
        "invalid_datetime": invalid_datetime,
        "min_datetime": min_datetime,
        "max_datetime": max_datetime,
        "unique_hours": unique_hours,
        "expected_hours_between_bounds": len(expected_hours),
        "missing_hours": len(missing_hours),
        "unexpected_hours": len(unexpected_hours),
        "missing_zone": missing_zone,
        "unique_zones": unique_zones,
        "duplicate_keys": duplicate_keys,
        "timezone": timezone_status,
    }

    # Сохраняем временное покрытие источника.
    coverage = pd.DataFrame(
        {
            "datetime": expected_hours,
            source_name: expected_hours.isin(actual_hours),
        }
    )

    # Вывод результатов.
    logger.info("Строк: %s", f"{len(df):,}")
    logger.info("Начало периода: %s", min_datetime)
    logger.info("Конец периода: %s", max_datetime)
    logger.info("Уникальных часов: %s", f"{unique_hours:,}")
    logger.info("Пропущенных часов: %s", f"{len(missing_hours):,}")
    logger.info("Неожиданных часов: %s", f"{len(unexpected_hours):,}")
    logger.info("Часовой пояс: %s", timezone_status)
    logger.info("Дубликатов ключей: %s", f"{duplicate_keys:,}")

    if zone_column:
        logger.info("Количество зон: %s", unique_zones)
        logger.info("Записей без зоны: %s", f"{missing_zone:,}")

    del df

    return profile, coverage


def main():
    logger.info("=" * 65)
    logger.info("АУДИТ ВРЕМЕННОЙ СОГЛАСОВАННОСТИ ИСТОЧНИКОВ")
    logger.info("=" * 65)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    profiles = []
    coverages = []

    for source_name, source_config in SOURCES.items():
        profile, coverage = audit_source(
            source_name,
            source_config,
        )

        profiles.append(profile)
        coverages.append(coverage)

    # Объединяем временное покрытие источников.
    coverage_result = coverages[0]

    for coverage in coverages[1:]:
        coverage_result = coverage_result.merge(
            coverage,
            on="datetime",
            how="outer",
        )

    coverage_result = coverage_result.sort_values(
        "datetime"
    ).reset_index(drop=True)

    source_columns = list(SOURCES.keys())

    # Считаем количество источников, содержащих каждый час.
    coverage_result["sources_available"] = (
        coverage_result[source_columns]
        .fillna(False)
        .astype(bool)
        .sum(axis=1)
    )

    coverage_result["all_sources_available"] = (
        coverage_result["sources_available"] == len(SOURCES)
    )

    # Сохраняем отчёты.
    profile_df = pd.DataFrame(profiles)

    profile_file = REPORT_DIR / "source_temporal_audit.csv"
    coverage_file = REPORT_DIR / "source_hourly_coverage.csv"

    profile_df.to_csv(
        profile_file,
        index=False,
        encoding="utf-8-sig",
    )

    coverage_result.to_csv(
        coverage_file,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info("=" * 65)
    logger.info("СВОДНЫЕ РЕЗУЛЬТАТЫ")
    logger.info("=" * 65)

    for _, row in profile_df.iterrows():
        logger.info(
            "%s: период %s — %s; часов: %s; пропусков: %s; "
            "дубликатов: %s",
            row["source"],
            row["min_datetime"],
            row["max_datetime"],
            f"{row['unique_hours']:,}",
            f"{row['missing_hours']:,}",
            f"{row['duplicate_keys']:,}",
        )

    common_hours = int(
        coverage_result["all_sources_available"].sum()
    )

    logger.info(
        "Часов, присутствующих одновременно во всех источниках: %s",
        f"{common_hours:,}",
    )

    logger.info(
        "Всего часов в объединённом временном покрытии: %s",
        f"{len(coverage_result):,}",
    )

    logger.info("Отчёт профилей: %s", profile_file)
    logger.info("Отчёт покрытия: %s", coverage_file)

    logger.info("Аудит временной согласованности завершён")


if __name__ == "__main__":
    main()
