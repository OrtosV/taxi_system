import pandas as pd

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_project_root
from src.logging_config import setup_logging

PROJECT_ROOT = get_project_root()

INPUT_DIR = PROJECT_ROOT / "data" / "interim" / "311_with_zones"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "tables"

OUTPUT_FILE = OUTPUT_DIR / "hourly_311_by_zone.parquet"
REPORT_FILE = REPORT_DIR / "311_hourly_aggregation_summary.csv"

CHUNK_SIZE = 100_000

logger = setup_logging()


# Группы обращений для формирования тематических признаков.
TRAFFIC_TYPES = {
    "illegal parking",
    "blocked driveway",
    "abandoned vehicle",
    "derelict vehicles",
    "traffic signal condition",
}

ROAD_TYPES = {
    "street condition",
    "street sign",
    "sidewalk condition",
    "street light condition",
}

URBAN_ACTIVITY_TYPES = {
    "noise - residential",
    "noise - street/sidewalk",
    "noise - commercial",
    "noise - vehicle",
    "noise",
    "encampment",
}

RESIDENTIAL_TYPES = {
    "heat/hot water",
    "unsanitary condition",
    "dirty condition",
    "plumbing",
    "paint/plaster",
    "water leak",
    "door/window",
}


def normalize_text(series: pd.Series) -> pd.Series:
    """Нормализация текстовых категориальных значений."""
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def main():
    logger.info("Начало почасовой агрегации обращений 311")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(INPUT_DIR.rglob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"Parquet-файлы не найдены в {INPUT_DIR}"
        )

    logger.info("Найдено Parquet-файлов: %s", len(parquet_files))

    aggregated_parts = []

    total_rows = 0
    valid_rows = 0
    invalid_datetime = 0
    missing_zone = 0

    for file_number, file_path in enumerate(parquet_files, start=1):
        logger.info(
            "Обработка файла %s/%s: %s",
            file_number,
            len(parquet_files),
            file_path.name,
        )

        # Читаем файл и обрабатываем его целиком.
        # Размер каждого Parquet-файла ограничен при пространственном
        # сопоставлении, поэтому это не требует загрузки всего массива 311.
        df = pd.read_parquet(file_path)

        total_rows += len(df)

        required_columns = {
            "zone_id",
            "created_date",
            "complaint_type",
            "descriptor",
        }

        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"В файле {file_path.name} отсутствуют столбцы: "
                f"{sorted(missing_columns)}"
            )

        df["datetime"] = pd.to_datetime(
            df["created_date"],
            errors="coerce",
        )

        invalid_datetime += int(df["datetime"].isna().sum())
        missing_zone += int(df["zone_id"].isna().sum())

        df = df.dropna(subset=["datetime", "zone_id"]).copy()

        valid_rows += len(df)

        df["zone_id"] = pd.to_numeric(
            df["zone_id"],
            errors="coerce",
        )

        df = df.dropna(subset=["zone_id"]).copy()
        df["zone_id"] = df["zone_id"].astype("int32")

        df["datetime"] = df["datetime"].dt.floor("h")

        complaint = normalize_text(df["complaint_type"])
        descriptor = normalize_text(df["descriptor"])

        df["traffic_related_311_count"] = complaint.isin(
            TRAFFIC_TYPES
        ).astype("int32")

        df["road_condition_311_count"] = complaint.isin(
            ROAD_TYPES
        ).astype("int32")

        df["urban_activity_311_count"] = complaint.isin(
            URBAN_ACTIVITY_TYPES
        ).astype("int32")

        df["residential_environment_311_count"] = complaint.isin(
            RESIDENTIAL_TYPES
        ).astype("int32")

        df["described_311_count"] = descriptor.ne("").astype("int32")

        # Учитываем обращения по каждой зоне и каждому часу.
        grouped = (
            df.groupby(["zone_id", "datetime"], as_index=False)
            .agg(
                total_311_count=("unique_key", "count"),
                traffic_related_311_count=(
                    "traffic_related_311_count",
                    "sum",
                ),
                road_condition_311_count=(
                    "road_condition_311_count",
                    "sum",
                ),
                urban_activity_311_count=(
                    "urban_activity_311_count",
                    "sum",
                ),
                residential_environment_311_count=(
                    "residential_environment_311_count",
                    "sum",
                ),
                described_311_count=("described_311_count", "sum"),
            )
        )

        aggregated_parts.append(grouped)

        logger.info(
            "Файл обработан: исходных строк %s, "
            "агрегированных строк %s",
            f"{len(df):,}",
            f"{len(grouped):,}",
        )

        del df

    logger.info("Объединение промежуточных агрегаций")

    result = pd.concat(
        aggregated_parts,
        ignore_index=True,
    )

    # Один час и одна зона могут встречаться в разных частях данных.
    # Поэтому после объединения выполняем повторную агрегацию.
    feature_columns = [
        "total_311_count",
        "traffic_related_311_count",
        "road_condition_311_count",
        "urban_activity_311_count",
        "residential_environment_311_count",
        "described_311_count",
    ]

    result = (
        result.groupby(
            ["zone_id", "datetime"],
            as_index=False,
        )[feature_columns]
        .sum()
        .sort_values(["zone_id", "datetime"])
        .reset_index(drop=True)
    )

    # Контроль итоговых данных.
    duplicate_count = int(
        result.duplicated(["zone_id", "datetime"]).sum()
    )

    total_aggregated = int(result["total_311_count"].sum())

    report = pd.DataFrame(
        [
            {
                "source_rows": total_rows,
                "invalid_datetime": invalid_datetime,
                "missing_zone": missing_zone,
                "valid_rows": valid_rows,
                "aggregated_requests": total_aggregated,
                "zones": result["zone_id"].nunique(),
                "unique_hours": result["datetime"].nunique(),
                "rows_after_aggregation": len(result),
                "duplicate_zone_hour": duplicate_count,
                "traffic_related_requests": int(
                    result["traffic_related_311_count"].sum()
                ),
                "road_condition_requests": int(
                    result["road_condition_311_count"].sum()
                ),
                "urban_activity_requests": int(
                    result["urban_activity_311_count"].sum()
                ),
                "residential_environment_requests": int(
                    result["residential_environment_311_count"].sum()
                ),
                "requests_with_descriptor": int(
                    result["described_311_count"].sum()
                ),
            }
        ]
    )

    result.to_parquet(OUTPUT_FILE, index=False)
    report.to_csv(REPORT_FILE, index=False, encoding="utf-8-sig")

    logger.info("=" * 60)
    logger.info("РЕЗУЛЬТАТЫ ПОЧАСОВОЙ АГРЕГАЦИИ 311")
    logger.info("=" * 60)
    logger.info("Исходное количество обращений: %s", f"{total_rows:,}")
    logger.info("Некорректных временных меток: %s", f"{invalid_datetime:,}")
    logger.info("Записей без зоны: %s", f"{missing_zone:,}")
    logger.info("Записей после фильтрации: %s", f"{valid_rows:,}")
    logger.info("Количество агрегированных обращений: %s", f"{total_aggregated:,}")
    logger.info("Количество зон: %s", result["zone_id"].nunique())
    logger.info("Количество уникальных часов: %s", result["datetime"].nunique())
    logger.info("Строк после агрегации: %s", f"{len(result):,}")
    logger.info("Дубликаты zone_id + datetime: %s", duplicate_count)

    logger.info(
        "Обращения, связанные с транспортом: %s",
        f"{int(result['traffic_related_311_count'].sum()):,}",
    )
    logger.info(
        "Обращения о состоянии дорог: %s",
        f"{int(result['road_condition_311_count'].sum()):,}",
    )
    logger.info(
        "Обращения, связанные с городской активностью: %s",
        f"{int(result['urban_activity_311_count'].sum()):,}",
    )
    logger.info(
        "Обращения о жилой среде: %s",
        f"{int(result['residential_environment_311_count'].sum()):,}",
    )

    logger.info("Результат сохранён: %s", OUTPUT_FILE)
    logger.info("Отчёт сохранён: %s", REPORT_FILE)
    logger.info("Агрегация обращений 311 завершена")


if __name__ == "__main__":
    main()
