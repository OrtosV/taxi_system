
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logging_config import setup_logging


INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "collisions_with_zones.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hourly_collisions_by_zone.parquet"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "collisions_hourly_aggregation_summary.csv"
)

logger = setup_logging()


def main():
    logger.info("Начало почасовой агрегации ДТП")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Не найден файл: {INPUT_PATH}"
        )

    # 1. Загрузка пространственно привязанных ДТП
    logger.info("Загрузка данных ДТП")

    df = pd.read_parquet(INPUT_PATH)

    initial_rows = len(df)

    logger.info(f"Загружено записей: {initial_rows:,}")

    # 2. Проверка необходимых столбцов
    required_columns = {
        "collision_id",
        "crash_date",
        "crash_time",
        "zone_id",
        "number_of_persons_injured",
        "number_of_persons_killed",
        "number_of_pedestrians_injured",
        "number_of_pedestrians_killed",
        "number_of_cyclist_injured",
        "number_of_cyclist_killed",
        "number_of_motorist_injured",
        "number_of_motorist_killed",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Отсутствуют необходимые столбцы: {missing_columns}"
        )

    # 3. Формирование временной метки
    logger.info("Формирование временной метки")

    date_time = (
        df["crash_date"].astype("string").str.strip()
        + " "
        + df["crash_time"].astype("string").str.strip()
    )

    df["datetime"] = pd.to_datetime(
        date_time,
        errors="coerce",
    )

    invalid_datetime = int(
        df["datetime"].isna().sum()
    )

    # 4. Фильтрация записей без зоны или корректного времени
    valid_mask = (
        df["zone_id"].notna()
        & df["datetime"].notna()
    )

    excluded_rows = int((~valid_mask).sum())

    df = df.loc[valid_mask].copy()

    df["zone_id"] = df["zone_id"].astype(int)

    logger.info(
        f"Записей после фильтрации: {len(df):,}"
    )

    # 5. Подготовка числовых показателей
    aggregation_columns = [
        "number_of_persons_injured",
        "number_of_persons_killed",
        "number_of_pedestrians_injured",
        "number_of_pedestrians_killed",
        "number_of_cyclist_injured",
        "number_of_cyclist_killed",
        "number_of_motorist_injured",
        "number_of_motorist_killed",
    ]

    for column in aggregation_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        ).fillna(0)

    # 6. Округление времени до часа
    df["datetime"] = df["datetime"].dt.floor("h")

    # 7. Почасовая агрегация по зонам
    logger.info("Агрегация ДТП по зонам и часам")

    hourly = (
        df.groupby(
            ["zone_id", "datetime"],
            as_index=False,
        )
        .agg(
            collision_count=("collision_id", "count"),
            injured_count=(
                "number_of_persons_injured",
                "sum",
            ),
            killed_count=(
                "number_of_persons_killed",
                "sum",
            ),
            pedestrians_injured=(
                "number_of_pedestrians_injured",
                "sum",
            ),
            pedestrians_killed=(
                "number_of_pedestrians_killed",
                "sum",
            ),
            cyclists_injured=(
                "number_of_cyclist_injured",
                "sum",
            ),
            cyclists_killed=(
                "number_of_cyclist_killed",
                "sum",
            ),
            motorists_injured=(
                "number_of_motorist_injured",
                "sum",
            ),
            motorists_killed=(
                "number_of_motorist_killed",
                "sum",
            ),
        )
    )

    hourly = hourly.sort_values(
        ["zone_id", "datetime"]
    ).reset_index(drop=True)

    # 8. Контроль результатов
    aggregated_collisions = int(
        hourly["collision_count"].sum()
    )

    unique_zones = int(
        hourly["zone_id"].nunique()
    )

    unique_hours = int(
        hourly["datetime"].nunique()
    )

    duplicate_keys = int(
        hourly.duplicated(
            subset=["zone_id", "datetime"]
        ).sum()
    )

    # 9. Сохранение результата
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    hourly.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # 10. Формирование отчёта
    summary = [
        ("Исходное количество ДТП", initial_rows),
        ("Записей с некорректным временем", invalid_datetime),
        ("Исключено записей", excluded_rows),
        ("Записей после фильтрации", len(df)),
        ("Количество агрегированных ДТП", aggregated_collisions),
        ("Количество зон", unique_zones),
        ("Количество уникальных часов", unique_hours),
        ("Количество строк после агрегации", len(hourly)),
        ("Дубли zone_id + datetime", duplicate_keys),
        (
            "Общее количество пострадавших",
            int(hourly["injured_count"].sum()),
        ),
        (
            "Общее количество погибших",
            int(hourly["killed_count"].sum()),
        ),
    ]

    summary_df = pd.DataFrame(
        summary,
        columns=["metric", "value"],
    )

    summary_df.to_csv(
        REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # 11. Итоговый вывод
    logger.info("=" * 60)
    logger.info("РЕЗУЛЬТАТЫ ПОЧАСОВОЙ АГРЕГАЦИИ ДТП")
    logger.info("=" * 60)

    logger.info(f"Исходное количество ДТП: {initial_rows:,}")
    logger.info(f"Некорректное время: {invalid_datetime:,}")
    logger.info(f"Исключено записей: {excluded_rows:,}")
    logger.info(f"Осталось записей: {len(df):,}")

    logger.info(
        f"Количество агрегированных ДТП: {aggregated_collisions:,}"
    )

    logger.info(f"Количество зон: {unique_zones}")
    logger.info(f"Количество уникальных часов: {unique_hours}")
    logger.info(f"Строк после агрегации: {len(hourly):,}")

    logger.info(f"Дубли zone_id + datetime: {duplicate_keys}")

    logger.info(
        f"Общее количество пострадавших: "
        f"{hourly['injured_count'].sum():,.0f}"
    )

    logger.info(
        f"Общее количество погибших: "
        f"{hourly['killed_count'].sum():,.0f}"
    )

    logger.info(f"Результат сохранён: {OUTPUT_PATH}")
    logger.info(f"Отчёт сохранён: {REPORT_PATH}")

    logger.info("Почасовая агрегация ДТП завершена")


if __name__ == "__main__":
    main()
