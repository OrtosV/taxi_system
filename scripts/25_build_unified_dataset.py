
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_project_root
from src.logging_config import setup_logging


PROJECT_ROOT = get_project_root()

TARGET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "historical_demand_with_zones.parquet"
)

WEATHER_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "weather"
    / "open_meteo_2024-01-01_2025-12-31.csv"
)

COLLISIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hourly_collisions_by_zone.parquet"
)

REQUESTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hourly_311_by_zone.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unified_hourly_dataset.parquet"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "unified_dataset_summary.csv"
)

logger = setup_logging()


def load_target():
    logger.info("Загрузка целевого набора данных...")

    df = pd.read_parquet(TARGET_PATH)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["zone_id"] = pd.to_numeric(
        df["zone_id"],
        errors="raise"
    ).astype("int64")

    df = df[
        [
            "zone_id",
            "datetime",
            "trip_count",
            "Borough",
            "Zone",
            "service_zone",
        ]
    ].copy()

    if df.duplicated(["zone_id", "datetime"]).any():
        raise ValueError(
            "В целевом наборе обнаружены дубликаты zone_id + datetime"
        )

    logger.info(
        "Целевой набор загружен: %s строк, %s зон, %s часов",
        len(df),
        df["zone_id"].nunique(),
        df["datetime"].nunique(),
    )

    return df


def load_weather():
    logger.info("Загрузка погодных данных...")

    df = pd.read_csv(WEATHER_PATH)

    # В Open-Meteo временная колонка обычно называется time.
    time_column = "time" if "time" in df.columns else "datetime"

    if time_column not in df.columns:
        raise ValueError(
            "В погодном файле отсутствует колонка time или datetime"
        )

    df = df.rename(columns={time_column: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])

    if df["datetime"].duplicated().any():
        raise ValueError(
            "В погодных данных обнаружены повторяющиеся временные метки"
        )

    logger.info(
        "Погодные данные загружены: %s строк, %s показателей",
        len(df),
        len(df.columns) - 1,
    )

    return df


def load_event_data(path, source_name):
    logger.info("Загрузка агрегированных данных: %s", source_name)

    df = pd.read_parquet(path)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["zone_id"] = pd.to_numeric(
        df["zone_id"],
        errors="raise"
    ).astype("int64")

    if df.duplicated(["zone_id", "datetime"]).any():
        raise ValueError(
            f"В источнике {source_name} обнаружены дубликаты zone_id + datetime"
        )

    logger.info(
        "%s: %s строк, %s зон, %s временных меток",
        source_name,
        len(df),
        df["zone_id"].nunique(),
        df["datetime"].nunique(),
    )

    return df


def build_dst_features(df):
    logger.info("Формирование признаков перехода на летнее и зимнее время...")

    spring_dates = [
        pd.Timestamp("2024-03-10"),
        pd.Timestamp("2025-03-09"),
    ]

    fall_dates = [
        pd.Timestamp("2024-11-03"),
        pd.Timestamp("2025-11-02"),
    ]

    date = df["datetime"].dt.normalize()
    hour = df["datetime"].dt.hour

    df["is_spring_dst_hour"] = (
        date.isin(spring_dates) & (hour == 2)
    ).astype("int8")

    df["is_fall_dst_hour"] = (
        date.isin(fall_dates) & (hour == 1)
    ).astype("int8")

    df["is_dst_irregular_hour"] = (
        (df["is_spring_dst_hour"] == 1)
        | (df["is_fall_dst_hour"] == 1)
    ).astype("int8")

    return df


def main():
    logger.info("=" * 70)
    logger.info("ФОРМИРОВАНИЕ ЕДИНОГО ПОЧАСОВОГО НАБОРА ДАННЫХ")
    logger.info("=" * 70)

    # 1. Загружаем целевой набор.
    target = load_target()

    # 2. Загружаем погоду.
    weather = load_weather()

    # 3. Загружаем агрегированные ДТП.
    collisions = load_event_data(
        COLLISIONS_PATH,
        "collisions"
    )

    # 4. Загружаем агрегированные обращения 311.
    requests = load_event_data(
        REQUESTS_PATH,
        "requests_311"
    )

    initial_rows = len(target)

    # 5. Присоединяем погодные данные по времени.
    logger.info("Объединение с погодными данными...")

    weather_columns = [
        column
        for column in weather.columns
        if column != "datetime"
    ]

    result = target.merge(
        weather,
        on="datetime",
        how="left",
        validate="many_to_one",
        indicator="_weather_merge",
    )

    result["weather_data_available"] = (
        result["_weather_merge"] == "both"
    ).astype("int8")

    result = result.drop(columns="_weather_merge")

    # 6. Присоединяем ДТП по зоне и часу.
    logger.info("Объединение с данными о ДТП...")

    collision_columns = [
        column
        for column in collisions.columns
        if column not in ["zone_id", "datetime"]
    ]

    result = result.merge(
        collisions,
        on=["zone_id", "datetime"],
        how="left",
        validate="one_to_one",
    )

    # Отсутствие агрегированной записи означает отсутствие события.
    for column in collision_columns:
        result[column] = result[column].fillna(0)

    # 7. Присоединяем обращения 311 по зоне и часу.
    logger.info("Объединение с данными 311...")

    request_columns = [
        column
        for column in requests.columns
        if column not in ["zone_id", "datetime"]
    ]

    result = result.merge(
        requests,
        on=["zone_id", "datetime"],
        how="left",
        validate="one_to_one",
    )

    for column in request_columns:
        result[column] = result[column].fillna(0)

    # 8. Обрабатываем особенности перехода на летнее время.
    result = build_dst_features(result)

    spring_mask = result["is_spring_dst_hour"] == 1

    # Погодные наблюдения в несуществующий локальный час
    # не используем из-за неоднозначности временной привязки.
    for column in weather_columns:
        result.loc[spring_mask, column] = pd.NA

    result.loc[spring_mask, "weather_data_available"] = 0

    # 9. Проверяем сохранность целевого набора.
    if len(result) != initial_rows:
        raise ValueError(
            "После объединения изменилось количество строк: "
            f"{initial_rows} -> {len(result)}"
        )

    if result.duplicated(["zone_id", "datetime"]).any():
        raise ValueError(
            "После объединения появились дубликаты zone_id + datetime"
        )

    if result["trip_count"].isna().any():
        raise ValueError(
            "В целевой переменной обнаружены пропуски"
        )

    # 10. Формируем отчет о качестве объединенного набора.
    logger.info("Формирование отчета о качестве данных...")

    summary = []

    for column in result.columns:
        summary.append({
            "column": column,
            "dtype": str(result[column].dtype),
            "missing_count": int(result[column].isna().sum()),
            "missing_percent": round(
                result[column].isna().mean() * 100,
                4
            ),
            "unique_values": int(result[column].nunique()),
        })

    summary_df = pd.DataFrame(summary)

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_df.to_csv(
        REPORT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # 11. Сохраняем единый набор.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result = result.sort_values(
        ["zone_id", "datetime"]
    ).reset_index(drop=True)

    result.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    logger.info("=" * 70)
    logger.info("ОБЪЕДИНЕНИЕ ЗАВЕРШЕНО")
    logger.info("Итоговое количество строк: %s", len(result))
    logger.info("Количество зон: %s", result["zone_id"].nunique())
    logger.info("Количество часов: %s", result["datetime"].nunique())
    logger.info("Количество признаков: %s", len(result.columns))
    logger.info(
        "Пропуски в погодных данных: %s",
        int(result[weather_columns].isna().any(axis=1).sum())
    )
    logger.info(
        "Весенние DST-строки: %s",
        int(spring_mask.sum())
    )
    logger.info(
        "Осенние DST-строки: %s",
        int(result["is_fall_dst_hour"].sum())
    )
    logger.info("Набор сохранен: %s", OUTPUT_PATH)
    logger.info("Отчет сохранен: %s", REPORT_PATH)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
