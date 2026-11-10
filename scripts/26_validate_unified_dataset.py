
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logging_config import setup_logging


INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unified_hourly_dataset.parquet"
)

OUTPUT_PATH = INPUT_PATH

REPORT_DIR = PROJECT_ROOT / "reports" / "tables"

QUALITY_REPORT_PATH = REPORT_DIR / "unified_dataset_quality.csv"
NUMERIC_REPORT_PATH = REPORT_DIR / "unified_dataset_numeric_profile.csv"

logger = setup_logging()


COUNT_COLUMNS = [
    "trip_count",
    "collision_count",
    "injured_count",
    "killed_count",
    "pedestrians_injured",
    "pedestrians_killed",
    "cyclists_injured",
    "cyclists_killed",
    "motorists_injured",
    "motorists_killed",
    "total_311_count",
    "traffic_related_311_count",
    "road_condition_311_count",
    "urban_activity_311_count",
    "residential_environment_311_count",
    "described_311_count",
]

WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "weather_code",
]

DST_COLUMNS = [
    "is_spring_dst_hour",
    "is_fall_dst_hour",
    "is_dst_irregular_hour",
]


def load_dataset():
    logger.info("Загрузка объединенного набора данных...")

    df = pd.read_parquet(INPUT_PATH)

    df["datetime"] = pd.to_datetime(df["datetime"])

    logger.info(
        "Загружено %s строк и %s столбцов",
        len(df),
        len(df.columns),
    )

    return df


def check_structure(df):
    logger.info("Проверка структуры данных...")

    required_columns = (
        ["zone_id", "datetime", "Borough", "Zone", "service_zone"]
        + COUNT_COLUMNS
        + WEATHER_COLUMNS
        + DST_COLUMNS
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Отсутствуют обязательные столбцы: {missing_columns}"
        )

    duplicates = int(
        df.duplicated(["zone_id", "datetime"]).sum()
    )

    if duplicates:
        raise ValueError(
            f"Обнаружено дубликатов zone_id + datetime: {duplicates}"
        )

    if df["zone_id"].isna().any():
        raise ValueError("В zone_id обнаружены пропуски")

    if df["datetime"].isna().any():
        raise ValueError("В datetime обнаружены пропуски")

    logger.info("Дубликаты отсутствуют")
    logger.info("Обязательные столбцы присутствуют")


def check_temporal_grid(df):
    logger.info("Проверка временной сетки...")

    timestamps = pd.DatetimeIndex(
        df["datetime"].drop_duplicates().sort_values()
    )

    expected_hours = pd.date_range(
        start=timestamps.min(),
        end=timestamps.max(),
        freq="h",
    )

    missing_hours = expected_hours.difference(timestamps)
    extra_hours = timestamps.difference(expected_hours)

    if len(missing_hours) or len(extra_hours):
        raise ValueError(
            "Обнаружены нарушения общей временной сетки: "
            f"пропущено {len(missing_hours)}, "
            f"лишних {len(extra_hours)}"
        )

    zone_counts = df.groupby("zone_id")["datetime"].agg(
        rows="count",
        unique_hours="nunique",
        min_datetime="min",
        max_datetime="max",
    )

    expected_rows = len(timestamps)

    invalid_zones = zone_counts[
        (zone_counts["rows"] != expected_rows)
        | (zone_counts["unique_hours"] != expected_rows)
        | (zone_counts["min_datetime"] != timestamps.min())
        | (zone_counts["max_datetime"] != timestamps.max())
    ]

    if not invalid_zones.empty:
        raise ValueError(
            "Нарушена полнота временной сетки для зон:\n"
            f"{invalid_zones.head(10)}"
        )

    logger.info(
        "Временная сетка корректна: %s часов, %s зон",
        len(timestamps),
        df["zone_id"].nunique(),
    )


def check_count_columns(df):
    logger.info("Проверка счетчиков и целевой переменной...")

    report = []

    for column in COUNT_COLUMNS:
        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        missing_count = int(values.isna().sum())
        negative_count = int((values < 0).sum())

        fractional_count = int(
            (
                values.notna()
                & ((values % 1).abs() > 1e-9)
            ).sum()
        )

        if missing_count:
            raise ValueError(
                f"В счетчике {column} обнаружено "
                f"{missing_count} пропусков"
            )

        if negative_count:
            raise ValueError(
                f"В счетчике {column} обнаружено "
                f"{negative_count} отрицательных значений"
            )

        if fractional_count:
            raise ValueError(
                f"В счетчике {column} обнаружено "
                f"{fractional_count} нецелых значений"
            )

        # Используем int64 для целевой переменной,
        # int32 для остальных счетчиков.
        if column == "trip_count":
            df[column] = values.astype("int64")
        else:
            if values.max() > np.iinfo(np.int32).max:
                raise ValueError(
                    f"Значения {column} превышают диапазон int32"
                )

            df[column] = values.astype("int32")

        report.append({
            "column": column,
            "missing_count": missing_count,
            "negative_count": negative_count,
            "fractional_count": fractional_count,
            "minimum": values.min(),
            "maximum": values.max(),
            "mean": round(values.mean(), 4),
            "dtype_after_conversion": str(df[column].dtype),
        })

    logger.info(
        "Проверено счетчиков: %s",
        len(COUNT_COLUMNS),
    )

    return report


def check_weather(df):
    logger.info("Проверка погодных данных...")

    spring_mask = df["is_spring_dst_hour"] == 1

    if int(spring_mask.sum()) != 526:
        raise ValueError(
            "Неожиданное количество строк весеннего перехода DST: "
            f"{int(spring_mask.sum())}"
        )

    for column in WEATHER_COLUMNS:
        missing_count = int(df[column].isna().sum())

        expected_missing = int(spring_mask.sum())

        if missing_count != expected_missing:
            raise ValueError(
                f"В {column} ожидалось {expected_missing} пропусков, "
                f"получено {missing_count}"
            )

        if df.loc[~spring_mask, column].isna().any():
            raise ValueError(
                f"В {column} есть пропуски вне весеннего перехода DST"
            )

    logger.info(
        "Пропуски погоды соответствуют весенним переходам DST"
    )


def check_dst_features(df):
    logger.info("Проверка признаков DST...")

    for column in DST_COLUMNS:
        invalid_values = ~df[column].isin([0, 1])

        if invalid_values.any():
            raise ValueError(
                f"В {column} обнаружены значения, отличные от 0 и 1"
            )

    expected_irregular = (
        (df["is_spring_dst_hour"] == 1)
        | (df["is_fall_dst_hour"] == 1)
    ).astype("int8")

    if not df["is_dst_irregular_hour"].equals(expected_irregular):
        raise ValueError(
            "Признак is_dst_irregular_hour рассчитан некорректно"
        )

    logger.info("Признаки DST корректны")


def build_numeric_profile(df):
    logger.info("Формирование профиля числовых признаков...")

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns

    profile = df[numeric_columns].describe(
        percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]
    ).T

    profile = profile.reset_index()
    profile = profile.rename(
        columns={"index": "feature"}
    )

    return profile


def build_quality_report(df, count_report):
    logger.info("Формирование итогового отчета...")

    quality_rows = []

    for column in df.columns:
        quality_rows.append({
            "column": column,
            "dtype": str(df[column].dtype),
            "missing_count": int(df[column].isna().sum()),
            "missing_percent": round(
                df[column].isna().mean() * 100,
                4,
            ),
            "unique_values": int(df[column].nunique()),
        })

    quality_report = pd.DataFrame(quality_rows)

    count_report_df = pd.DataFrame(count_report)

    return quality_report, count_report_df


def main():
    logger.info("=" * 70)
    logger.info("КОНТРОЛЬ КАЧЕСТВА ОБЪЕДИНЕННОГО НАБОРА")
    logger.info("=" * 70)

    df = load_dataset()

    check_structure(df)
    check_temporal_grid(df)

    count_report = check_count_columns(df)

    check_weather(df)
    check_dst_features(df)

    quality_report, count_report_df = build_quality_report(
        df,
        count_report,
    )

    numeric_profile = build_numeric_profile(df)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    quality_report.to_csv(
        QUALITY_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    count_report_df.to_csv(
        REPORT_DIR / "unified_count_columns_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )

    numeric_profile.to_csv(
        NUMERIC_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # Сохраняем подготовленные типы данных.
    df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    logger.info("=" * 70)
    logger.info("КОНТРОЛЬ КАЧЕСТВА ЗАВЕРШЕН")
    logger.info("Строк: %s", len(df))
    logger.info("Зон: %s", df["zone_id"].nunique())
    logger.info("Часов: %s", df["datetime"].nunique())
    logger.info("Столбцов: %s", len(df.columns))
    logger.info(
        "Целевой показатель trip_count: min=%s, max=%s, mean=%.2f",
        df["trip_count"].min(),
        df["trip_count"].max(),
        df["trip_count"].mean(),
    )
    logger.info(
        "Счетчики приведены к целочисленным типам"
    )
    logger.info("Набор сохранен: %s", OUTPUT_PATH)
    logger.info("Отчет по качеству: %s", QUALITY_REPORT_PATH)
    logger.info("Профиль числовых признаков: %s", NUMERIC_REPORT_PATH)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
