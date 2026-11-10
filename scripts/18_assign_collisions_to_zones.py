import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import pandas as pd

from src.logging_config import setup_logging

COLLISIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "collisions"
    / "nyc_collisions_2024-01-01_2025-12-31.csv"
)

ZONES_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "taxi_zones"
    / "taxi_zones"
    / "taxi_zones"
    / "taxi_zones.shp"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "collisions_with_zones.parquet"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "collisions_spatial_join_summary.csv"
)

logger = setup_logging()


def main():
    logger.info("Начало пространственной привязки ДТП к зонам такси")

    # 1. Проверка наличия исходных файлов
    if not COLLISIONS_PATH.exists():
        raise FileNotFoundError(
            f"Не найден файл ДТП: {COLLISIONS_PATH}"
        )

    if not ZONES_PATH.exists():
        raise FileNotFoundError(
            f"Не найден shapefile зон такси: {ZONES_PATH}"
        )

    # 2. Загрузка ДТП
    logger.info("Загрузка данных о ДТП")

    collisions = pd.read_csv(
        COLLISIONS_PATH,
        low_memory=False
    )

    initial_rows = len(collisions)

    logger.info(f"Загружено записей: {initial_rows:,}")

    required_columns = {
        "collision_id",
        "latitude",
        "longitude",
        "crash_date",
        "crash_time",
    }

    missing_columns = required_columns - set(collisions.columns)

    if missing_columns:
        raise ValueError(
            f"В данных ДТП отсутствуют столбцы: {missing_columns}"
        )

    # 3. Подготовка координат
    collisions["latitude"] = pd.to_numeric(
        collisions["latitude"],
        errors="coerce"
    )

    collisions["longitude"] = pd.to_numeric(
        collisions["longitude"],
        errors="coerce"
    )

    valid_coordinates = (
        collisions["latitude"].notna()
        & collisions["longitude"].notna()
    )

    collisions_without_coordinates = int(
        (~valid_coordinates).sum()
    )

    logger.info(
        "Записей без координат: "
        f"{collisions_without_coordinates:,}"
    )

    # 4. Загрузка географических границ зон
    logger.info("Загрузка полигонов зон такси")

    zones = gpd.read_file(ZONES_PATH)

    if zones.crs is None:
        raise ValueError(
            "У shapefile отсутствует система координат (CRS)"
        )

    zones = zones[
        ["LocationID", "zone", "borough", "geometry"]
    ].copy()

    zones = zones.rename(
        columns={
            "LocationID": "zone_id",
            "zone": "taxi_zone",
            "borough": "taxi_borough",
        }
    )

    zones["zone_id"] = zones["zone_id"].astype(int)

    logger.info(f"Загружено зон: {len(zones)}")
    logger.info(f"Система координат полигонов: {zones.crs}")

    # 5. Создание географических точек из координат ДТП
    points_df = collisions.loc[valid_coordinates].copy()

    points = gpd.GeoDataFrame(
        points_df,
        geometry=gpd.points_from_xy(
            points_df["longitude"],
            points_df["latitude"]
        ),
        crs="EPSG:4326"
    )

    # 6. Преобразование координат в CRS полигонов
    points = points.to_crs(zones.crs)

    logger.info(
        f"Система координат точек после преобразования: {points.crs}"
    )

    # 7. Пространственное объединение
    logger.info("Выполнение пространственного объединения")

    joined = gpd.sjoin(
        points,
        zones,
        how="left",
        predicate="within"
    )

    # Точки на границах или в пересекающихся полигонах
    # дополнительно проверим по количеству результатов объединения.
    duplicated_indices = int(
        joined.index.duplicated(keep="first").sum()
    )

    if duplicated_indices > 0:
        logger.warning(
            "Обнаружены повторные пространственные соответствия: "
            f"{duplicated_indices:,}. Для каждой записи оставлено "
            "первое соответствие."
        )

    joined = joined.loc[
        ~joined.index.duplicated(keep="first")
    ]

    # 8. Добавление результатов к исходной таблице
    result = collisions.copy()

    result["zone_id"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64"
    )

    result["taxi_zone"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string"
    )

    result["taxi_borough"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string"
    )

    result.loc[joined.index, "zone_id"] = (
        joined["zone_id"].astype("Int64")
    )

    result.loc[joined.index, "taxi_zone"] = (
        joined["taxi_zone"].astype("string")
    )

    result.loc[joined.index, "taxi_borough"] = (
        joined["taxi_borough"].astype("string")
    )

    # 9. Итоговые показатели
    matched = int(result["zone_id"].notna().sum())

    unmatched_with_coordinates = int(
        (valid_coordinates & result["zone_id"].isna()).sum()
    )

    unmatched_percent = (
        unmatched_with_coordinates
        / int(valid_coordinates.sum())
        * 100
        if valid_coordinates.sum() > 0
        else 0
    )

    logger.info("=" * 60)
    logger.info("ИТОГИ ПРОСТРАНСТВЕННОЙ ПРИВЯЗКИ ДТП")
    logger.info("=" * 60)
    logger.info(f"Исходное количество записей: {initial_rows:,}")
    logger.info(f"С координатами: {int(valid_coordinates.sum()):,}")
    logger.info(f"Без координат: {collisions_without_coordinates:,}")
    logger.info(f"Привязано к зонам: {matched:,}")
    logger.info(
        "Не привязано при наличии координат: "
        f"{unmatched_with_coordinates:,} "
        f"({unmatched_percent:.2f}%)"
    )
    logger.info(
        f"Повторных пространственных соответствий: {duplicated_indices:,}"
    )

    # 10. Сохранение результатов
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    result.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    summary = pd.DataFrame([
        ("Исходное количество записей", initial_rows),
        ("Записей с координатами", int(valid_coordinates.sum())),
        ("Записей без координат", collisions_without_coordinates),
        ("Привязано к зонам", matched),
        ("Не привязано при наличии координат", unmatched_with_coordinates),
        ("Доля непривязанных среди записей с координатами, %", round(unmatched_percent, 4)),
        ("Повторных пространственных соответствий", duplicated_indices),
        ("Количество зон в shapefile", len(zones)),
        ("Количество зон, представленных в ДТП", result["zone_id"].nunique()),
    ], columns=["metric", "value"])

    summary.to_csv(
        REPORT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    logger.info(f"Результат сохранён: {OUTPUT_PATH}")
    logger.info(f"Отчёт сохранён: {REPORT_PATH}")
    logger.info("Пространственная привязка завершена")


if __name__ == "__main__":
    main()