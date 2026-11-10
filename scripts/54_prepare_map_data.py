import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.logging_config import setup_logging


SHAPEFILE_PATH = (
    PROJECT_ROOT
    / "data/raw/taxi_zones/taxi_zones/taxi_zones/taxi_zones.shp"
)

TARGET_PATH = (
    PROJECT_ROOT
    / "data/processed/historical_demand_with_zones.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "data/processed/map"
OUTPUT_PATH = OUTPUT_DIR / "taxi_zones.geojson"

logger = setup_logging()


def main():
    logger.info("Начало подготовки геоданных для карты")

    # Загружаем геометрию зон такси
    zones = gpd.read_file(SHAPEFILE_PATH)

    # Загружаем список зон, присутствующих в исторических данных
    historical = pd.read_parquet(
        TARGET_PATH,
        columns=["zone_id"],
    )

    active_zone_ids = set(
        historical["zone_id"].dropna().astype(int).unique()
    )

    # Проверяем уникальность идентификаторов геометрий
    if zones["LocationID"].duplicated().any():
        raise ValueError("В shapefile обнаружены повторяющиеся LocationID")

    # Оставляем только зоны, присутствующие в историческом датасете
    zones["LocationID"] = zones["LocationID"].astype(int)
    zones = zones[zones["LocationID"].isin(active_zone_ids)].copy()

    # Зоны 264 и 265 являются специальными категориями TLC
    # и не имеют географических полигонов.
    zones_without_geometry = active_zone_ids - set(zones["LocationID"])

    expected_special_zones = {264, 265}
    unexpected_missing_zones = (
            zones_without_geometry - expected_special_zones
    )

    if unexpected_missing_zones:
        raise ValueError(
            f"Не найдены геометрии для зон: "
            f"{sorted(unexpected_missing_zones)}"
        )

    logger.info(
        "Зоны без геометрии (исключены только из карты): %s",
        sorted(zones_without_geometry),
    )

    # Проверяем геометрии
    if zones.geometry.isna().any():
        raise ValueError("Обнаружены пустые геометрии")

    if not zones.geometry.is_valid.all():
        raise ValueError("Обнаружены невалидные геометрии")

    # Преобразуем координаты в WGS84 для отображения на веб-карте
    zones = zones.to_crs(epsg=4326)

    # Оставляем поля, необходимые для интерфейса
    zones = zones[
        ["LocationID", "zone", "borough", "geometry"]
    ].copy()

    zones = zones.rename(
        columns={
            "LocationID": "zone_id",
            "zone": "zone_name",
            "borough": "borough",
        }
    )

    # Сохраняем GeoJSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    zones.to_file(OUTPUT_PATH, driver="GeoJSON")

    logger.info("Подготовка геоданных завершена")
    logger.info("Количество зон: %s", len(zones))
    logger.info("Система координат: %s", zones.crs)
    logger.info("Файл сохранен: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
