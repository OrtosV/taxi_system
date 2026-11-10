
import logging

import geopandas as gpd

from src.config import get_project_root
from src.logging_config import setup_logging


def main():
    setup_logging()
    logger = logging.getLogger("taxi_system")

    project_root = get_project_root()

    shapefile_path = (
            project_root
            / "data"
            / "raw"
            / "taxi_zones"
            / "taxi_zones"
            / "taxi_zones"
            / "taxi_zones.shp"
    )

    zones = gpd.read_file(shapefile_path)

    logger.info("=" * 60)
    logger.info("АНАЛИЗ ГЕОГРАФИЧЕСКИХ ДАННЫХ TAXI ZONES")
    logger.info("=" * 60)

    logger.info("Количество полигонов: %s", len(zones))
    logger.info("Столбцы: %s", list(zones.columns))
    logger.info("Система координат: %s", zones.crs)
    logger.info("Типы геометрии: %s", zones.geometry.geom_type.value_counts().to_dict())

    logger.info(
        "Количество пустых геометрий: %s",
        zones.geometry.isna().sum()
    )

    logger.info(
        "Количество невалидных геометрий: %s",
        (~zones.geometry.is_valid).sum()
    )

    logger.info(
        "Количество уникальных LocationID: %s",
        zones["LocationID"].nunique()
        if "LocationID" in zones.columns
        else zones["locationid"].nunique()
    )

    logger.info("Распределение зон по Borough:")

    borough_column = (
        "borough"
        if "borough" in zones.columns
        else "Borough"
    )

    logger.info(
        "\n%s",
        zones[borough_column].value_counts(dropna=False).to_string()
    )

    logger.info("Первые пять записей:")
    logger.info("\n%s", zones.head().to_string())

    # Проверка возможности преобразования в географические координаты.
    zones_wgs84 = zones.to_crs(epsg=4326)

    logger.info("Преобразование в EPSG:4326 выполнено")
    logger.info("Границы геоданных: %s", zones_wgs84.total_bounds)


if __name__ == "__main__":
    main()
