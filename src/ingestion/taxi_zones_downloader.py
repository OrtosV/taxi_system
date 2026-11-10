
import logging
import zipfile
from pathlib import Path

import requests

from src.config import get_project_root
from src.logging_config import setup_logging


URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"


def download_taxi_zones():
    project_root = get_project_root()

    output_dir = project_root / "data" / "raw" / "taxi_zones"
    extract_dir = output_dir / "taxi_zones"
    archive_path = output_dir / "taxi_zones.zip"

    logger = logging.getLogger("taxi_system")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Если геометрия уже распакована, повторно не загружаем.
    shp_files = list(extract_dir.glob("*.shp"))

    if shp_files:
        logger.info("Shapefile уже существует: %s", shp_files[0])
        return shp_files[0]

    temp_path = archive_path.with_suffix(".zip.part")

    logger.info("Начинаем загрузку Taxi Zones")

    try:
        with requests.get(
            URL,
            stream=True,
            timeout=(15, 120)
        ) as response:
            response.raise_for_status()

            with open(temp_path, "wb") as file:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        file.write(chunk)

        # Проверяем архив до его переименования.
        with zipfile.ZipFile(temp_path, "r") as archive:
            bad_file = archive.testzip()

            if bad_file is not None:
                raise ValueError(
                    f"Повреждён файл внутри архива: {bad_file}"
                )

            archive.extractall(extract_dir)

        temp_path.replace(archive_path)

        required_extensions = [
            ".shp",
            ".shx",
            ".dbf",
            ".prj",
        ]

        shp_files = list(extract_dir.rglob("*.shp"))

        if not shp_files:
            raise FileNotFoundError(
                "В архиве не найден файл .shp"
            )

        shp_path = shp_files[0]
        base_path = shp_path.with_suffix("")

        missing_files = [
            ext
            for ext in required_extensions
            if not base_path.with_suffix(ext).exists()
        ]

        if missing_files:
            raise FileNotFoundError(
                f"Отсутствуют компоненты Shapefile: {missing_files}"
            )

        logger.info("Архив успешно загружен: %s", archive_path)
        logger.info("Shapefile распакован: %s", shp_path)

        return shp_path

    except Exception:
        if temp_path.exists():
            temp_path.unlink()

        logger.exception("Ошибка загрузки Taxi Zones")
        raise


if __name__ == "__main__":
    setup_logging()
    download_taxi_zones()
