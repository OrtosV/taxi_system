import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, get_project_root
from src.logging_config import setup_logging

def main():
    config = load_config()
    setup_logging()

    logger = __import__("logging").getLogger("taxi_system")

    project_root = get_project_root()
    data_dir = project_root / config["data"]["raw_dir"] / "requests_311"

    files = sorted(data_dir.glob("nyc_311_*.csv.gz"))

    if not files:
        raise FileNotFoundError(
            f"Файлы NYC 311 не найдены: {data_dir}"
        )

    logger.info("Найдено файлов: %s", len(files))

    # Сводные показатели
    total_rows = 0
    unique_keys = set()
    duplicate_keys = 0

    min_date = None
    max_date = None

    missing_coordinates = 0
    invalid_coordinates = 0
    outside_nyc_coordinates = 0

    empty_strings = {
        "agency": 0,
        "complaint_type": 0,
        "descriptor": 0,
        "descriptor_2": 0,
        "borough": 0,
    }

    complaint_types = set()
    descriptors = set()
    descriptor_pairs = {}

    # Приблизительные границы географической области Нью-Йорка.
    # Это не границы административной территории, а проверочный диапазон.
    NYC_LAT_MIN, NYC_LAT_MAX = 40.45, 41.05
    NYC_LON_MIN, NYC_LON_MAX = -74.30, -73.65

    for file_path in files:
        logger.info("Проверка файла: %s", file_path.name)

        for chunk in pd.read_csv(
            file_path,
            compression="gzip",
            chunksize=100_000,
            low_memory=False,
        ):
            total_rows += len(chunk)

            # 1. Уникальность идентификаторов
            keys = chunk["unique_key"].dropna().astype(str)

            for key in keys:
                if key in unique_keys:
                    duplicate_keys += 1
                else:
                    unique_keys.add(key)

            # 2. Временные границы
            dates = pd.to_datetime(
                chunk["created_date"],
                errors="coerce"
            )

            valid_dates = dates.dropna()

            if not valid_dates.empty:
                chunk_min = valid_dates.min()
                chunk_max = valid_dates.max()

                min_date = (
                    chunk_min if min_date is None
                    else min(min_date, chunk_min)
                )
                max_date = (
                    chunk_max if max_date is None
                    else max(max_date, chunk_max)
                )

            # 3. Координаты
            lat = pd.to_numeric(chunk["latitude"], errors="coerce")
            lon = pd.to_numeric(chunk["longitude"], errors="coerce")

            missing_mask = lat.isna() | lon.isna()
            missing_coordinates += int(missing_mask.sum())

            invalid_mask = (
                (lat.notna() & ~lat.between(-90, 90))
                | (lon.notna() & ~lon.between(-180, 180))
            )
            invalid_coordinates += int(invalid_mask.sum())

            valid_coordinate_mask = (
                lat.between(-90, 90)
                & lon.between(-180, 180)
            )

            outside_mask = (
                valid_coordinate_mask
                & (
                    ~lat.between(NYC_LAT_MIN, NYC_LAT_MAX)
                    | ~lon.between(NYC_LON_MIN, NYC_LON_MAX)
                )
            )
            outside_nyc_coordinates += int(outside_mask.sum())

            # 4. Пустые строки
            for column in empty_strings:
                values = chunk[column].astype("string")
                empty_mask = (
                    values.notna()
                    & values.str.strip().eq("")
                )
                empty_strings[column] += int(empty_mask.sum())

            # 5. Уникальные категории и описания
            complaint_values = (
                chunk["complaint_type"]
                .dropna()
                .astype(str)
                .str.strip()
            )
            complaint_types.update(complaint_values.unique())

            descriptor_values = (
                chunk["descriptor"]
                .dropna()
                .astype(str)
                .str.strip()
            )
            descriptors.update(descriptor_values.unique())

            # 6. Частота сочетаний типа обращения и описания
            pairs = (
                chunk.groupby(
                    ["complaint_type", "descriptor"],
                    dropna=False
                )
                .size()
            )

            for pair, count in pairs.items():
                descriptor_pairs[pair] = (
                    descriptor_pairs.get(pair, 0) + int(count)
                )

    # Итоговые показатели
    logger.info("=" * 60)
    logger.info("ИТОГИ ПРОВЕРКИ NYC 311")
    logger.info("=" * 60)

    logger.info("Всего записей: %s", f"{total_rows:,}")
    logger.info("Уникальных идентификаторов: %s", f"{len(unique_keys):,}")
    logger.info("Повторных идентификаторов: %s", f"{duplicate_keys:,}")

    logger.info("Минимальная дата: %s", min_date)
    logger.info("Максимальная дата: %s", max_date)

    logger.info(
        "Записей без координат: %s (%.2f%%)",
        f"{missing_coordinates:,}",
        missing_coordinates / total_rows * 100
    )

    logger.info(
        "Некорректных координат: %s (%.4f%%)",
        f"{invalid_coordinates:,}",
        invalid_coordinates / total_rows * 100
    )

    logger.info(
        "Координат за пределами проверочного диапазона NYC: %s (%.2f%%)",
        f"{outside_nyc_coordinates:,}",
        outside_nyc_coordinates / total_rows * 100
    )

    logger.info("Количество уникальных типов обращений: %s", len(complaint_types))
    logger.info("Количество уникальных описаний: %s", len(descriptors))

    logger.info("Пустые строки:")
    for column, count in empty_strings.items():
        logger.info("  %s: %s", column, f"{count:,}")

    # Сохраняем наиболее частые сочетания категорий
    top_pairs = sorted(
        descriptor_pairs.items(),
        key=lambda item: item[1],
        reverse=True
    )[:30]

    pairs_df = pd.DataFrame(
        [
            {
                "complaint_type": pair[0],
                "descriptor": pair[1],
                "count": count,
            }
            for pair, count in top_pairs
        ]
    )

    reports_dir = project_root / "reports" / "tables"
    reports_dir.mkdir(parents=True, exist_ok=True)

    output_path = reports_dir / "311_top_category_pairs.csv"
    pairs_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    logger.info("Топ сочетаний сохранён: %s", output_path)
    logger.info("Проверка целостности завершена.")


if __name__ == "__main__":
    main()
