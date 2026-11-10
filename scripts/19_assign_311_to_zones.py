
import sys
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logging_config import setup_logging


# Пути к исходным данным
SOURCE_DIR = PROJECT_ROOT / "data" / "raw" / "requests_311"

ZONES_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "taxi_zones"
    / "taxi_zones"
    / "taxi_zones"
    / "taxi_zones.shp"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "311_with_zones"

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "311_spatial_join_summary.csv"
)

CHUNK_SIZE = 100_000

logger = setup_logging()


def process_month(file_path, zones):
    """Обрабатывает один месячный архив NYC 311."""

    month_name = file_path.name.replace(".csv.gz", "")
    month_output_dir = OUTPUT_DIR / month_name
    success_marker = month_output_dir / "_SUCCESS.txt"

    # Пропускаем уже успешно обработанный месяц
    if success_marker.exists():
        logger.info(
            f"Месяц уже обработан, пропускаем: {month_name}"
        )

        summary_path = month_output_dir / "summary.csv"

        if summary_path.exists():
            return pd.read_csv(summary_path).iloc[0].to_dict()

        raise RuntimeError(
            f"Найден маркер завершения, но отсутствует отчёт: "
            f"{month_name}"
        )

    # Удаляем незавершённую предыдущую обработку
    if month_output_dir.exists():
        shutil.rmtree(month_output_dir)

    month_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    logger.info(f"Обработка месяца: {month_name}")

    total_rows = 0
    rows_with_coordinates = 0
    rows_without_coordinates = 0
    matched_rows = 0
    unmatched_rows = 0
    multiple_matches = 0
    parts_written = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            file_path,
            compression="gzip",
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ),
        start=1,
    ):
        chunk = chunk.reset_index(drop=True)

        chunk_rows = len(chunk)
        total_rows += chunk_rows

        # Приведение координат к числовому формату
        chunk["latitude"] = pd.to_numeric(
            chunk["latitude"],
            errors="coerce"
        )

        chunk["longitude"] = pd.to_numeric(
            chunk["longitude"],
            errors="coerce"
        )

        valid_coordinates = (
            chunk["latitude"].notna()
            & chunk["longitude"].notna()
        )

        rows_without_coordinates += int(
            (~valid_coordinates).sum()
        )

        rows_with_coordinates += int(
            valid_coordinates.sum()
        )

        # Создание столбцов пространственной привязки
        chunk["zone_id"] = pd.Series(
            pd.NA,
            index=chunk.index,
            dtype="Int64"
        )

        chunk["taxi_zone"] = pd.Series(
            pd.NA,
            index=chunk.index,
            dtype="string"
        )

        chunk["taxi_borough"] = pd.Series(
            pd.NA,
            index=chunk.index,
            dtype="string"
        )

        # Пространственное объединение
        points_df = chunk.loc[valid_coordinates].copy()

        if not points_df.empty:

            points = gpd.GeoDataFrame(
                points_df,
                geometry=gpd.points_from_xy(
                    points_df["longitude"],
                    points_df["latitude"],
                ),
                crs="EPSG:4326",
            )

            # Приведение точек к CRS полигонов
            points = points.to_crs(zones.crs)

            joined = gpd.sjoin(
                points,
                zones,
                how="left",
                predicate="within",
                lsuffix="request",
                rsuffix="zone",
            )

            # После spatial join используем поля из правой таблицы — полигонов
            zone_columns = {
                "zone_id": "zone_id_zone",
                "taxi_zone": "taxi_zone_zone",
                "taxi_borough": "taxi_borough_zone",
            }

            missing_join_columns = (
                    set(zone_columns.values()) - set(joined.columns)
            )

            if missing_join_columns:
                raise ValueError(
                    "После пространственного объединения "
                    "отсутствуют столбцы полигонов: "
                    f"{missing_join_columns}. "
                    f"Доступные столбцы: {list(joined.columns)}"
                )

            # Переименовываем поля полигонов для дальнейшей обработки
            joined = joined.rename(
                columns={
                    "zone_id_zone": "zone_id",
                    "taxi_zone_zone": "taxi_zone",
                    "taxi_borough_zone": "taxi_borough",
                }
            )

            # Контроль повторных соответствий
            duplicated = int(
                joined.index.duplicated(keep="first").sum()
            )

            multiple_matches += duplicated

            if duplicated:
                logger.warning(
                    f"{month_name}, часть {chunk_number}: "
                    f"повторных соответствий: {duplicated}"
                )

            # Оставляем по одной записи на исходный индекс
            joined = joined.loc[
                ~joined.index.duplicated(keep="first")
            ]

            matched_mask = joined["zone_id"].notna()

            matched_rows += int(matched_mask.sum())

            unmatched_rows += int(
                (~matched_mask).sum()
            )

            # Перенос результатов в исходный DataFrame
            chunk.loc[joined.index, "zone_id"] = (
                joined["zone_id"].astype("Int64")
            )

            chunk.loc[joined.index, "taxi_zone"] = (
                joined["taxi_zone"].astype("string")
            )

            chunk.loc[joined.index, "taxi_borough"] = (
                joined["taxi_borough"].astype("string")
            )

        # Сохранение обработанной части
        output_path = (
            month_output_dir
            / f"part_{chunk_number:04d}.parquet"
        )

        chunk.to_parquet(
            output_path,
            index=False
        )

        parts_written += 1

        logger.info(
            f"{month_name}: часть {chunk_number}, "
            f"обработано {chunk_rows:,} записей"
        )

    # Итоговая статистика месяца
    summary = {
        "month": month_name,
        "total_rows": total_rows,
        "rows_with_coordinates": rows_with_coordinates,
        "rows_without_coordinates": rows_without_coordinates,
        "matched_rows": matched_rows,
        "unmatched_rows": unmatched_rows,
        "multiple_matches": multiple_matches,
        "parts_written": parts_written,
    }

    # Сохранение статистики
    pd.DataFrame([summary]).to_csv(
        month_output_dir / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Маркер успешного завершения
    success_marker.write_text(
        "Month successfully processed",
        encoding="utf-8",
    )

    logger.info(
        f"{month_name} завершён: "
        f"{matched_rows:,} привязано, "
        f"{unmatched_rows:,} не привязано"
    )

    return summary


def main():
    logger.info(
        "Начало пространственной привязки NYC 311"
    )

    # Проверка исходных файлов
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Не найдена папка исходных данных: {SOURCE_DIR}"
        )

    if not ZONES_PATH.exists():
        raise FileNotFoundError(
            f"Не найден shapefile зон такси: {ZONES_PATH}"
        )

    files = sorted(
        SOURCE_DIR.glob("nyc_311_*.csv.gz")
    )

    if not files:
        raise FileNotFoundError(
            f"Не найдены архивы NYC 311: {SOURCE_DIR}"
        )

    logger.info(
        f"Найдено месячных файлов: {len(files)}"
    )

    # Загрузка полигонов
    logger.info("Загрузка полигонов зон такси")

    zones = gpd.read_file(ZONES_PATH)

    if zones.crs is None:
        raise ValueError(
            "У shapefile отсутствует система координат"
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

    # Оставляем только необходимые поля.
    # Это исключает конфликты названий с исходным CSV.
    zones = zones[
        [
            "zone_id",
            "taxi_zone",
            "taxi_borough",
            "geometry",
        ]
    ]

    logger.info(
        f"Загружено полигонов: {len(zones)}"
    )

    logger.info(
        f"Система координат: {zones.crs}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Обработка всех месяцев
    monthly_results = []

    for file_path in files:
        result = process_month(
            file_path,
            zones
        )

        monthly_results.append(result)

    # Итоговая статистика
    summary_df = pd.DataFrame(monthly_results)

    numeric_columns = [
        "total_rows",
        "rows_with_coordinates",
        "rows_without_coordinates",
        "matched_rows",
        "unmatched_rows",
        "multiple_matches",
        "parts_written",
    ]

    totals = summary_df[numeric_columns].sum()

    total_rows = int(totals["total_rows"])

    rows_with_coordinates = int(
        totals["rows_with_coordinates"]
    )

    matched_rows = int(
        totals["matched_rows"]
    )

    unmatched_rows = int(
        totals["unmatched_rows"]
    )

    match_percent = (
        matched_rows / rows_with_coordinates * 100
        if rows_with_coordinates > 0
        else 0
    )

    # Сохранение итогового отчёта
    summary_df.to_csv(
        REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info("=" * 60)
    logger.info(
        "ИТОГИ ПРОСТРАНСТВЕННОЙ ПРИВЯЗКИ NYC 311"
    )
    logger.info("=" * 60)

    logger.info(
        f"Обработано файлов: {len(files)}"
    )

    logger.info(
        f"Всего записей: {total_rows:,}"
    )

    logger.info(
        f"Записей с координатами: {rows_with_coordinates:,}"
    )

    logger.info(
        f"Записей без координат: "
        f"{int(totals['rows_without_coordinates']):,}"
    )

    logger.info(
        f"Привязано к зонам: {matched_rows:,}"
    )

    logger.info(
        f"Не привязано: {unmatched_rows:,}"
    )

    logger.info(
        f"Доля привязанных точек: {match_percent:.2f}%"
    )

    logger.info(
        f"Повторных соответствий: "
        f"{int(totals['multiple_matches']):,}"
    )

    logger.info(
        f"Создано файлов Parquet: "
        f"{int(totals['parts_written']):,}"
    )

    logger.info(
        f"Итоговый отчёт: {REPORT_PATH}"
    )

    logger.info(
        f"Данные сохранены в: {OUTPUT_DIR}"
    )

    logger.info(
        "Пространственная привязка NYC 311 завершена"
    )


if __name__ == "__main__":
    main()
