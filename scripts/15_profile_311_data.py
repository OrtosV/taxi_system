
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

from src.config import get_project_root


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("taxi_system")


def profile_311_data():
    root = get_project_root()

    input_dir = root / "data" / "raw" / "requests_311"
    reports_dir = root / "reports" / "tables"
    reports_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("nyc_311_*.csv.gz"))

    if len(files) != 24:
        raise ValueError(
            f"Ожидалось 24 месячных файла, найдено: {len(files)}"
        )

    use_columns = [
        "unique_key",
        "created_date",
        "complaint_type",
        "descriptor",
        "descriptor_2",
        "borough",
        "latitude",
        "longitude",
    ]

    monthly_results = []
    total_missing = Counter()
    complaint_counter = Counter()
    descriptor_counter = Counter()
    total_rows = 0

    for file_path in files:
        logger.info("Проверка файла: %s", file_path.name)

        month_rows = 0
        month_min_date = None
        month_max_date = None

        for chunk in pd.read_csv(
            file_path,
            compression="gzip",
            usecols=use_columns,
            chunksize=100_000,
            dtype={"unique_key": "string"},
        ):
            month_rows += len(chunk)

            for column in use_columns:
                total_missing[column] += int(
                    chunk[column].isna().sum()
                )

            dates = pd.to_datetime(
                chunk["created_date"],
                errors="coerce",
            )

            valid_dates = dates.dropna()

            if not valid_dates.empty:
                current_min = valid_dates.min()
                current_max = valid_dates.max()

                month_min_date = (
                    current_min
                    if month_min_date is None
                    else min(month_min_date, current_min)
                )

                month_max_date = (
                    current_max
                    if month_max_date is None
                    else max(month_max_date, current_max)
                )

            complaint_counter.update(
                chunk["complaint_type"].dropna().astype(str)
            )

            descriptor_counter.update(
                chunk["descriptor"].dropna().astype(str)
            )

        total_rows += month_rows

        monthly_results.append({
            "file": file_path.name,
            "month": file_path.stem.replace("nyc_311_", ""),
            "records": month_rows,
            "min_created_date": month_min_date,
            "max_created_date": month_max_date,
        })

        logger.info(
            "Файл проверен. Записей: %s",
            month_rows,
        )

    monthly_df = pd.DataFrame(monthly_results)

    missing_df = pd.DataFrame([
        {
            "column": column,
            "missing": missing,
            "missing_share_pct": round(
                missing / total_rows * 100, 2
            ),
        }
        for column, missing in total_missing.items()
    ])

    complaint_df = pd.DataFrame(
        complaint_counter.most_common(20),
        columns=["complaint_type", "records"],
    )

    descriptor_df = pd.DataFrame(
        descriptor_counter.most_common(20),
        columns=["descriptor", "records"],
    )

    monthly_df.to_csv(
        reports_dir / "311_monthly_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )

    missing_df.to_csv(
        reports_dir / "311_missing_values.csv",
        index=False,
        encoding="utf-8-sig",
    )

    complaint_df.to_csv(
        reports_dir / "311_top_complaint_types.csv",
        index=False,
        encoding="utf-8-sig",
    )

    descriptor_df.to_csv(
        reports_dir / "311_top_descriptors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 55)
    print("ИТОГОВЫЙ АУДИТ NYC 311")
    print("=" * 55)

    print(f"Количество файлов: {len(files)}")
    print(f"Всего записей: {total_rows:,}")

    print("\nПропущенные значения:")
    print(missing_df.to_string(index=False))

    print("\nТоп-20 категорий обращений:")
    print(complaint_df.to_string(index=False))

    print("\nТоп-20 текстовых описаний:")
    print(descriptor_df.to_string(index=False))

    print("\nПомесячная статистика:")
    print(monthly_df.to_string(index=False))

    print("\nОтчёты сохранены в:", reports_dir)


if __name__ == "__main__":
    profile_311_data()
