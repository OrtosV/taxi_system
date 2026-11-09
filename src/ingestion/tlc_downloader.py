from pathlib import Path
from datetime import date

import pyarrow.parquet as pq
import requests
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "taxi"

BASE_URL = (
    "https://d37ci6vzurychx.cloudfront.net/"
    "trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
)

CHUNK_SIZE = 1024 * 1024
TIMEOUT = (15, 120)


def get_months(start_year: int, start_month: int,
               end_year: int, end_month: int) -> list[tuple[int, int]]:
    """Формирует список месяцев включительно."""

    start = start_year * 12 + start_month - 1
    end = end_year * 12 + end_month - 1

    if start > end:
        raise ValueError("Start period must not be later than end period")

    return [
        (index // 12, index % 12 + 1)
        for index in range(start, end + 1)
    ]


def validate_parquet(file_path: Path) -> bool:
    """Проверяет, что файл является читаемым Parquet."""

    try:
        if not file_path.exists() or file_path.stat().st_size == 0:
            return False

        parquet_file = pq.ParquetFile(file_path)

        return parquet_file.metadata.num_rows > 0

    except Exception:
        return False


def download_month(year: int, month: int) -> Path | None:
    """Загружает один месячный файл с проверкой результата."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    file_name = f"yellow_tripdata_{year}-{month:02d}.parquet"
    output_path = RAW_DATA_DIR / file_name
    temp_path = RAW_DATA_DIR / f"{file_name}.part"

    # Повторный запуск не скачивает уже проверенный файл.
    if validate_parquet(output_path):
        print(f"[SKIP] {file_name} already exists and is valid")
        return output_path

    # Удаляем повреждённый ранее загруженный файл.
    if output_path.exists():
        output_path.unlink()

    if temp_path.exists():
        temp_path.unlink()

    url = BASE_URL.format(year=year, month=month)

    print(f"\n[DOWNLOAD] {file_name}")

    try:
        with requests.get(
            url,
            stream=True,
            timeout=TIMEOUT,
        ) as response:

            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(temp_path, "wb") as file:
                with tqdm(
                    total=total_size if total_size > 0 else None,
                    unit="B",
                    unit_scale=True,
                    desc=file_name,
                ) as progress:

                    for chunk in response.iter_content(
                        chunk_size=CHUNK_SIZE
                    ):
                        if chunk:
                            file.write(chunk)
                            progress.update(len(chunk))

        # Проверяем скачанный файл до его публикации
        # под окончательным именем.
        if not validate_parquet(temp_path):
            raise ValueError("Downloaded file is not a valid Parquet")

        temp_path.replace(output_path)

        size_mb = output_path.stat().st_size / (1024 ** 2)

        print(f"[SUCCESS] {file_name}: {size_mb:.2f} MB")

        return output_path

    except Exception as error:
        print(f"[ERROR] {file_name}: {error}")

        if temp_path.exists():
            temp_path.unlink()

        return None


def download_period(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
) -> None:
    """Загружает все месяцы выбранного периода."""

    months = get_months(
        start_year,
        start_month,
        end_year,
        end_month,
    )

    successful = []
    failed = []

    print("=" * 60)
    print("NYC TLC YELLOW TAXI DATA DOWNLOADER")
    print("=" * 60)
    print(f"Period: {start_year}-{start_month:02d} "
          f"— {end_year}-{end_month:02d}")
    print(f"Files to process: {len(months)}")

    for year, month in months:
        result = download_month(year, month)

        if result is not None:
            successful.append(result)
        else:
            failed.append((year, month))

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    print(f"Successful files: {len(successful)}")
    print(f"Failed files: {len(failed)}")

    total_size = sum(
        path.stat().st_size for path in successful
    )

    print(f"Total downloaded/available size: "
          f"{total_size / (1024 ** 3):.2f} GB")

    if failed:
        print("\nFailed months:")

        for year, month in failed:
            print(f"{year}-{month:02d}")

    print(f"\nStorage directory: {RAW_DATA_DIR}")


if __name__ == "__main__":
    download_period(
        start_year=2024,
        start_month=1,
        end_year=2025,
        end_month=12,
    )