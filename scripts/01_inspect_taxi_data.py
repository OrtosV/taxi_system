from pathlib import Path

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "taxi"
    / "yellow_tripdata_2025-01.parquet"
)

SAMPLE_SIZE = 10_000


def inspect_parquet(file_path: Path) -> None:
    """
    Выполняет первичное исследование Parquet-файла.

    Изучает:
    - размер файла;
    - количество строк;
    - количество столбцов;
    - структуру данных;
    - типы данных;
    - первые наблюдения.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Файл данных не найден: {file_path}"
        )

    print("=" * 70)
    print("TAXI DATASET INSPECTION")
    print("=" * 70)

    # Размер файла
    file_size_mb = file_path.stat().st_size / (1024 ** 2)

    print(f"\nFile: {file_path.name}")
    print(f"Size: {file_size_mb:.2f} MB")

    # Открываем Parquet без полной загрузки в память
    parquet_file = pq.ParquetFile(file_path)
    metadata = parquet_file.metadata

    print("\nDATASET INFORMATION")
    print("-" * 70)

    print(f"Rows: {metadata.num_rows:,}")
    print(f"Columns: {metadata.num_columns}")
    print(f"Row groups: {metadata.num_row_groups}")

    # Изучаем структуру
    print("\nSCHEMA")
    print("-" * 70)

    print(parquet_file.schema)

    # Получаем небольшой фрагмент данных
    print("\nSAMPLE DATA")
    print("-" * 70)

    batch = next(
        parquet_file.iter_batches(
            batch_size=SAMPLE_SIZE
        )
    )

    sample = batch.to_pandas()

    print(sample.head(10).to_string(index=False))

    # Основная информация о типах данных
    print("\nDATA TYPES")
    print("-" * 70)

    print(sample.dtypes.to_string())

    # Пропуски в исследуемой выборке
    print("\nMISSING VALUES IN SAMPLE")
    print("-" * 70)

    missing = sample.isna().sum()

    missing_percentage = (
        sample.isna().mean() * 100
    )

    missing_report = (
        missing
        .to_frame(name="missing_count")
        .assign(
            missing_percentage=missing_percentage
        )
    )

    print(missing_report.to_string())

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    inspect_parquet(DATA_PATH)