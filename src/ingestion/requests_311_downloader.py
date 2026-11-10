import time
from requests.exceptions import RequestException
import gzip
import logging
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import load_config, get_project_root


logger = logging.getLogger("taxi_system")


def create_session() -> requests.Session:
    """Создаёт HTTP-сессию с повторными попытками."""

    session = requests.Session()

    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)

    return session



def download_month(
    session: requests.Session,
    source_config: dict,
    output_path: Path,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
) -> int:
    """Загружает обращения за один календарный месяц."""

    selected_columns = source_config["selected_columns"]
    page_size = int(source_config["page_size"])

    start_str = month_start.strftime("%Y-%m-%dT00:00:00")
    end_str = month_end.strftime("%Y-%m-%dT00:00:00")

    where_clause = (
        f"created_date >= '{start_str}' "
        f"AND created_date < '{end_str}'"
    )

    offset = 0
    total_rows = 0
    max_attempts = 5

    temp_path = output_path.with_suffix(".tmp")

    if temp_path.exists():
        temp_path.unlink()

    try:
        while True:
            params = {
                "$select": ",".join(selected_columns),
                "$where": where_clause,
                "$order": "unique_key",
                "$limit": page_size,
                "$offset": offset,
            }

            page_data = None

            for attempt in range(1, max_attempts + 1):
                try:
                    response = session.get(
                        source_config["base_url"],
                        params=params,
                        timeout=(15, 180),
                    )

                    if not response.ok:
                        logger.error(
                            "Ошибка API %s: %s",
                            response.status_code,
                            response.text[:2000],
                        )

                    response.raise_for_status()

                    page_data = response.json()
                    break

                except (RequestException, ValueError) as error:
                    logger.warning(
                        "Ошибка загрузки страницы месяца %s, "
                        "offset=%s, попытка %s/%s: %s",
                        month_start.strftime("%Y-%m"),
                        offset,
                        attempt,
                        max_attempts,
                        error,
                    )

                    if attempt == max_attempts:
                        raise

                    time.sleep(min(2 ** attempt, 30))

            if not isinstance(page_data, list):
                raise ValueError(
                    "API вернул данные в неожиданном формате."
                )

            if not page_data:
                break

            page = pd.DataFrame(page_data)

            missing_columns = [
                column
                for column in selected_columns
                if column not in page.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"Отсутствуют столбцы: {missing_columns}"
                )

            page = page[selected_columns]

            if page["unique_key"].isna().any():
                raise ValueError(
                    "Обнаружены обращения без unique_key."
                )

            if page["unique_key"].duplicated().any():
                raise ValueError(
                    "В одной странице обнаружены дубликаты unique_key."
                )

            # Каждая страница записывается отдельным gzip-блоком.
            # Несколько gzip-блоков могут храниться в одном файле.
            with gzip.open(
                temp_path,
                mode="at",
                encoding="utf-8",
                newline="",
            ) as file:
                page.to_csv(
                    file,
                    index=False,
                    header=(total_rows == 0),
                )

            rows_in_page = len(page)
            total_rows += rows_in_page
            offset += rows_in_page

            logger.info(
                "%s: загружено %s записей",
                month_start.strftime("%Y-%m"),
                total_rows,
            )

            if rows_in_page < page_size:
                break

        if total_rows == 0:
            raise ValueError(
                f"За месяц {month_start:%Y-%m} API не вернул записей."
            )

        # Проверяем сохранённый файл.
        check = pd.read_csv(
            temp_path,
            compression="gzip",
            usecols=["unique_key", "created_date"],
        )

        if check["unique_key"].isna().any():
            raise ValueError("Обнаружены пустые unique_key.")

        if check["unique_key"].duplicated().any():
            raise ValueError(
                "Обнаружены повторяющиеся unique_key в месячном файле."
            )

        check["created_date"] = pd.to_datetime(
            check["created_date"],
            errors="coerce",
        )

        if check["created_date"].isna().any():
            raise ValueError(
                "Обнаружены некорректные даты создания обращений."
            )

        if len(check) != total_rows:
            raise ValueError(
                "Количество записей после сохранения не совпадает."
            )

        if not check["created_date"].between(
            month_start,
            month_end,
            inclusive="left",
        ).all():
            raise ValueError(
                "В файле обнаружены записи за пределами месяца."
            )

        temp_path.replace(output_path)

        logger.info(
            "Месяц %s успешно сохранён. Всего записей: %s",
            month_start.strftime("%Y-%m"),
            total_rows,
        )

        return total_rows

    finally:
        if temp_path.exists():
            temp_path.unlink()


def download_requests_311(force: bool = False) -> list[Path]:
    """Загружает NYC 311 Service Requests за заданный период."""

    config = load_config()
    source_config = config["sources"]["requests_311"]

    if not source_config["enabled"]:
        raise RuntimeError(
            "Загрузка NYC 311 отключена в конфигурации."
        )

    project_root = get_project_root()

    output_dir = (
        project_root
        / config["data"]["raw_dir"]
        / "requests_311"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    start_date = pd.Timestamp(source_config["start_date"])
    end_date = pd.Timestamp(source_config["end_date"])

    if end_date < start_date:
        raise ValueError("Дата окончания раньше даты начала.")

    months = pd.date_range(
        start=start_date.to_period("M").to_timestamp(),
        end=end_date.to_period("M").to_timestamp(),
        freq="MS",
    )

    session = create_session()
    downloaded_files = []

    total_rows = 0

    logger.info(
        "Начало загрузки NYC 311 за период %s — %s",
        start_date.date(),
        end_date.date(),
    )

    for month_start in months:

        month_end = month_start + pd.offsets.MonthBegin(1)

        # Учитываем границы заданного периода.
        period_start = max(month_start, start_date)
        period_end = min(month_end, end_date + pd.Timedelta(days=1))

        output_path = output_dir / (
            f"nyc_311_{month_start:%Y-%m}.csv.gz"
        )

        if output_path.exists() and not force:
            logger.info(
                "Файл уже существует, пропускаем: %s",
                output_path.name,
            )
            downloaded_files.append(output_path)
            continue

        rows = download_month(
            session=session,
            source_config=source_config,
            output_path=output_path,
            month_start=period_start,
            month_end=period_end,
        )

        total_rows += rows
        downloaded_files.append(output_path)

        logger.info(
            "Месяц %s завершён. Записей: %s",
            month_start.strftime("%Y-%m"),
            rows,
        )

    logger.info(
        "Загрузка завершена. Новых записей: %s",
        total_rows,
    )

    logger.info(
        "Количество месячных файлов: %s",
        len(downloaded_files),
    )

    return downloaded_files


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    files = download_requests_311()

    print("\nЗагрузка завершена.")

    for file_path in files:
        print(file_path)
