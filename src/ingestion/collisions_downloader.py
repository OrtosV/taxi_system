
import logging
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import load_config, get_project_root


logger = logging.getLogger("taxi_system")


def create_session() -> requests.Session:
    """Создаёт HTTP-сессию с повторными попытками при временных ошибках."""

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


def download_collisions(force: bool = False) -> Path:
    """Загружает сведения о ДТП из NYC Open Data за заданный период."""

    config = load_config()
    source_config = config["sources"]["collisions"]

    if not source_config["enabled"]:
        raise RuntimeError("Загрузка данных о ДТП отключена в конфигурации.")

    project_root = get_project_root()

    output_dir = (
        project_root
        / config["data"]["raw_dir"]
        / "collisions"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (
        f"nyc_collisions_{source_config['start_date']}_"
        f"{source_config['end_date']}.csv"
    )

    if output_path.exists() and not force:
        logger.info("Файл уже существует: %s", output_path)
        return output_path

    temp_path = output_path.with_suffix(".tmp")

    if temp_path.exists():
        temp_path.unlink()

    selected_columns = source_config["selected_columns"]
    page_size = int(source_config["page_size"])

    end_date_exclusive = (
            pd.Timestamp(source_config["end_date"]) + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%dT00:00:00")

    where_clause = (
        f"crash_date >= '{source_config['start_date']}T00:00:00' "
        f"AND crash_date < '{end_date_exclusive}'"
    )

    session = create_session()

    offset = 0
    total_rows = 0

    logger.info(
        "Начало загрузки ДТП за период %s — %s",
        source_config["start_date"],
        source_config["end_date"],
    )

    try:
        while True:
            params = {
                "$select": ",".join(selected_columns),
                "$where": where_clause,
                "$order": "collision_id",
                "$limit": page_size,
                "$offset": offset,
            }

            response = session.get(
                source_config["base_url"],
                params=params,
                timeout=120,
            )
            response.raise_for_status()

            page_data = response.json()

            if not isinstance(page_data, list):
                raise ValueError("API вернул данные в неожиданном формате.")

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
                    f"В ответе API отсутствуют столбцы: {missing_columns}"
                )

            page = page[selected_columns]

            page.to_csv(
                temp_path,
                mode="a",
                index=False,
                header=(total_rows == 0),
                encoding="utf-8",
            )

            rows_in_page = len(page)
            total_rows += rows_in_page
            offset += rows_in_page

            logger.info(
                "Загружено записей: %s",
                total_rows,
            )

            if rows_in_page < page_size:
                break

        if total_rows == 0:
            raise ValueError("API не вернул ни одной записи о ДТП.")

        # Проверяем сформированный CSV перед публикацией файла.
        result = pd.read_csv(
            temp_path,
            usecols=["collision_id", "crash_date"],
        )

        if result["collision_id"].isna().any():
            raise ValueError("Обнаружены записи без идентификатора ДТП.")

        if result["collision_id"].duplicated().any():
            raise ValueError("Обнаружены повторяющиеся идентификаторы ДТП.")

        result["crash_date"] = pd.to_datetime(
            result["crash_date"],
            errors="coerce",
        )

        if result["crash_date"].isna().any():
            raise ValueError("Обнаружены некорректные даты ДТП.")

        start_date = pd.Timestamp(source_config["start_date"])
        end_date = pd.Timestamp(source_config["end_date"]) + pd.Timedelta(days=1)

        if not result["crash_date"].between(
            start_date,
            end_date,
            inclusive="left",
        ).all():
            raise ValueError("В данных обнаружены записи вне заданного периода.")

        temp_path.replace(output_path)

        logger.info(
            "Загрузка завершена. Всего записей: %s",
            total_rows,
        )
        logger.info("Файл сохранён: %s", output_path)

        return output_path

    except Exception:
        logger.exception("Ошибка при загрузке данных о ДТП.")
        raise

    finally:
        if temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    downloaded_file = download_collisions()
    print(f"\nФайл сохранён: {downloaded_file}")
