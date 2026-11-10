
from pathlib import Path
import logging

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
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)

    return session


def download_weather(force: bool = False) -> Path:
    """Загружает исторические почасовые данные погоды Open-Meteo."""

    config = load_config()
    weather_config = config["sources"]["weather"]

    if not weather_config["enabled"]:
        raise RuntimeError("Загрузка погодных данных отключена в конфигурации.")

    project_root = get_project_root()
    output_dir = project_root / config["data"]["raw_dir"] / "weather"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (
        f"open_meteo_{weather_config['start_date']}_"
        f"{weather_config['end_date']}.csv"
    )

    if output_path.exists() and not force:
        logger.info("Файл погоды уже существует: %s", output_path)
        return output_path

    params = {
        "latitude": weather_config["latitude"],
        "longitude": weather_config["longitude"],
        "start_date": weather_config["start_date"],
        "end_date": weather_config["end_date"],
        "hourly": ",".join(weather_config["hourly_variables"]),
        "timezone": weather_config["timezone"],
    }

    logger.info(
        "Начало загрузки погоды за период %s — %s",
        params["start_date"],
        params["end_date"],
    )

    session = create_session()

    try:
        response = session.get(
            weather_config["base_url"],
            params=params,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

    except requests.RequestException:
        logger.exception("Ошибка при обращении к Open-Meteo API")
        raise

    if "hourly" not in data:
        raise ValueError("В ответе API отсутствует раздел hourly.")

    hourly_data = data["hourly"]

    if "time" not in hourly_data:
        raise ValueError("В ответе API отсутствует временная колонка time.")

    expected_variables = weather_config["hourly_variables"]

    missing_variables = [
        variable
        for variable in expected_variables
        if variable not in hourly_data
    ]

    if missing_variables:
        raise ValueError(
            f"В ответе API отсутствуют переменные: {missing_variables}"
        )

    result = pd.DataFrame(hourly_data)

    if result.empty:
        raise ValueError("API вернул пустой набор погодных данных.")

    result["time"] = pd.to_datetime(result["time"])

    if result["time"].duplicated().any():
        raise ValueError("В погодных данных обнаружены повторяющиеся временные метки.")

    if result[expected_variables].isna().all(axis=1).any():
        logger.warning(
            "Обнаружены временные строки, в которых отсутствуют все погодные признаки."
        )

    temp_path = output_path.with_suffix(".tmp")

    try:
        result.to_csv(temp_path, index=False, encoding="utf-8")
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    logger.info(
        "Загрузка завершена. Получено %s почасовых наблюдений.",
        len(result),
    )
    logger.info("Файл сохранён: %s", output_path)

    return output_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    downloaded_file = download_weather()
    print(f"\nФайл сохранён: {downloaded_file}")
