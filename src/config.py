from pathlib import Path
from typing import Any

import yaml


# Корневая директория проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Путь к конфигурационному файлу
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config() -> dict[str, Any]:
    """
    Загружает конфигурацию проекта из YAML-файла.

    Returns:
        dict[str, Any]: настройки проекта.
    """

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Файл конфигурации не найден: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Конфигурационный файл должен содержать словарь YAML"
        )

    return config


def get_project_root() -> Path:
    """
    Возвращает путь к корневой директории проекта.
    """

    return PROJECT_ROOT


if __name__ == "__main__":
    configuration = load_config()

    print("Конфигурация успешно загружена")
    print(f"Проект: {configuration['project']['name']}")
    print(f"Целевая переменная: {configuration['forecast']['target']}")
    print(f"Горизонт прогнозирования: {configuration['forecast']['horizon']} час.")
    print(f"Корневая директория: {get_project_root()}")