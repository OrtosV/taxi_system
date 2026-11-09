from typing import Any


def validate_config(config: dict[str, Any]) -> None:
    """
    Проверяет корректность основных настроек проекта.

    При обнаружении ошибки генерирует ValueError.
    """

    required_sections = [
        "project",
        "data",
        "forecast",
        "sources",
        "modeling"
    ]

    # Проверяем наличие обязательных разделов
    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Отсутствует обязательный раздел: {section}"
            )

    # Проверяем параметры проекта
    project = config["project"]

    if not project.get("name"):
        raise ValueError("Не указано название проекта")

    if not isinstance(project.get("random_seed"), int):
        raise ValueError("random_seed должен быть целым числом")

    # Проверяем параметры прогнозирования
    forecast = config["forecast"]

    if not forecast.get("target"):
        raise ValueError("Не указана целевая переменная")

    if not isinstance(forecast.get("horizon"), int):
        raise ValueError("Горизонт прогнозирования должен быть целым числом")

    if forecast["horizon"] <= 0:
        raise ValueError("Горизонт прогнозирования должен быть больше нуля")

    if not forecast.get("frequency"):
        raise ValueError("Не указана частота прогнозирования")

    # Проверяем источники данных
    sources = config["sources"]

    if not isinstance(sources, dict):
        raise ValueError("Раздел sources должен быть словарём")

    enabled_sources = [
        name
        for name, settings in sources.items()
        if isinstance(settings, dict) and settings.get("enabled") is True
    ]

    if len(enabled_sources) < 2:
        raise ValueError(
            "Для работы системы необходимо минимум два активных источника данных"
        )

    # Проверяем параметры моделирования
    modeling = config["modeling"]

    if not isinstance(modeling.get("random_seed"), int):
        raise ValueError("Не задан корректный random_seed для моделирования")

    if modeling.get("validation_strategy") != "time_series":
        raise ValueError(
            "Для проекта должна использоваться временная валидация"
        )

    metrics = modeling.get("metrics")

    if not isinstance(metrics, list) or len(metrics) < 2:
        raise ValueError(
            "Необходимо определить минимум две метрики качества"
        )