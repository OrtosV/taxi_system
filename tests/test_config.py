import copy

import pytest

from src.config import load_config
from src.validation.config_validation import validate_config


@pytest.fixture
def config():
    """
    Загружает конфигурацию для тестов.
    """

    return load_config()


def test_config_loading(config):
    """
    Проверяет успешную загрузку конфигурации.
    """

    assert isinstance(config, dict)


def test_required_sections(config):
    """
    Проверяет наличие обязательных разделов.
    """

    required_sections = [
        "project",
        "data",
        "forecast",
        "sources",
        "modeling"
    ]

    for section in required_sections:
        assert section in config


def test_config_validation(config):
    """
    Проверяет корректность исходной конфигурации.
    """

    validate_config(config)


def test_forecast_horizon_validation(config):
    """
    Проверяет обнаружение некорректного горизонта.
    """

    invalid_config = copy.deepcopy(config)
    invalid_config["forecast"]["horizon"] = 0

    with pytest.raises(ValueError):
        validate_config(invalid_config)


def test_sources_validation(config):
    """
    Проверяет требование к количеству активных источников.
    """

    invalid_config = copy.deepcopy(config)

    for source in invalid_config["sources"].values():
        source["enabled"] = False

    with pytest.raises(ValueError):
        validate_config(invalid_config)


def test_metrics_validation(config):
    """
    Проверяет наличие нескольких метрик качества.
    """

    invalid_config = copy.deepcopy(config)
    invalid_config["modeling"]["metrics"] = ["MAE"]

    with pytest.raises(ValueError):
        validate_config(invalid_config)