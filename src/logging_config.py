import logging
from pathlib import Path

from src.config import get_project_root


def setup_logging(
    log_level: str = "INFO",
    log_file: str = "logs/app.log"
) -> logging.Logger:
    """
    Настраивает централизованное логирование проекта.

    Логи одновременно выводятся в консоль и записываются в файл.

    Args:
        log_level: уровень логирования.
        log_file: путь к файлу журнала.

    Returns:
        logging.Logger: настроенный логгер.
    """

    project_root = get_project_root()
    log_path = project_root / log_file

    # Создаём директорию для журналов, если её нет
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("taxi_system")

    # Устанавливаем уровень логирования
    level = getattr(logging, log_level.upper(), None)

    if not isinstance(level, int):
        raise ValueError(f"Некорректный уровень логирования: {log_level}")

    logger.setLevel(level)

    # Не допускаем повторного добавления обработчиков
    if logger.handlers:
        return logger

    # Формат сообщений
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Вывод в консоль
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Запись в файл
    file_handler = logging.FileHandler(
        log_path,
        encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    logger = setup_logging()

    logger.debug("Сообщение для отладки")
    logger.info("Система запущена")
    logger.warning("Пример предупреждения")
    logger.error("Пример ошибки")