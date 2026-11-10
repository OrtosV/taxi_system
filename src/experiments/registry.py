
import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

# Корневая директория проекта
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Директории хранения моделей и экспериментов
MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
REGISTRY_DIR = MODELS_DIR / "registry"
EXPERIMENTS_DIR = REGISTRY_DIR / "experiments"

# Общая таблица экспериментов
REGISTRY_CSV = REGISTRY_DIR / "experiments.csv"


def _json_serializer(value):
    """
    Преобразует типы NumPy и Path в форматы,
    поддерживаемые JSON.
    """
    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, Path):
        return str(value)

    raise TypeError(
        f"Объект типа {type(value).__name__} "
        "не может быть сериализован в JSON"
    )


def _to_csv_value(value):
    """
    Преобразует сложные значения в безопасный формат для CSV.
    """
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value,
            ensure_ascii=False,
            default=_json_serializer,
        )

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return json.dumps(
            value.tolist(),
            ensure_ascii=False,
        )

    if isinstance(value, np.bool_):
        return bool(value)

    return value


def _rebuild_registry_csv():
    """
    Полностью перестраивает общую таблицу экспериментов
    на основе сохранённых JSON-файлов.

    Такой подход обеспечивает единый набор столбцов
    для экспериментов с разными моделями и параметрами.
    """
    experiments = []

    if not EXPERIMENTS_DIR.exists():
        logger.warning(
            "Директория экспериментов не найдена: %s",
            EXPERIMENTS_DIR,
        )
        return

    json_files = sorted(EXPERIMENTS_DIR.glob("*.json"))

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as file:
                experiment = json.load(file)

            row = {
                "experiment_id": experiment.get("experiment_id"),
                "model_name": experiment.get("model_name"),
                "created_at": experiment.get("created_at"),
                "random_state": experiment.get("random_state"),
                "model_artifact": experiment.get("model_artifact"),
            }

            # Гиперпараметры модели
            parameters = experiment.get("parameters", {})

            for param_name, param_value in parameters.items():
                row[f"param_{param_name}"] = _to_csv_value(
                    param_value
                )

            # Метрики на каждой выборке
            metrics_by_split = experiment.get("metrics", {})

            for split_name, metrics in metrics_by_split.items():
                for metric_name, metric_value in metrics.items():
                    row[f"{split_name}_{metric_name}"] = (
                        _to_csv_value(metric_value)
                    )

            # Информация о наборе данных
            dataset_info = experiment.get("dataset_info", {})

            for info_name, info_value in dataset_info.items():
                if not isinstance(info_value, (dict, list, tuple)):
                    row[f"dataset_{info_name}"] = _to_csv_value(
                        info_value
                    )

            # Количество признаков
            row["features_count"] = len(
                experiment.get("features", [])
            )

            experiments.append(row)

        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            logger.exception(
                "Не удалось обработать файл эксперимента: %s",
                json_path,
            )

    if not experiments:
        logger.warning(
            "Нет JSON-файлов экспериментов для построения реестра."
        )
        return

    # Pandas автоматически объединяет столбцы всех экспериментов.
    # Если у модели нет определённого параметра, значение будет NaN.
    registry_df = pd.DataFrame(experiments)

    # Стабильный порядок строк по времени создания
    if "created_at" in registry_df.columns:
        registry_df = registry_df.sort_values(
            by="created_at",
            na_position="last",
        )

    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    registry_df.to_csv(
        REGISTRY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(
        "Реестр экспериментов обновлён: %s строк, %s столбцов.",
        len(registry_df),
        len(registry_df.columns),
    )


def save_experiment(
    model,
    model_name,
    params,
    metrics_by_split,
    features,
    dataset_info,
    random_state=42,
):
    """
    Сохраняет обученную модель и информацию об эксперименте.

    Сохраняются:
    - обученная модель в формате joblib;
    - JSON с параметрами, метриками и признаками;
    - общая CSV-таблица экспериментов.
    """
    # Создание директорий при необходимости
    TRAINED_MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXPERIMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Уникальный идентификатор эксперимента
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"exp_{timestamp}"

    # Пути к файлам
    model_path = (
        TRAINED_MODELS_DIR / f"{experiment_id}.joblib"
    )

    experiment_path = (
        EXPERIMENTS_DIR / f"{experiment_id}.json"
    )

    # Сохранение обученной модели
    joblib.dump(model, model_path)

    # Формирование информации об эксперименте
    experiment = {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "random_state": random_state,
        "parameters": params,
        "metrics": metrics_by_split,
        "features": features,
        "features_count": len(features),
        "dataset_info": dataset_info,
        "model_artifact": str(
            model_path.relative_to(PROJECT_ROOT)
        ),
    }

    # Сохранение JSON
    with open(
        experiment_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            experiment,
            file,
            ensure_ascii=False,
            indent=4,
            default=_json_serializer,
        )

    # Перестроение общей CSV-таблицы из всех JSON
    _rebuild_registry_csv()

    logger.info(
        "Эксперимент сохранён: %s",
        experiment_id,
    )

    logger.info(
        "Модель сохранена: %s",
        model_path,
    )

    logger.info(
        "Информация об эксперименте сохранена: %s",
        experiment_path,
    )

    return experiment_id
