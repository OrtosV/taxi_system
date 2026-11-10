import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_DIR = MODELS_DIR / "trained"
REGISTRY_DIR = MODELS_DIR / "registry"
EXPERIMENTS_DIR = REGISTRY_DIR / "experiments"

REGISTRY_CSV = REGISTRY_DIR / "experiments.csv"


def _json_serializer(value):
    """Преобразует типы NumPy и Path в форматы, поддерживаемые JSON."""
    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.ndarray,)):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    raise TypeError(f"Unsupported type: {type(value)}")


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

    metrics_by_split:
        {
            "train": {"MAE": ..., "RMSE": ...},
            "validation": {"MAE": ..., "RMSE": ...},
            "test": {"MAE": ..., "RMSE": ...}
        }
    """

    # Создание директорий, если они отсутствуют
    TRAINED_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Генерация уникального ID эксперимента
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"exp_{timestamp}"

    # Подготовка имени файла
    model_slug = model_name.lower().replace(" ", "_")

    # Сохранение обученной модели
    model_path = TRAINED_DIR / f"{model_slug}_{experiment_id}.joblib"
    joblib.dump(model, model_path)

    # Формирование подробной информации об эксперименте
    experiment = {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "random_state": random_state,
        "parameters": params,
        "features": features,
        "dataset_info": dataset_info,
        "metrics": metrics_by_split,
        "model_artifact": str(model_path.relative_to(PROJECT_ROOT)),
    }

    # Сохранение подробной информации в JSON
    json_path = EXPERIMENTS_DIR / f"{experiment_id}.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            experiment,
            file,
            ensure_ascii=False,
            indent=4,
            default=_json_serializer,
        )

    # Подготовка строки для общей таблицы экспериментов
    registry_row = {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "created_at": experiment["created_at"],
        "random_state": random_state,
        "model_artifact": experiment["model_artifact"],
    }

    # Добавление параметров модели
    for param_name, param_value in params.items():
        registry_row[f"param_{param_name}"] = param_value

    # Добавление метрик каждой выборки
    for split_name, metrics in metrics_by_split.items():
        for metric_name, metric_value in metrics.items():
            registry_row[f"{split_name}_{metric_name}"] = metric_value

    # Добавление количества наблюдений
    for split_name, info in dataset_info.items():
        if isinstance(info, dict) and "observations" in info:
            registry_row[f"{split_name}_observations"] = info["observations"]

    # Обновление общей таблицы
    registry_df = pd.DataFrame([registry_row])

    registry_df.to_csv(
        REGISTRY_CSV,
        mode="a",
        header=not REGISTRY_CSV.exists(),
        index=False,
        encoding="utf-8-sig",
    )

    return experiment_id