
from pathlib import Path
import json
import sys
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.logging_config import setup_logging


EXPERIMENT_ID = "exp_20261110_153017"
VERSION = "1.0.0"

REGISTRY_PATH = (
    PROJECT_ROOT / "models" / "registry" / "model_registry.json"
)

EXPERIMENT_PATH = (
    PROJECT_ROOT
    / "models"
    / "registry"
    / "experiments"
    / f"{EXPERIMENT_ID}.json"
)


def main():
    logger = setup_logging()
    logger.info("Проверка и восстановление реестра моделей")

    if not EXPERIMENT_PATH.exists():
        raise FileNotFoundError(EXPERIMENT_PATH)

    with open(EXPERIMENT_PATH, "r", encoding="utf-8") as file:
        experiment = json.load(file)

    required_sections = ["features", "parameters", "metrics", "model_artifact"]

    for section in required_sections:
        if section not in experiment:
            raise KeyError(
                f"В эксперименте отсутствует поле {section}"
            )

    if "validation" not in experiment["metrics"]:
        raise KeyError("В эксперименте отсутствуют метрики Validation")

    validation_metrics = experiment["metrics"]["validation"]

    if not REGISTRY_PATH.exists():
        registry = {
            "active_version": VERSION,
            "previous_version": None,
            "models": {},
            "history": [],
        }
    else:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as file:
            registry = json.load(file)

    if registry["active_version"] != VERSION:
        raise ValueError(
            "Активная версия отличается от ожидаемой. "
            "Автоматическое восстановление остановлено."
        )

    model_record = {
        "version": VERSION,
        "experiment_id": EXPERIMENT_ID,
        "model_name": experiment["model_name"],
        "artifact_path": experiment["model_artifact"],
        "status": "production",
        "registered_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "features": experiment["features"],
        "params": experiment["parameters"],
        "validation_metrics": validation_metrics,
    }

    # Исправляем только запись активной версии.
    # Остальные модели и история остаются без изменений.
    registry.setdefault("models", {})
    registry.setdefault("history", [])

    registry["models"][VERSION] = model_record

    registry["history"].append({
        "action": "registry_metadata_repaired",
        "version": VERSION,
        "experiment_id": EXPERIMENT_ID,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "reason": (
            "Восстановление параметров и метрик из JSON эксперимента"
        ),
    })

    REGISTRY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(REGISTRY_PATH, "w", encoding="utf-8") as file:
        json.dump(
            registry,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\nРеестр успешно восстановлен.")
    print(f"Активная версия: {registry['active_version']}")
    print(f"Модель: {model_record['model_name']}")
    print(f"Количество признаков: {len(model_record['features'])}")
    print(f"Количество параметров: {len(model_record['params'])}")
    print(
        f"Validation MAE: "
        f"{model_record['validation_metrics']['MAE']:.6f}"
    )
    print(
        f"Validation WAPE: "
        f"{model_record['validation_metrics']['WAPE_percent']:.6f}%"
    )

    logger.info("Реестр моделей восстановлен")


if __name__ == "__main__":
    main()
