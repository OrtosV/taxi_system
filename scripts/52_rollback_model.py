
from pathlib import Path
import json
import sys
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.logging_config import setup_logging


REGISTRY_PATH = (
    PROJECT_ROOT / "models" / "registry" / "model_registry.json"
)

DEGRADATION_THRESHOLD = 0.0


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=4
        )


def get_test_metrics(model_record):
    """
    Получает Test-метрики модели.
    Сначала проверяет реестр, затем исходный JSON эксперимента.
    """

    test_metrics = model_record.get("test_metrics")

    if test_metrics:
        return test_metrics

    experiment_id = model_record.get("experiment_id")

    if not experiment_id:
        raise ValueError(
            "У модели отсутствует experiment_id"
        )

    experiment_path = (
        PROJECT_ROOT
        / "models"
        / "registry"
        / "experiments"
        / f"{experiment_id}.json"
    )

    if not experiment_path.exists():
        raise FileNotFoundError(experiment_path)

    experiment = load_json(experiment_path)

    test_metrics = experiment.get("metrics", {}).get("test")

    if not test_metrics:
        raise ValueError(
            f"В эксперименте {experiment_id} отсутствуют Test-метрики"
        )

    return test_metrics


def main():
    logger = setup_logging()
    logger.info("Проверка необходимости отката модели")

    registry = load_json(REGISTRY_PATH)

    active_version = registry["active_version"]
    previous_version = registry["previous_version"]

    if previous_version is None:
        raise ValueError(
            "Нет предыдущей версии для отката. "
            "Сначала необходимо продвинуть кандидата."
        )

    active_model = registry["models"][active_version]
    previous_model = registry["models"][previous_version]

    # Получаем Test-метрики из реестра или JSON экспериментов.
    active_test = get_test_metrics(active_model)
    previous_test = get_test_metrics(previous_model)

    active_mae = float(active_test["MAE"])
    previous_mae = float(previous_test["MAE"])

    active_wape = float(active_test["WAPE_percent"])
    previous_wape = float(previous_test["WAPE_percent"])

    # Проверяем ухудшение обеих метрик.
    mae_degraded = (
        active_mae > previous_mae * (1 + DEGRADATION_THRESHOLD)
    )

    wape_degraded = (
        active_wape > previous_wape * (1 + DEGRADATION_THRESHOLD)
    )

    rollback_required = mae_degraded and wape_degraded

    print("\nПроверка качества моделей:")
    print("-" * 55)

    print(f"Активная версия: {active_version}")
    print(f"Предыдущая версия: {previous_version}")

    print("\nTest MAE:")
    print(f"Активная модель: {active_mae:.6f}")
    print(f"Предыдущая модель: {previous_mae:.6f}")

    print("\nTest WAPE:")
    print(f"Активная модель: {active_wape:.6f}%")
    print(f"Предыдущая модель: {previous_wape:.6f}%")

    if not rollback_required:
        print("\nОткат не требуется.")
        logger.info("Откат модели не требуется")
        return

    # Возвращаем предыдущую модель в production.
    active_model["status"] = "rejected"
    previous_model["status"] = "production"

    registry["active_version"] = previous_version
    registry["previous_version"] = active_version

    registry["history"].append({
        "action": "model_rollback",
        "from_version": active_version,
        "to_version": previous_version,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "reason": (
            "Ухудшение Test MAE и Test WAPE "
            "относительно предыдущей версии"
        ),
    })

    save_json(REGISTRY_PATH, registry)

    print("\nОткат выполнен.")
    print(f"Новая активная версия: {registry['active_version']}")

    logger.info(
        "Откат модели завершён: %s -> %s",
        active_version,
        previous_version
    )


if __name__ == "__main__":
    main()
