
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

CANDIDATE_EXPERIMENT_ID = "exp_20261110_171542"
CANDIDATE_VERSION = "1.1.0"

CANDIDATE_EXPERIMENT_PATH = (
    PROJECT_ROOT
    / "models"
    / "registry"
    / "experiments"
    / f"{CANDIDATE_EXPERIMENT_ID}.json"
)


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


def main():
    logger = setup_logging()
    logger.info("Регистрация модели-кандидата")

    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(REGISTRY_PATH)

    if not CANDIDATE_EXPERIMENT_PATH.exists():
        raise FileNotFoundError(CANDIDATE_EXPERIMENT_PATH)

    registry = load_json(REGISTRY_PATH)
    experiment = load_json(CANDIDATE_EXPERIMENT_PATH)

    # Проверяем структуру эксперимента.
    required_sections = [
        "features",
        "parameters",
        "metrics",
        "model_artifact",
        "model_name",
    ]

    for section in required_sections:
        if section not in experiment:
            raise KeyError(
                f"В эксперименте отсутствует поле {section}"
            )

    if "validation" not in experiment["metrics"]:
        raise KeyError(
            "В эксперименте отсутствуют метрики Validation"
        )

    if CANDIDATE_VERSION in registry["models"]:
        raise ValueError(
            f"Версия {CANDIDATE_VERSION} уже зарегистрирована"
        )

    # Получаем активную модель.
    active_version = registry["active_version"]
    active_model = registry["models"][active_version]

    active_metrics = active_model["validation_metrics"]
    candidate_metrics = experiment["metrics"]["validation"]

    # Проверяем наличие нужных метрик.
    for metrics_name, metrics in [
        ("active", active_metrics),
        ("candidate", candidate_metrics),
    ]:
        for metric in ["MAE", "WAPE_percent"]:
            if metric not in metrics:
                raise KeyError(
                    f"В метриках {metrics_name} отсутствует {metric}"
                )

    active_mae = float(active_metrics["MAE"])
    candidate_mae = float(candidate_metrics["MAE"])

    active_wape = float(active_metrics["WAPE_percent"])
    candidate_wape = float(candidate_metrics["WAPE_percent"])

    # Кандидат должен улучшить обе основные метрики.
    candidate_passed = (
        candidate_mae < active_mae
        and candidate_wape < active_wape
    )

    status = "approved" if candidate_passed else "rejected"

    # Проверяем наличие файла обученной модели.
    artifact_path = (
        PROJECT_ROOT / experiment["model_artifact"]
    )

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Артефакт модели не найден: {artifact_path}"
        )

    candidate_record = {
        "version": CANDIDATE_VERSION,
        "experiment_id": CANDIDATE_EXPERIMENT_ID,
        "model_name": experiment["model_name"],
        "artifact_path": experiment["model_artifact"],
        "status": status,
        "registered_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "features": experiment["features"],
        "params": experiment["parameters"],
        "validation_metrics": candidate_metrics,
        "comparison": {
            "compared_with_version": active_version,
            "active_MAE": active_mae,
            "candidate_MAE": candidate_mae,
            "active_WAPE": active_wape,
            "candidate_WAPE": candidate_wape,
            "passed": candidate_passed,
        },
    }

    registry["models"][CANDIDATE_VERSION] = candidate_record

    registry["history"].append({
        "action": "candidate_evaluation",
        "version": CANDIDATE_VERSION,
        "experiment_id": CANDIDATE_EXPERIMENT_ID,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "reason": (
            "Кандидат улучшил обе основные метрики"
            if candidate_passed
            else (
                "Кандидат отклонён: "
                "не удалось улучшить одновременно MAE и WAPE"
            )
        ),
    })

    save_json(REGISTRY_PATH, registry)

    print("\nРезультат проверки модели-кандидата:")
    print("-" * 55)

    print(f"Активная версия: {active_version}")
    print(f"Версия кандидата: {CANDIDATE_VERSION}")

    print("\nСравнение качества:")
    print(f"Активный MAE: {active_mae:.6f}")
    print(f"Кандидат MAE: {candidate_mae:.6f}")

    print(f"\nАктивный WAPE: {active_wape:.6f}%")
    print(f"Кандидат WAPE: {candidate_wape:.6f}%")

    print(f"\nСтатус кандидата: {status}")
    print(
        f"Активная версия после проверки: "
        f"{registry['active_version']}"
    )

    logger.info(
        "Проверка кандидата завершена: %s",
        status
    )


if __name__ == "__main__":
    main()
