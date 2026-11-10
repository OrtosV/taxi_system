
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

EXPERIMENT_ID = "exp_20261110_173122"
CANDIDATE_VERSION = "1.2.0"

EXPERIMENT_PATH = (
    PROJECT_ROOT
    / "models"
    / "registry"
    / "experiments"
    / f"{EXPERIMENT_ID}.json"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def main():
    logger = setup_logging()
    logger.info("Регистрация кандидата XGBoost Tuned")

    registry = load_json(REGISTRY_PATH)
    experiment = load_json(EXPERIMENT_PATH)

    if CANDIDATE_VERSION in registry["models"]:
        raise ValueError(
            f"Версия {CANDIDATE_VERSION} уже существует"
        )

    artifact_path = PROJECT_ROOT / experiment["model_artifact"]

    if not artifact_path.exists():
        raise FileNotFoundError(artifact_path)

    active_version = registry["active_version"]
    active_model = registry["models"][active_version]

    active_metrics = active_model["validation_metrics"]
    candidate_metrics = experiment["metrics"]["validation"]

    active_mae = float(active_metrics["MAE"])
    candidate_mae = float(candidate_metrics["MAE"])

    active_wape = float(active_metrics["WAPE_percent"])
    candidate_wape = float(candidate_metrics["WAPE_percent"])

    approved = (
        candidate_mae < active_mae
        and candidate_wape < active_wape
    )

    status = "approved" if approved else "rejected"

    candidate_record = {
        "version": CANDIDATE_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "model_name": experiment["model_name"],
        "artifact_path": experiment["model_artifact"],
        "status": status,
        "registered_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "features": experiment["features"],
        "params": experiment["parameters"],
        "validation_metrics": candidate_metrics,
        "test_metrics": experiment["metrics"].get("test", {}),
        "comparison": {
            "compared_with_version": active_version,
            "active_MAE": active_mae,
            "candidate_MAE": candidate_mae,
            "active_WAPE": active_wape,
            "candidate_WAPE": candidate_wape,
            "passed": approved,
        },
    }

    registry["models"][CANDIDATE_VERSION] = candidate_record

    registry["history"].append({
        "action": "candidate_evaluation",
        "version": CANDIDATE_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "reason": (
            "Кандидат прошёл проверку Validation"
            if approved
            else "Кандидат не улучшил обе метрики Validation"
        ),
    })

    save_json(REGISTRY_PATH, registry)

    print("\nПроверка XGBoost Tuned:")
    print(f"Активная версия: {active_version}")
    print(f"Кандидат: {CANDIDATE_VERSION}")
    print(f"Активный Validation MAE: {active_mae:.6f}")
    print(f"Кандидат Validation MAE: {candidate_mae:.6f}")
    print(f"Активный Validation WAPE: {active_wape:.6f}%")
    print(f"Кандидат Validation WAPE: {candidate_wape:.6f}%")
    print(f"\nСтатус кандидата: {status}")

    logger.info("Регистрация кандидата завершена: %s", status)


if __name__ == "__main__":
    main()
