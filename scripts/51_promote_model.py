
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

CANDIDATE_VERSION = "1.2.0"


def main():
    logger = setup_logging()
    logger.info("Продвижение модели-кандидата")

    with open(REGISTRY_PATH, "r", encoding="utf-8") as file:
        registry = json.load(file)

    if CANDIDATE_VERSION not in registry["models"]:
        raise ValueError("Кандидат не зарегистрирован")

    candidate = registry["models"][CANDIDATE_VERSION]

    if candidate["status"] != "approved":
        raise ValueError(
            "Нельзя продвинуть кандидата без статуса approved"
        )

    artifact_path = PROJECT_ROOT / candidate["artifact_path"]

    if not artifact_path.exists():
        raise FileNotFoundError(artifact_path)

    old_version = registry["active_version"]

    if old_version == CANDIDATE_VERSION:
        raise ValueError("Эта версия уже активна")

    registry["models"][old_version]["status"] = "previous"
    candidate["status"] = "production"

    registry["previous_version"] = old_version
    registry["active_version"] = CANDIDATE_VERSION

    registry["history"].append({
        "action": "model_promotion",
        "from_version": old_version,
        "to_version": CANDIDATE_VERSION,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "reason": "Кандидат прошёл проверку Validation",
    })

    with open(REGISTRY_PATH, "w", encoding="utf-8") as file:
        json.dump(registry, file, ensure_ascii=False, indent=4)

    print("\nМодель успешно продвинута.")
    print(f"Предыдущая версия: {old_version}")
    print(f"Новая активная версия: {CANDIDATE_VERSION}")

    logger.info("Продвижение модели завершено")


if __name__ == "__main__":
    main()
