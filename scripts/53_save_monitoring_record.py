from pathlib import Path
import json
import sys
from datetime import datetime, timezone

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.logging_config import setup_logging

MONITORING_DIR = PROJECT_ROOT / "reports" / "monitoring"
REGISTRY_PATH = PROJECT_ROOT / "models" / "registry" / "model_registry.json"

DATA_DRIFT_PATH = MONITORING_DIR / "data_drift.csv"
PERFORMANCE_PATH = MONITORING_DIR / "performance_drift.csv"
DECISION_PATH = MONITORING_DIR / "retraining_decision.csv"

RECORDS_PATH = MONITORING_DIR / "monitoring_records.jsonl"


def main():
    logger = setup_logging()
    logger.info("Формирование записи мониторинга")

    required_files = [
        DATA_DRIFT_PATH,
        PERFORMANCE_PATH,
        DECISION_PATH,
        REGISTRY_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(path)

    data_drift = pd.read_csv(DATA_DRIFT_PATH)
    performance = pd.read_csv(PERFORMANCE_PATH)
    decision = pd.read_csv(DECISION_PATH)

    with open(REGISTRY_PATH, "r", encoding="utf-8") as file:
        registry = json.load(file)

    decision_row = decision.iloc[0]

    record = {
        "run_id": datetime.now(timezone.utc).strftime(
            "monitor_%Y%m%d_%H%M%S"
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_model_version": registry["active_version"],
        "data_drift": {
            "features_checked": int(data_drift["feature"].nunique()),
            "significant_features": data_drift.loc[
                data_drift["PSI"] >= 0.2,
                "feature"
            ].drop_duplicates().tolist(),
            "max_psi": float(data_drift["PSI"].max()),
        },
        "performance": {
            "periods_checked": int(len(performance)),
            "mean_MAE": float(performance["MAE"].mean()),
            "mean_WAPE_percent": float(
                performance["WAPE_percent"].mean()
            ),
            "periods": performance.to_dict(orient="records"),
        },
        "retraining": {
            "decision": str(decision_row["decision"]),
            "significant_data_drift": bool(
                decision_row["significant_data_drift"]
            ),
            "consecutive_performance_degradation": bool(
                decision_row[
                    "consecutive_performance_degradation"
                ]
            ),
        },
        "model_registry_history_events": len(
            registry.get("history", [])
        ),
    }

    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    with open(
            RECORDS_PATH,
            "a",
            encoding="utf-8"
    ) as file:
        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )

    print("\nЗапись мониторинга сохранена:")
    print(f"ID: {record['run_id']}")
    print(f"Активная версия: {record['active_model_version']}")
    print(
        "Признаки с существенным Data Drift:",
        record["data_drift"]["significant_features"]
    )
    print(
        "Решение:",
        record["retraining"]["decision"]
    )
    print(f"Файл: {RECORDS_PATH}")

    logger.info("Запись мониторинга сохранена")


if __name__ == "__main__":
    main()
