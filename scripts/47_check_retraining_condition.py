
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.logging_config import setup_logging


PERFORMANCE_PATH = (
    PROJECT_ROOT / "reports" / "monitoring" / "performance_drift.csv"
)

DATA_DRIFT_PATH = (
    PROJECT_ROOT / "reports" / "monitoring" / "data_drift.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "monitoring" / "retraining_decision.csv"
)

BASELINE_MAE = 4.072391
BASELINE_WAPE = 19.711498

DEGRADATION_THRESHOLD = 0.15
CONSECUTIVE_PERIODS = 2
PSI_THRESHOLD = 0.2


def main():
    logger = setup_logging()
    logger.info("Проверка условия переобучения модели")

    performance = pd.read_csv(PERFORMANCE_PATH)
    data_drift = pd.read_csv(DATA_DRIFT_PATH)

    performance = performance.sort_values("period").reset_index(drop=True)

    mae_limit = BASELINE_MAE * (1 + DEGRADATION_THRESHOLD)
    wape_limit = BASELINE_WAPE * (1 + DEGRADATION_THRESHOLD)

    performance["mae_limit"] = mae_limit
    performance["wape_limit"] = wape_limit

    performance["mae_degraded"] = (
        performance["MAE"] > mae_limit
    )

    performance["wape_degraded"] = (
        performance["WAPE_percent"] > wape_limit
    )

    performance["quality_degraded"] = (
        performance["mae_degraded"]
        | performance["wape_degraded"]
    )

    # Проверяем наличие двух последовательных периодов
    # с ухудшением хотя бы одной из основных метрик.
    consecutive_degradation = False

    for index in range(
        CONSECUTIVE_PERIODS - 1,
        len(performance)
    ):
        recent_periods = performance.iloc[
            index - CONSECUTIVE_PERIODS + 1:index + 1
        ]

        if recent_periods["quality_degraded"].all():
            consecutive_degradation = True
            break

    # Data Drift используется как дополнительный сигнал.
    significant_drift = (
        data_drift["PSI"] >= PSI_THRESHOLD
    ).any()

    if consecutive_degradation:
        decision = "Требуется переобучение"
    elif significant_drift:
        decision = "Наблюдение: обнаружен Data Drift"
    else:
        decision = "Переобучение не требуется"

    result = pd.DataFrame([{
        "baseline_MAE": BASELINE_MAE,
        "baseline_WAPE": BASELINE_WAPE,
        "MAE_limit": mae_limit,
        "WAPE_limit": wape_limit,
        "significant_data_drift": significant_drift,
        "consecutive_performance_degradation": consecutive_degradation,
        "decision": decision,
    }])

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nПроверка условий переобучения:")
    print(f"Базовый MAE: {BASELINE_MAE:.4f}")
    print(f"Порог MAE: {mae_limit:.4f}")
    print(f"Базовый WAPE: {BASELINE_WAPE:.4f}%")
    print(f"Порог WAPE: {wape_limit:.4f}%")
    print(f"Обнаружен Data Drift: {significant_drift}")
    print(
        "Ухудшение качества два периода подряд: "
        f"{consecutive_degradation}"
    )
    print(f"\nРешение системы: {decision}")

    logger.info("Проверка условий переобучения завершена")


if __name__ == "__main__":
    main()
