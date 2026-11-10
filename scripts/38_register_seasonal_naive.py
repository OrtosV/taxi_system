import json
import logging
import sys
from datetime import datetime
from pathlib import Path


# Добавляем корень проекта в путь поиска модулей
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.experiments.registry import (
    EXPERIMENTS_DIR,
    _rebuild_registry_csv,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    EXPERIMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiment_id = (
        f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        "_seasonal_naive"
    )

    experiment = {
        "experiment_id": experiment_id,
        "model_name": "Seasonal Naive",
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "random_state": None,

        "parameters": {
            "method": "Seasonal Naive",
            "forecast_horizon_hours": 1,
            "seasonality_hours": 24,
            "prediction_rule": (
                "Прогноз спроса на следующий час "
                "равен спросу в соответствующий час "
                "предыдущего дня"
            ),
        },

        "metrics": {
            "train": {
                "MAE": 5.909499,
                "MSE": 478.162453,
                "RMSE": 21.866926,
                "WAPE_percent": 31.599147,
                "R2": 0.850229,
            },
            "validation": {
                "MAE": 6.455390,
                "MSE": 461.814316,
                "RMSE": 21.489865,
                "WAPE_percent": 31.245874,
                "R2": 0.856990,
            },
            "test": {
                "MAE": 6.842111,
                "MSE": 555.023679,
                "RMSE": 23.558941,
                "WAPE_percent": 30.656566,
                "R2": 0.858478,
            },
        },

        "features": [
            "trip_count_lag_24",
        ],

        "dataset_info": {
            "method": "Deterministic baseline",
            "forecast_horizon_hours": 1,
        },

        # У базового алгоритма нет обученного объекта.
        "model_artifact": None,
    }

    experiment_path = (
        EXPERIMENTS_DIR / f"{experiment_id}.json"
    )

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
        )

    logger.info(
        "Эксперимент Seasonal Naive сохранён: %s",
        experiment_id,
    )

    _rebuild_registry_csv()

    logger.info(
        "Общая таблица экспериментов обновлена."
    )


if __name__ == "__main__":
    main()
