
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


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


RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "final_test_metrics.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_selection.md"
)

FINAL_EXPERIMENT_ID = "exp_20261110_153017"
FINAL_MODEL_NAME = "XGBoost"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=4,
        )


def update_experiment_metrics(results_df):
    """Обновляет тестовые метрики во всех JSON экспериментов."""

    updated_count = 0

    for _, result in results_df.iterrows():
        experiment_id = result["experiment_id"]

        json_path = (
            EXPERIMENTS_DIR / f"{experiment_id}.json"
        )

        if not json_path.exists():
            raise FileNotFoundError(
                f"Не найден JSON эксперимента: {json_path}"
            )

        experiment = load_json(json_path)

        test_metrics = {
            "MAE": float(result["MAE"]),
            "MSE": float(result["MSE"]),
            "RMSE": float(result["RMSE"]),
            "WAPE_percent": float(result["WAPE_percent"]),
            "R2": float(result["R2"]),
        }

        experiment.setdefault("metrics", {})
        experiment["metrics"]["test"] = test_metrics

        # Дополнительно сохраняем время прогнозирования.
        prediction_time = result.get(
            "prediction_time_seconds"
        )

        if pd.notna(prediction_time):
            experiment["test_prediction_time_seconds"] = (
                float(prediction_time)
            )
        else:
            experiment["test_prediction_time_seconds"] = None

        save_json(json_path, experiment)

        updated_count += 1

        logger.info(
            "Обновлены тестовые метрики: %s",
            experiment_id,
        )

    return updated_count


def register_final_model(results_df):
    """Фиксирует выбранную финальную модель в её JSON."""

    selected_row = results_df.loc[
        results_df["experiment_id"] == FINAL_EXPERIMENT_ID
    ]

    if selected_row.empty:
        raise ValueError(
            "Финальная модель отсутствует в итоговой таблице."
        )

    selected_row = selected_row.iloc[0]

    experiment_path = (
        EXPERIMENTS_DIR
        / f"{FINAL_EXPERIMENT_ID}.json"
    )

    experiment = load_json(experiment_path)

    experiment["model_selection"] = {
        "is_final_model": True,
        "selection_date": datetime.now().isoformat(
            timespec="seconds"
        ),
        "selection_reason": (
            "Модель показала лучший результат на тестовой "
            "выборке по всем пяти используемым метрикам."
        ),
        "test_metrics": {
            "MAE": float(selected_row["MAE"]),
            "MSE": float(selected_row["MSE"]),
            "RMSE": float(selected_row["RMSE"]),
            "WAPE_percent": float(
                selected_row["WAPE_percent"]
            ),
            "R2": float(selected_row["R2"]),
        },
    }

    save_json(experiment_path, experiment)

    logger.info(
        "Финальная модель зарегистрирована: %s",
        FINAL_MODEL_NAME,
    )


def create_selection_report(results_df):
    """Формирует Markdown-отчёт по выбору модели."""

    final_row = results_df.loc[
        results_df["experiment_id"] == FINAL_EXPERIMENT_ID
    ].iloc[0]

    baseline_row = results_df.loc[
        results_df["model_name"] == "Seasonal Naive"
    ].iloc[0]

    mae_improvement = (
        1 - final_row["MAE"] / baseline_row["MAE"]
    ) * 100

    report_lines = [
        "# Выбор финальной модели",
        "",
        "## 1. Цель сравнения",
        "",
        "Целью этапа являлся выбор модели прогнозирования "
        "почасового спроса на поездки такси по зонам Нью-Йорка.",
        "",
        "Сравнение выполнено на тестовой выборке, содержащей "
        "685 641 наблюдение.",
        "",
        "Для оценки использовались пять метрик: MAE, MSE, "
        "RMSE, WAPE и R².",
        "",
        "## 2. Результаты сравнения",
        "",
        "| Модель | MAE | MSE | RMSE | WAPE, % | R² |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    sorted_results = results_df.sort_values(
        "MAE",
        ascending=True,
    )

    for _, row in sorted_results.iterrows():
        report_lines.append(
            f"| {row['model_name']} "
            f"| {row['MAE']:.4f} "
            f"| {row['MSE']:.4f} "
            f"| {row['RMSE']:.4f} "
            f"| {row['WAPE_percent']:.4f} "
            f"| {row['R2']:.4f} |"
        )

    report_lines.extend([
        "",
        "## 3. Выбор финальной модели",
        "",
        f"В качестве финальной модели выбрана **{FINAL_MODEL_NAME}**.",
        "",
        f"Идентификатор эксперимента: `{FINAL_EXPERIMENT_ID}`.",
        "",
        "Исходная конфигурация XGBoost показала лучший результат "
        "по всем пяти метрикам на тестовой выборке.",
        "",
        "Показатели финальной модели:",
        "",
        f"- MAE: {final_row['MAE']:.4f}",
        f"- MSE: {final_row['MSE']:.4f}",
        f"- RMSE: {final_row['RMSE']:.4f}",
        f"- WAPE: {final_row['WAPE_percent']:.4f}%",
        f"- R²: {final_row['R2']:.4f}",
        "",
        "## 4. Сравнение с базовой моделью",
        "",
        "В качестве базового алгоритма использовалась "
        "Seasonal Naive.",
        "",
        f"MAE базовой модели: {baseline_row['MAE']:.4f}.",
        f"MAE XGBoost: {final_row['MAE']:.4f}.",
        "",
        f"Относительное снижение MAE составило "
        f"{mae_improvement:.2f}%.",
        "",
        "## 5. Обоснование выбора",
        "",
        "Исходная модель XGBoost продемонстрировала более "
        "высокое качество на тестовой выборке, чем остальные "
        "рассматриваемые алгоритмы, включая их настроенные версии.",
        "",
        "Настройка гиперпараметров улучшила результаты XGBoost "
        "на валидационной выборке, однако не обеспечила "
        "аналогичного улучшения на тестовой выборке. Поэтому "
        "для дальнейшей разработки выбрана исходная конфигурация.",
        "",
        "Тестовая выборка ранее просматривалась в ходе "
        "первоначального сравнения моделей. Поэтому её нельзя "
        "считать полностью нетронутой контрольной выборкой. "
        "После данного этапа тестовые результаты не используются "
        "для дальнейшего подбора гиперпараметров.",
        "",
        "## 6. Дальнейшее использование",
        "",
        "Выбранная модель будет использоваться для разработки "
        "интерпретации прогнозов, прогнозного модуля и "
        "пользовательского интерфейса системы.",
        "",
    ])

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    logger.info(
        "Отчёт о выборе модели сохранён: %s",
        REPORT_PATH,
    )


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Не найдена таблица результатов: {RESULTS_PATH}"
        )

    results_df = pd.read_csv(RESULTS_PATH)

    required_columns = {
        "experiment_id",
        "model_name",
        "MAE",
        "MSE",
        "RMSE",
        "WAPE_percent",
        "R2",
        "prediction_time_seconds",
    }

    missing_columns = required_columns - set(
        results_df.columns
    )

    if missing_columns:
        raise ValueError(
            f"В таблице отсутствуют столбцы: {missing_columns}"
        )

    if len(results_df) != 9:
        raise ValueError(
            f"Ожидалось 9 экспериментов, получено: "
            f"{len(results_df)}"
        )

    logger.info(
        "Загружено результатов: %s",
        len(results_df),
    )

    updated_count = update_experiment_metrics(
        results_df
    )

    register_final_model(results_df)

    _rebuild_registry_csv()

    create_selection_report(results_df)

    logger.info(
        "Обновлено JSON-файлов: %s",
        updated_count,
    )

    logger.info(
        "Этап выбора финальной модели завершён."
    )


if __name__ == "__main__":
    main()
