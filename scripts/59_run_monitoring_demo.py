from __future__ import annotations

import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import calculate_metrics
from src.logging_config import setup_logging


TRAIN_PATH = PROJECT_ROOT / "data/processed/modeling/train.parquet"
VALIDATION_PATH = PROJECT_ROOT / "data/processed/modeling/validation.parquet"
REGISTRY_PATH = PROJECT_ROOT / "models/registry/model_registry.json"

ACTIVE_VERSION = "1.0.0"
CANDIDATE_VERSION = "1.2.0"
CANDIDATE_EXPERIMENT_ID = "exp_20261110_173122"

FEATURES_TO_MONITOR = [
    "demand_lag_1h",
    "demand_lag_168h",
    "temperature_2m_prev_hour",
    "precipitation_prev_hour",
    "wind_speed_10m_prev_hour",
    "total_311_count_prev_hour",
]

CATEGORICAL_FEATURES = ["zone_id", "Borough", "Zone", "service_zone"]
RANDOM_STATE = 42
SAMPLE_SIZE = 100_000
N_BINS = 10
EPSILON = 1e-6
BASELINE_MAE = 4.072391
BASELINE_WAPE = 19.711498
DEGRADATION_THRESHOLD = 0.15
PSI_THRESHOLD = 0.2


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Объект типа {type(value).__name__} не сериализуется в JSON")


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )


def calculate_psi(reference, current) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) == 0 or len(current) == 0:
        return float("nan")

    edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, N_BINS + 1))
    )

    if len(edges) == 1:
        edges = np.array([-np.inf, np.inf])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf

    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)

    reference_share = np.clip(reference_counts / len(reference), EPSILON, None)
    current_share = np.clip(current_counts / len(current), EPSILON, None)

    return float(
        np.sum(
            (current_share - reference_share)
            * np.log(current_share / reference_share)
        )
    )


def create_shifted_sample(validation: pd.DataFrame) -> pd.DataFrame:
    """Создаёт демонстрационный сдвиг признаков, не меняя исходные файлы."""
    sample_size = min(SAMPLE_SIZE, len(validation))
    sample = validation[FEATURES_TO_MONITOR].sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    ).copy()

    # Искусственно меняем распределения нескольких входных признаков.
    sample["demand_lag_1h"] = sample["demand_lag_1h"] * 1.5
    sample["demand_lag_168h"] = sample["demand_lag_168h"] * 1.25
    sample["temperature_2m_prev_hour"] = (
        sample["temperature_2m_prev_hour"] + 12.0
    )
    sample["precipitation_prev_hour"] = (
        sample["precipitation_prev_hour"] * 2.0 + 2.0
    )
    sample["wind_speed_10m_prev_hour"] = (
        sample["wind_speed_10m_prev_hour"] + 15.0
    )
    sample["total_311_count_prev_hour"] = (
        sample["total_311_count_prev_hour"] * 1.5 + 2.0
    )

    return sample


def run_data_drift_check(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    reference = train[FEATURES_TO_MONITOR].sample(
        n=min(SAMPLE_SIZE, len(train)),
        random_state=RANDOM_STATE,
    )
    shifted = create_shifted_sample(validation)

    rows = []
    for feature in FEATURES_TO_MONITOR:
        psi = calculate_psi(
            reference[feature].dropna(),
            shifted[feature].dropna(),
        )

        if pd.isna(psi):
            status = "Недостаточно данных"
        elif psi < 0.1:
            status = "Незначительное изменение"
        elif psi < PSI_THRESHOLD:
            status = "Умеренное изменение"
        else:
            status = "Существенное изменение"

        rows.append({
            "feature": feature,
            "PSI": psi,
            "status": status,
            "reference_rows": int(reference[feature].notna().sum()),
            "shifted_rows": int(shifted[feature].notna().sum()),
        })

    return pd.DataFrame(rows)


def run_performance_check(validation: pd.DataFrame) -> pd.DataFrame:
    model_path = (
        PROJECT_ROOT
        / "models/trained/xgboost_exp_20261110_153017.joblib"
    )
    model = joblib.load(model_path)

    features = list(model.feature_names_in_)
    X = validation[features].copy()

    for column in CATEGORICAL_FEATURES:
        if column in X.columns:
            X[column] = X[column].fillna("Unknown").astype(str)

    predictions = model.predict(X)

    results = pd.DataFrame({
        "forecast_datetime": pd.to_datetime(validation["forecast_datetime"]),
        "y_true": validation["target_next_hour"].to_numpy(),
        "y_pred": predictions,
    })
    results["period"] = results["forecast_datetime"].dt.to_period("M").astype(str)

    rows = []
    for period, group in results.groupby("period", sort=True):
        rows.append({
            "period": period,
            "rows": int(len(group)),
            **calculate_metrics(group["y_true"], group["y_pred"]),
        })

    return pd.DataFrame(rows)


def evaluate_performance_drift(performance: pd.DataFrame) -> dict:
    mae_limit = BASELINE_MAE * (1 + DEGRADATION_THRESHOLD)
    wape_limit = BASELINE_WAPE * (1 + DEGRADATION_THRESHOLD)

    checked = performance.sort_values("period").copy()
    checked["mae_degraded"] = checked["MAE"] > mae_limit
    checked["wape_degraded"] = checked["WAPE_percent"] > wape_limit
    checked["quality_degraded"] = (
        checked["mae_degraded"] | checked["wape_degraded"]
    )

    consecutive = False
    for index in range(1, len(checked)):
        if checked.iloc[index - 1:index + 1]["quality_degraded"].all():
            consecutive = True
            break

    return {
        "baseline_MAE": BASELINE_MAE,
        "baseline_WAPE_percent": BASELINE_WAPE,
        "MAE_limit": mae_limit,
        "WAPE_limit_percent": wape_limit,
        "consecutive_performance_degradation": bool(consecutive),
        "decision": (
            "Требуется переобучение"
            if consecutive
            else "Ухудшение качества не подтверждено"
        ),
    }


def get_test_metrics(model_record: dict) -> dict:
    """Берёт Test-метрики из реестра либо JSON исходного эксперимента."""
    metrics = model_record.get("test_metrics")
    if metrics:
        return metrics

    experiment_id = model_record.get("experiment_id")
    if not experiment_id:
        raise ValueError("У модели отсутствует experiment_id")

    experiment_path = (
        PROJECT_ROOT
        / "models/registry/experiments"
        / f"{experiment_id}.json"
    )
    if not experiment_path.exists():
        raise FileNotFoundError(
            f"Не найден JSON эксперимента для Test-метрик: {experiment_path}"
        )

    experiment = load_json(experiment_path)
    metrics = experiment.get("metrics", {}).get("test")
    if not metrics:
        raise ValueError(
            f"В эксперименте {experiment_id} отсутствуют Test-метрики"
        )
    return metrics


def simulate_candidate_lifecycle(source_registry: dict) -> tuple[dict, list]:
    """
    Воспроизводит продвижение и откат только в копии реестра.
    Production model_registry.json не изменяется.
    """
    demo_registry = copy.deepcopy(source_registry)
    events = []

    if demo_registry.get("active_version") != ACTIVE_VERSION:
        raise ValueError(
            f"Ожидалась рабочая версия {ACTIVE_VERSION}, "
            f"получена {demo_registry.get('active_version')}"
        )

    active = demo_registry["models"][ACTIVE_VERSION]
    candidate = demo_registry["models"][CANDIDATE_VERSION]

    active_validation = active["validation_metrics"]
    candidate_validation = candidate["validation_metrics"]

    candidate_approved = (
        float(candidate_validation["MAE"])
        < float(active_validation["MAE"])
        and float(candidate_validation["WAPE_percent"])
        < float(active_validation["WAPE_percent"])
    )

    events.append({
        "stage": "Оценка кандидата на Validation",
        "action": "candidate_evaluation",
        "from_version": ACTIVE_VERSION,
        "candidate_version": CANDIDATE_VERSION,
        "approved": bool(candidate_approved),
        "reason": (
            "Кандидат улучшил MAE и WAPE на Validation"
            if candidate_approved
            else "Кандидат не улучшил обе метрики на Validation"
        ),
    })

    if not candidate_approved:
        return demo_registry, events

    active["status"] = "previous"
    candidate["status"] = "production"
    demo_registry["previous_version"] = ACTIVE_VERSION
    demo_registry["active_version"] = CANDIDATE_VERSION

    events.append({
        "stage": "Продвижение кандидата",
        "action": "model_promotion",
        "from_version": ACTIVE_VERSION,
        "to_version": CANDIDATE_VERSION,
        "reason": "Кандидат прошёл проверку Validation",
    })

    active_test = get_test_metrics(active)
    candidate_test = get_test_metrics(candidate)

    active_test_mae = float(active_test["MAE"])
    candidate_test_mae = float(candidate_test["MAE"])
    active_test_wape = float(active_test["WAPE_percent"])
    candidate_test_wape = float(candidate_test["WAPE_percent"])

    rollback_required = (
        candidate_test_mae > active_test_mae
        and candidate_test_wape > active_test_wape
    )

    events.append({
        "stage": "Контрольное сравнение Test",
        "action": "test_comparison",
        "active_candidate_version": CANDIDATE_VERSION,
        "reference_version": ACTIVE_VERSION,
        "candidate_MAE": candidate_test_mae,
        "reference_MAE": active_test_mae,
        "candidate_WAPE_percent": candidate_test_wape,
        "reference_WAPE_percent": active_test_wape,
        "rollback_required": bool(rollback_required),
        "reason": (
            "Кандидат уступил исходной модели по Test MAE и WAPE"
            if rollback_required
            else "Кандидат не показал ухудшения одновременно по MAE и WAPE"
        ),
    })

    if rollback_required:
        candidate["status"] = "rejected"
        active["status"] = "production"
        demo_registry["active_version"] = ACTIVE_VERSION
        demo_registry["previous_version"] = CANDIDATE_VERSION

        events.append({
            "stage": "Откат модели",
            "action": "model_rollback",
            "from_version": CANDIDATE_VERSION,
            "to_version": ACTIVE_VERSION,
            "reason": (
                "Демонстрационный откат: кандидат хуже на Test "
                "по MAE и WAPE"
            ),
        })

    return demo_registry, events


def main() -> None:
    logger = setup_logging()
    logger.info("Запуск демонстрационной проверки актуальности модели")

    for path in [TRAIN_PATH, VALIDATION_PATH, REGISTRY_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Не найден необходимый файл: {path}")

    train = pd.read_parquet(TRAIN_PATH)
    validation = pd.read_parquet(VALIDATION_PATH)
    source_registry = load_json(REGISTRY_PATH)

    if CANDIDATE_VERSION not in source_registry.get("models", {}):
        raise ValueError(
            f"В реестре отсутствует демонстрационный кандидат {CANDIDATE_VERSION}"
        )

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("demo_%Y%m%d_%H%M%S_%f")
    output_dir = PROJECT_ROOT / "reports/monitoring/demo_runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    logger.info("Расчёт Data Drift на искусственно смещённом наборе")
    data_drift = run_data_drift_check(train, validation)

    logger.info("Расчёт Performance Drift на исторической Validation")
    performance = run_performance_check(validation)
    performance_decision = evaluate_performance_drift(performance)

    demo_registry, lifecycle_events = simulate_candidate_lifecycle(
        source_registry
    )

    data_drift.to_csv(
        output_dir / "data_drift.csv",
        index=False,
        encoding="utf-8-sig",
    )
    performance.to_csv(
        output_dir / "performance_by_month.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_json(output_dir / "demo_registry.json", demo_registry)

    significant_features = data_drift.loc[
        data_drift["PSI"] >= PSI_THRESHOLD, "feature"
    ].tolist()

    summary = {
        "run_id": run_id,
        "timestamp": timestamp.isoformat(),
        "demo_only": True,
        "description": (
            "Демонстрация на исторических данных и искусственно "
            "смещённых признаках. Не является онлайн-мониторингом."
        ),
        "data_drift": {
            "features_checked": int(data_drift["feature"].nunique()),
            "significant_features": significant_features,
            "max_psi": float(data_drift["PSI"].max()),
            "records": data_drift.to_dict(orient="records"),
        },
        "performance_drift": {
            **performance_decision,
            "monthly_metrics": performance.to_dict(orient="records"),
        },
        "model_lifecycle": {
            "events": lifecycle_events,
            "initial_version": ACTIVE_VERSION,
            "candidate_version": CANDIDATE_VERSION,
            "final_active_version": demo_registry["active_version"],
            "production_registry_modified": False,
        },
        "output_directory": str(output_dir.relative_to(PROJECT_ROOT)),
    }

    save_json(output_dir / "summary.json", summary)

    records_path = PROJECT_ROOT / "reports/monitoring/demo_monitoring_records.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                summary,
                ensure_ascii=False,
                default=json_default,
            ) + "\n"
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nОтчёт сохранён: {output_dir}")
    print(f"Итоговая рабочая версия в демонстрации: {demo_registry['active_version']}")
    print("Рабочий model_registry.json не изменялся.")

    logger.info("Демонстрационная проверка завершена: %s", run_id)


if __name__ == "__main__":
    main()
