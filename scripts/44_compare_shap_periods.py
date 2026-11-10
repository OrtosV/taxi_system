
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENT_ID = "exp_20261110_153017"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "trained"
    / f"xgboost_{EXPERIMENT_ID}.joblib"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling"
    / "validation.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "shap" / "period_comparison"

CSV_PATH = OUTPUT_DIR / "shap_period_comparison.csv"
PLOT_PATH = OUTPUT_DIR / "shap_period_comparison.png"

SAMPLE_SIZE = 5000
RANDOM_STATE = 42
TOP_N = 15

CATEGORICAL_FEATURES = [
    "zone_id",
    "Borough",
    "Zone",
    "service_zone",
]


# ============================================================
# 2. LOGGING
# ============================================================

from src.logging_config import setup_logging

logger = setup_logging()


# ============================================================
# 3. DATA LOADING
# ============================================================

def load_data():
    logger.info("Загрузка финальной модели.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {MODEL_PATH}"
        )

    pipeline = joblib.load(MODEL_PATH)

    logger.info("Загрузка Validation.")

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation не найден: {VALIDATION_PATH}"
        )

    validation = pd.read_parquet(VALIDATION_PATH)

    if "forecast_datetime" not in validation.columns:
        raise ValueError(
            "В Validation отсутствует forecast_datetime."
        )

    validation["forecast_datetime"] = pd.to_datetime(
        validation["forecast_datetime"]
    )

    validation = validation.sort_values(
        "forecast_datetime"
    ).reset_index(drop=True)

    return pipeline, validation


# ============================================================
# 4. PERIOD SPLIT
# ============================================================

def split_periods(validation):
    """
    Делит Validation на две последовательные
    равные по времени части.
    """

    unique_dates = np.sort(
        validation["forecast_datetime"].unique()
    )

    if len(unique_dates) < 2:
        raise ValueError(
            "Недостаточно временных точек для разделения."
        )

    split_index = len(unique_dates) // 2
    split_datetime = unique_dates[split_index]

    first_period = validation[
        validation["forecast_datetime"] < split_datetime
    ].copy()

    second_period = validation[
        validation["forecast_datetime"] >= split_datetime
    ].copy()

    periods = {
        "first": first_period,
        "second": second_period,
    }

    print("\nPERIOD SPLIT")
    print("=" * 80)

    for period_name, period_data in periods.items():
        print(f"\n{period_name.upper()} PERIOD")
        print(f"Rows: {len(period_data):,}")
        print(
            f"Start: {period_data['forecast_datetime'].min()}"
        )
        print(
            f"End: {period_data['forecast_datetime'].max()}"
        )

    return periods


# ============================================================
# 5. FEATURE PREPARATION
# ============================================================

def prepare_features(data, pipeline):
    preprocessor = pipeline.named_steps["preprocessor"]

    feature_names = list(preprocessor.feature_names_in_)

    missing_features = [
        feature
        for feature in feature_names
        if feature not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Отсутствуют признаки: {missing_features}"
        )

    X = data[feature_names].copy()

    for feature in CATEGORICAL_FEATURES:
        if feature in X.columns:
            X[feature] = (
                X[feature]
                .fillna("Unknown")
                .astype(str)
            )

    return X


# ============================================================
# 6. FEATURE MAPPING
# ============================================================

def get_feature_indices(
    original_feature,
    transformed_feature_names,
):
    """
    Сопоставляет преобразованные признаки
    с исходными признаками.
    """

    indices = []

    for index, transformed_name in enumerate(
        transformed_feature_names
    ):
        name = transformed_name

        if name.startswith("numeric__"):
            name = name[len("numeric__"):]

        elif name.startswith("categorical__"):
            name = name[len("categorical__"):]

        if name.startswith("onehot__"):
            name = name[len("onehot__"):]

        if original_feature in CATEGORICAL_FEATURES:
            matches = (
                name == original_feature
                or name.startswith(f"{original_feature}_")
            )
        else:
            matches = name == original_feature

        if matches:
            indices.append(index)

    return indices


# ============================================================
# 7. SHAP IMPORTANCE FOR ONE PERIOD
# ============================================================

def calculate_period_importance(
    period_name,
    period_data,
    pipeline,
    explainer,
    transformed_feature_names,
):
    logger.info(
        "Расчёт SHAP для периода: %s",
        period_name,
    )

    X = prepare_features(
        period_data,
        pipeline,
    )

    if len(X) > SAMPLE_SIZE:
        X = X.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        )

    X = X.reset_index(drop=True)

    preprocessor = pipeline.named_steps["preprocessor"]

    X_transformed = preprocessor.transform(X)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    shap_values = explainer(
        X_transformed,
        check_additivity=True,
    )

    results = []

    for original_feature in X.columns:

        indices = get_feature_indices(
            original_feature,
            transformed_feature_names,
        )

        if not indices:
            raise ValueError(
                f"Не найдены SHAP-столбцы "
                f"для признака {original_feature}"
            )

        grouped_shap = (
            shap_values.values[:, indices].sum(axis=1)
        )

        results.append({
            "feature": original_feature,
            "mean_abs_shap": np.abs(grouped_shap).mean(),
        })

    importance = pd.DataFrame(results)

    importance = importance.sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)

    importance["importance_rank"] = (
        np.arange(1, len(importance) + 1)
    )

    return importance


# ============================================================
# 8. COMPARE PERIODS
# ============================================================

def compare_periods(first_importance, second_importance):
    comparison = first_importance.merge(
        second_importance,
        on="feature",
        how="inner",
        suffixes=("_first", "_second"),
    )

    comparison["importance_change"] = (
        comparison["mean_abs_shap_second"]
        - comparison["mean_abs_shap_first"]
    )

    comparison["rank_change"] = (
        comparison["importance_rank_second"]
        - comparison["importance_rank_first"]
    )

    comparison = comparison.sort_values(
        "mean_abs_shap_first",
        ascending=False,
    ).reset_index(drop=True)

    return comparison


# ============================================================
# 9. VISUALIZATION
# ============================================================

def build_comparison_plot(comparison):
    """
    Сравнивает важность 15 наиболее значимых
    признаков первого периода.
    """

    plot_data = (
        comparison
        .sort_values(
            "mean_abs_shap_first",
            ascending=False,
        )
        .head(TOP_N)
        .sort_values(
            "mean_abs_shap_first",
            ascending=True,
        )
    )

    y = np.arange(len(plot_data))
    height = 0.35

    plt.figure(figsize=(12, 9))

    plt.barh(
        y - height / 2,
        plot_data["mean_abs_shap_first"],
        height,
        label="Первый период",
    )

    plt.barh(
        y + height / 2,
        plot_data["mean_abs_shap_second"],
        height,
        label="Второй период",
    )

    plt.yticks(
        y,
        plot_data["feature"],
    )

    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")

    plt.title(
        "SHAP Feature Importance Comparison by Period"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 10. MAIN
# ============================================================

def main():
    start_time = time.time()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pipeline, validation = load_data()

    periods = split_periods(validation)

    preprocessor = pipeline.named_steps["preprocessor"]
    xgb_model = pipeline.named_steps["model"]

    transformed_feature_names = (
        preprocessor.get_feature_names_out()
    )

    logger.info("Создание TreeExplainer.")

    explainer = shap.TreeExplainer(xgb_model)

    first_importance = calculate_period_importance(
        period_name="first",
        period_data=periods["first"],
        pipeline=pipeline,
        explainer=explainer,
        transformed_feature_names=transformed_feature_names,
    )

    second_importance = calculate_period_importance(
        period_name="second",
        period_data=periods["second"],
        pipeline=pipeline,
        explainer=explainer,
        transformed_feature_names=transformed_feature_names,
    )

    comparison = compare_periods(
        first_importance,
        second_importance,
    )

    comparison.to_csv(
        CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nSHAP IMPORTANCE COMPARISON")
    print("=" * 110)

    print(
        comparison.to_string(index=False)
    )

    build_comparison_plot(comparison)

    logger.info(
        "Время выполнения: %.2f сек.",
        time.time() - start_time,
    )

    logger.info(
        "Таблица сохранена: %s",
        CSV_PATH,
    )

    logger.info(
        "График сохранён: %s",
        PLOT_PATH,
    )


if __name__ == "__main__":
    main()
