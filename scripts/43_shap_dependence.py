
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

OUTPUT_DIR = PROJECT_ROOT / "reports" / "shap" / "dependence"

SAMPLE_SIZE = 5000
RANDOM_STATE = 42

FEATURES_TO_ANALYZE = [
    "demand_lag_168h",
    "demand_lag_1h",
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

    return pipeline, validation


# ============================================================
# 4. FEATURE PREPARATION
# ============================================================

def prepare_features(validation, pipeline):
    preprocessor = pipeline.named_steps["preprocessor"]

    feature_names = list(preprocessor.feature_names_in_)

    X = validation[feature_names].copy()

    categorical_features = [
        "zone_id",
        "Borough",
        "Zone",
        "service_zone",
    ]

    for feature in categorical_features:
        if feature in X.columns:
            X[feature] = (
                X[feature]
                .fillna("Unknown")
                .astype(str)
            )

    if len(X) > SAMPLE_SIZE:
        X = X.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        )

    return X.reset_index(drop=True)


# ============================================================
# 5. SHAP DEPENDENCE ANALYSIS
# ============================================================

def build_dependence_plot(
    feature,
    X,
    shap_values,
    transformed_feature_names,
):
    """
    Строит график зависимости SHAP-вклада
    от значения исходного признака.
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

        if name == feature:
            indices.append(index)

    if len(indices) != 1:
        raise ValueError(
            f"Для признака {feature} ожидался один "
            f"преобразованный столбец, найдено: {len(indices)}"
        )

    feature_shap = shap_values.values[:, indices[0]]

    feature_values = X[feature].to_numpy(dtype=float)

    plt.figure(figsize=(10, 6))

    plt.scatter(
        feature_values,
        feature_shap,
        alpha=0.25,
        s=12,
    )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel(
        f"Значение признака {feature}"
    )

    plt.ylabel(
        "SHAP value"
    )

    plt.title(
        f"SHAP Dependence Plot: {feature}"
    )

    plt.grid(
        alpha=0.2,
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"dependence_{feature}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "График сохранён: %s",
        output_path,
    )


# ============================================================
# 6. MAIN
# ============================================================

def main():
    start_time = time.time()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pipeline, validation = load_data()

    X = prepare_features(
        validation,
        pipeline,
    )

    logger.info(
        "Выборка для анализа: %s строк.",
        len(X),
    )

    preprocessor = pipeline.named_steps["preprocessor"]
    xgb_model = pipeline.named_steps["model"]

    logger.info("Преобразование признаков.")

    X_transformed = preprocessor.transform(X)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    transformed_feature_names = (
        preprocessor.get_feature_names_out()
    )

    logger.info("Создание TreeExplainer.")

    explainer = shap.TreeExplainer(xgb_model)

    logger.info("Расчёт SHAP values.")

    shap_values = explainer(
        X_transformed,
        check_additivity=True,
    )

    logger.info("Построение графиков зависимости.")

    for feature in FEATURES_TO_ANALYZE:
        if feature not in X.columns:
            raise ValueError(
                f"Признак отсутствует в данных: {feature}"
            )

        build_dependence_plot(
            feature=feature,
            X=X,
            shap_values=shap_values,
            transformed_feature_names=transformed_feature_names,
        )

    logger.info(
        "Время выполнения: %.2f сек.",
        time.time() - start_time,
    )


if __name__ == "__main__":
    main()
