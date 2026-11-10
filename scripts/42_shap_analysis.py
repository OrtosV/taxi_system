
import json
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

EXPERIMENT_PATH = (
    PROJECT_ROOT
    / "models"
    / "registry"
    / "experiments"
    / f"{EXPERIMENT_ID}.json"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling"
    / "validation.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "shap"

GLOBAL_CSV_PATH = OUTPUT_DIR / "shap_global_importance.csv"
LOCAL_CSV_PATH = OUTPUT_DIR / "shap_local_explanation.csv"
GLOBAL_PLOT_PATH = OUTPUT_DIR / "shap_global_importance.png"

SAMPLE_SIZE = 5000
RANDOM_STATE = 42
TOP_N = 20

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

def load_experiment_metadata():
    logger.info("Загрузка метаданных эксперимента.")

    if not EXPERIMENT_PATH.exists():
        raise FileNotFoundError(
            f"Метаданные эксперимента не найдены: {EXPERIMENT_PATH}"
        )

    with open(EXPERIMENT_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def load_model():
    logger.info("Загрузка финального Pipeline.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Файл модели не найден: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def load_validation():
    logger.info("Загрузка Validation.")

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Файл Validation не найден: {VALIDATION_PATH}"
        )

    return pd.read_parquet(VALIDATION_PATH)


# ============================================================
# 4. FEATURE PREPARATION
# ============================================================

def prepare_features(validation, pipeline):
    preprocessor = pipeline.named_steps["preprocessor"]

    feature_names = list(preprocessor.feature_names_in_)

    missing_features = [
        feature
        for feature in feature_names
        if feature not in validation.columns
    ]

    if missing_features:
        raise ValueError(
            f"В Validation отсутствуют признаки: {missing_features}"
        )

    X = validation[feature_names].copy()

    for feature in CATEGORICAL_FEATURES:
        if feature in X.columns:
            X[feature] = (
                X[feature]
                .fillna("Unknown")
                .astype(str)
            )

    return X, feature_names


# ============================================================
# 5. TRANSFORMED FEATURE MAPPING
# ============================================================

def get_feature_indices(
    original_feature,
    transformed_feature_names,
):
    """
    Определяет индексы преобразованных признаков,
    соответствующих одному исходному признаку.

    Числовые признаки сопоставляются по точному имени.
    Категориальные признаки сопоставляются с One-Hot столбцами.
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
# 6. SHAP FEATURE GROUPING
# ============================================================

def group_shap_values(
    shap_values,
    transformed_feature_names,
    original_feature_names,
    X,
):
    """
    Объединяет SHAP-вклады преобразованных признаков
    обратно в исходные признаки.

    Для категориальных признаков суммируются вклады
    всех соответствующих One-Hot столбцов.
    """

    shap_array = shap_values.values

    grouped_results = []

    for original_feature in original_feature_names:

        indices = get_feature_indices(
            original_feature,
            transformed_feature_names,
        )

        if not indices:
            raise ValueError(
                f"Не удалось найти SHAP-столбцы "
                f"для признака {original_feature}"
            )

        grouped_shap = shap_array[:, indices].sum(axis=1)

        grouped_results.append({
            "feature": original_feature,
            "mean_abs_shap": np.abs(grouped_shap).mean(),
            "mean_shap": grouped_shap.mean(),
            "std_shap": grouped_shap.std(),
        })

    result = pd.DataFrame(grouped_results)

    result = result.sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)

    result.insert(
        0,
        "importance_rank",
        range(1, len(result) + 1),
    )

    return result


# ============================================================
# 7. SHAP ADDITIVITY CHECK
# ============================================================

def check_shap_additivity(
    shap_values,
    explainer,
    model,
    X_transformed,
):
    """
    Проверяет аддитивность SHAP:
    сумма SHAP-вкладов и базового значения
    должна быть близка к предсказанию модели.
    """

    print("\nSHAP ADDITIVITY CHECK")
    print("=" * 80)

    base_value = np.asarray(
        explainer.expected_value
    ).reshape(-1)

    if len(base_value) != 1:
        raise ValueError(
            "Ожидалось одно базовое значение SHAP "
            "для задачи регрессии."
        )

    predictions = model.predict(X_transformed)

    shap_sum = (
        shap_values.values.sum(axis=1)
        + base_value[0]
    )

    errors = np.abs(shap_sum - predictions)

    print(
        f"Mean absolute error: {errors.mean():.8f}"
    )

    print(
        f"Maximum absolute error: {errors.max():.8f}"
    )

    print(
        f"99th percentile error: "
        f"{np.percentile(errors, 99):.8f}"
    )


# ============================================================
# 8. CATEGORICAL FEATURE DIAGNOSTICS
# ============================================================

def diagnose_categorical_features(
    shap_values,
    transformed_feature_names,
):
    """
    Анализирует SHAP-вклады категориальных признаков
    после объединения One-Hot столбцов.
    """

    print("\nCATEGORICAL FEATURE DIAGNOSTICS")
    print("=" * 80)

    for original_feature in CATEGORICAL_FEATURES:

        indices = get_feature_indices(
            original_feature,
            transformed_feature_names,
        )

        if not indices:
            print(
                f"\n{original_feature}: "
                f"encoded columns not found"
            )
            continue

        grouped_values = (
            shap_values.values[:, indices].sum(axis=1)
        )

        print(f"\n{original_feature}")

        print(
            f"Encoded columns: {len(indices)}"
        )

        print(
            f"Mean SHAP: {grouped_values.mean():.6f}"
        )

        print(
            f"Mean absolute SHAP: "
            f"{np.abs(grouped_values).mean():.6f}"
        )

        print(
            f"Minimum SHAP: {grouped_values.min():.6f}"
        )

        print(
            f"Maximum SHAP: {grouped_values.max():.6f}"
        )

        print(
            f"Positive share: "
            f"{(grouped_values > 0).mean():.2%}"
        )

        print(
            f"Negative share: "
            f"{(grouped_values < 0).mean():.2%}"
        )


# ============================================================
# 9. LOCAL EXPLANATION
# ============================================================

def build_local_explanation(
    shap_values,
    X,
    row_index,
    transformed_feature_names,
):
    """
    Формирует локальное объяснение для одной строки.

    Все One-Hot вклады предварительно объединяются
    в соответствующие исходные признаки.
    """

    row_shap_values = shap_values.values[row_index]

    results = []

    for original_feature in X.columns:

        indices = get_feature_indices(
            original_feature,
            transformed_feature_names,
        )

        if not indices:
            raise ValueError(
                f"Не найдены SHAP-вклады для {original_feature}"
            )

        grouped_shap = row_shap_values[indices].sum()

        results.append({
            "feature": original_feature,
            "feature_value": X.iloc[row_index][original_feature],
            "shap_value": grouped_shap,
        })

    explanation = pd.DataFrame(results)

    explanation["absolute_shap"] = (
        explanation["shap_value"].abs()
    )

    explanation = explanation.sort_values(
        "absolute_shap",
        ascending=False,
    ).drop(columns=["absolute_shap"])

    return explanation


# ============================================================
# 10. GLOBAL IMPORTANCE VISUALIZATION
# ============================================================

def build_global_plot(global_importance):
    plot_data = (
        global_importance
        .head(TOP_N)
        .sort_values("mean_abs_shap", ascending=True)
    )

    plt.figure(figsize=(12, 9))

    plt.barh(
        plot_data["feature"],
        plot_data["mean_abs_shap"],
    )

    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.title("Global SHAP Feature Importance")

    plt.tight_layout()

    plt.savefig(
        GLOBAL_PLOT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 11. MAIN
# ============================================================

def main():
    start_time = time.time()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Загрузка метаданных эксперимента.
    metadata = load_experiment_metadata()

    # Загрузка финальной модели.
    pipeline = load_model()

    # Загрузка Validation.
    validation = load_validation()

    # Подготовка исходных признаков.
    X, original_feature_names = prepare_features(
        validation,
        pipeline,
    )

    # Формирование выборки.
    if len(X) > SAMPLE_SIZE:
        X = X.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        )

    X = X.reset_index(drop=True)

    logger.info(
        "Выборка для SHAP: %s строк, %s исходных признаков.",
        len(X),
        len(original_feature_names),
    )

    # Разделение Pipeline на предобработчик и модель.
    preprocessor = pipeline.named_steps["preprocessor"]
    xgb_model = pipeline.named_steps["model"]

    # Преобразование признаков.
    logger.info("Преобразование признаков.")

    X_transformed = preprocessor.transform(X)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    transformed_feature_names = (
        preprocessor.get_feature_names_out()
    )

    logger.info(
        "После преобразования получено %s признаков.",
        len(transformed_feature_names),
    )

    # Создание TreeExplainer.
    logger.info("Создание TreeExplainer.")

    explainer = shap.TreeExplainer(xgb_model)

    # Расчёт SHAP values.
    logger.info("Расчёт SHAP values.")

    shap_values = explainer(
        X_transformed,
        check_additivity=True,
    )

    logger.info(
        "SHAP values рассчитаны: %s",
        shap_values.values.shape,
    )

    # Проверка аддитивности.
    check_shap_additivity(
        shap_values=shap_values,
        explainer=explainer,
        model=xgb_model,
        X_transformed=X_transformed,
    )

    # Диагностика категориальных признаков.
    diagnose_categorical_features(
        shap_values=shap_values,
        transformed_feature_names=transformed_feature_names,
    )

    # Группировка SHAP по исходным признакам.
    logger.info("Группировка SHAP по исходным признакам.")

    global_importance = group_shap_values(
        shap_values=shap_values,
        transformed_feature_names=transformed_feature_names,
        original_feature_names=original_feature_names,
        X=X,
    )

    # Сохранение глобальной важности.
    global_importance.to_csv(
        GLOBAL_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nGLOBAL SHAP IMPORTANCE")
    print("=" * 90)

    print(
        global_importance.to_string(index=False)
    )

    # Формирование локального объяснения.
    local_explanation = build_local_explanation(
        shap_values=shap_values,
        X=X,
        row_index=0,
        transformed_feature_names=transformed_feature_names,
    )

    # Сохранение локального объяснения.
    local_explanation.to_csv(
        LOCAL_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nLOCAL SHAP EXPLANATION")
    print("=" * 90)

    print(
        local_explanation.head(TOP_N).to_string(
            index=False
        )
    )

    # Построение графика.
    build_global_plot(global_importance)

    elapsed_time = time.time() - start_time

    logger.info(
        "Время выполнения: %.2f сек.",
        elapsed_time,
    )

    logger.info(
        "Результаты сохранены:\n%s\n%s\n%s",
        GLOBAL_CSV_PATH,
        LOCAL_CSV_PATH,
        GLOBAL_PLOT_PATH,
    )


if __name__ == "__main__":
    main()
