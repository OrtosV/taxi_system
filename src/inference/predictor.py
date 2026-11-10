
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models/trained/xgboost_exp_20261110_153017.joblib"
)

FEATURE_DATA_PATH = (
    PROJECT_ROOT
    / "data/processed/feature_dataset.parquet"
)

WEATHER_FEATURES = {
    "temperature": "temperature_2m_prev_hour",
    "humidity": "relative_humidity_2m_prev_hour",
    "precipitation": "precipitation_prev_hour",
    "wind_speed": "wind_speed_10m_prev_hour",
    "weather_code": "weather_code_prev_hour",
}


def load_model():
    """Загружает обученную модель XGBoost."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Файл модели не найден: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def load_features_at_datetime(origin_datetime):
    """Загружает признаки на час, предшествующий прогнозу."""
    origin_datetime = pd.Timestamp(origin_datetime)

    df = pd.read_parquet(
        FEATURE_DATA_PATH,
        filters=[("datetime", "==", origin_datetime)],
    )

    if df.empty:
        raise ValueError(
            f"Нет исторических данных для времени {origin_datetime}"
        )

    if df["zone_id"].nunique() != 263:
        raise ValueError(
            f"Для {origin_datetime} ожидалось 263 зоны, "
            f"получено {df['zone_id'].nunique()}"
        )

    if df["zone_id"].duplicated().any():
        raise ValueError(
            f"Обнаружены повторяющиеся зоны для {origin_datetime}"
        )

    return df


def get_available_forecast_datetimes():
    """Возвращает временные метки, доступные для прогнозирования."""
    modeling_dir = PROJECT_ROOT / "data/processed/modeling"

    split_paths = [
        modeling_dir / "train.parquet",
        modeling_dir / "validation.parquet",
        modeling_dir / "test.parquet",
    ]

    available = []

    for path in split_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Не найден файл временной выборки: {path}"
            )

        split = pd.read_parquet(
            path,
            columns=["forecast_datetime"],
        )

        available.extend(
            pd.to_datetime(
                split["forecast_datetime"]
            ).dropna().unique()
        )

    return sorted(
        pd.Timestamp(value)
        for value in set(available)
    )


def add_forecast_calendar_features(
    df: pd.DataFrame,
    forecast_datetime: pd.Timestamp,
) -> pd.DataFrame:
    """Добавляет календарные признаки прогнозируемого часа."""
    result = df.copy()

    hour = forecast_datetime.hour
    day_of_week = forecast_datetime.dayofweek
    day_of_month = forecast_datetime.day
    month = forecast_datetime.month

    result["forecast_hour"] = hour
    result["forecast_day_of_week"] = day_of_week
    result["forecast_day_of_month"] = day_of_month
    result["forecast_month"] = month

    result["forecast_is_weekend"] = int(day_of_week >= 5)

    result["forecast_hour_sin"] = np.sin(
        2 * np.pi * hour / 24
    )
    result["forecast_hour_cos"] = np.cos(
        2 * np.pi * hour / 24
    )

    result["forecast_day_of_week_sin"] = np.sin(
        2 * np.pi * day_of_week / 7
    )
    result["forecast_day_of_week_cos"] = np.cos(
        2 * np.pi * day_of_week / 7
    )

    return result


def apply_weather_scenario(
    df: pd.DataFrame,
    weather_override: dict | None = None,
) -> pd.DataFrame:
    """Применяет пользовательский погодный сценарий."""
    result = df.copy()

    if weather_override is None:
        return result

    unknown_features = (
        set(weather_override) - set(WEATHER_FEATURES)
    )

    if unknown_features:
        raise ValueError(
            f"Неизвестные погодные параметры: "
            f"{sorted(unknown_features)}"
        )

    for parameter, column in WEATHER_FEATURES.items():
        if parameter in weather_override:
            result[column] = weather_override[parameter]

    result["weather_data_available_prev_hour"] = 1

    return result


def prepare_inference_data(
    latest: pd.DataFrame,
    forecast_datetime: pd.Timestamp,
    feature_names: list,
    weather_override: dict | None = None,
) -> pd.DataFrame:
    """Формирует входные данные для модели."""
    result = add_forecast_calendar_features(
        latest,
        forecast_datetime,
    )

    result = apply_weather_scenario(
        result,
        weather_override,
    )

    categorical_features = [
        "zone_id",
        "Borough",
        "Zone",
        "service_zone",
    ]

    for column in categorical_features:
        result[column] = (
            result[column]
            .fillna("Unknown")
            .astype(str)
        )

    missing_features = [
        feature
        for feature in feature_names
        if feature not in result.columns
    ]

    if missing_features:
        raise ValueError(
            f"Отсутствуют признаки: {missing_features}"
        )

    X = result[feature_names].copy()

    if X.isna().any().any():
        missing_counts = X.isna().sum()
        missing_counts = missing_counts[
            missing_counts > 0
        ]

        raise ValueError(
            "В признаках для модели обнаружены пропуски:\n"
            f"{missing_counts}"
        )

    return X


def predict_for_datetime(
    forecast_datetime,
    weather_override=None,
):
    """Прогнозирует спрос на выбранный час по всем зонам."""
    forecast_datetime = pd.Timestamp(forecast_datetime)

    origin_datetime = (
        forecast_datetime - pd.Timedelta(hours=1)
    )

    model = load_model()

    latest = load_features_at_datetime(origin_datetime)

    feature_names = list(model.feature_names_in_)

    X = prepare_inference_data(
        latest=latest,
        forecast_datetime=forecast_datetime,
        feature_names=feature_names,
        weather_override=weather_override,
    )

    predictions = model.predict(X)

    result = latest[
        [
            "zone_id",
            "Borough",
            "Zone",
            "service_zone",
            "trip_count",
        ]
    ].copy()

    result["forecast_datetime"] = forecast_datetime

    result["predicted_demand"] = np.maximum(
        predictions,
        0,
    )

    result = result.rename(
        columns={
            "trip_count": "previous_hour_demand",
        }
    )

    return result.sort_values(
        "predicted_demand",
        ascending=False,
    ).reset_index(drop=True)


def predict_next_hour(weather_override=None):
    """Прогнозирует спрос на час после последней исторической записи."""
    latest_datetime = pd.Timestamp(
        pd.read_parquet(
            FEATURE_DATA_PATH,
            columns=["datetime"],
        )["datetime"].max()
    )

    forecast_datetime = (
        latest_datetime + pd.Timedelta(hours=1)
    )

    return predict_for_datetime(
        forecast_datetime=forecast_datetime,
        weather_override=weather_override,
    )
