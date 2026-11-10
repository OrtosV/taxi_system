
import json
import os
import subprocess
import sys
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_folium import st_folium


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import (
    get_available_forecast_datetimes,
    predict_for_datetime,
)


GEOJSON_PATH = (
    PROJECT_ROOT
    / "data/processed/map/taxi_zones.geojson"
)


MODEL_REGISTRY_PATH = PROJECT_ROOT / "models/registry/model_registry.json"
MONITORING_DEMO_SCRIPT = PROJECT_ROOT / "scripts/59_run_monitoring_demo.py"


@st.cache_data(show_spinner=False)
def load_model_registry():
    with MODEL_REGISTRY_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def run_monitoring_demo():
    """Запускает демонстрационную проверку отдельным Python-процессом."""
    if not MONITORING_DEMO_SCRIPT.exists():
        raise FileNotFoundError(
            f"Не найден скрипт демонстрационного мониторинга: "
            f"{MONITORING_DEMO_SCRIPT}"
        )

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    completed = subprocess.run(
        [sys.executable, str(MONITORING_DEMO_SCRIPT)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )

    if completed.returncode != 0:
        error_text = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            error_text or "Скрипт мониторинга завершился с ошибкой."
        )

    # Скрипт печатает JSON-результат; извлекаем его из stdout.
    output = completed.stdout
    json_start = output.find("{")
    if json_start < 0:
        raise RuntimeError(
            "Скрипт завершился без JSON-отчёта. "
            f"Вывод: {output[-2000:]}"
        )

    decoder = json.JSONDecoder()
    result, _ = decoder.raw_decode(output[json_start:])
    return result


EXCLUDED_ZONE_IDS = [264, 265]

WEATHER_CODES = {
    0: "Ясно",
    1: "Преимущественно ясно",
    2: "Переменная облачность",
    3: "Пасмурно",
    45: "Туман",
    48: "Туман с изморозью",
    51: "Слабая морось",
    53: "Умеренная морось",
    55: "Сильная морось",
    56: "Слабая ледяная морось",
    57: "Сильная ледяная морось",
    61: "Слабый дождь",
    63: "Умеренный дождь",
    65: "Сильный дождь",
    66: "Слабый ледяной дождь",
    67: "Сильный ледяной дождь",
    71: "Слабый снег",
    73: "Умеренный снег",
    75: "Сильный снег",
    77: "Снежные зёрна",
    80: "Слабые ливни",
    81: "Умеренные ливни",
    82: "Сильные ливни",
    85: "Слабые снегопады",
    86: "Сильные снегопады",
    95: "Гроза",
    96: "Гроза с небольшим градом",
    99: "Гроза с сильным градом",
}


st.set_page_config(
    page_title="Taxi Demand Analytics",
    page_icon="🚕",
    layout="wide",
)


# ==================================================
# ЗАГРУЗКА ДАННЫХ
# ==================================================

@st.cache_data
def load_zones():
    return gpd.read_file(GEOJSON_PATH)


@st.cache_data
def load_available_datetimes():
    return get_available_forecast_datetimes()


@st.cache_data(show_spinner=False)
def load_zone_history(zone_id: int, forecast_datetime):
    """Загружает фактический спрос выбранной зоны за 24 часа до прогноза."""
    feature_data_path = (
        PROJECT_ROOT / "data/processed/feature_dataset.parquet"
    )

    history = pd.read_parquet(
        feature_data_path,
        columns=["zone_id", "datetime", "trip_count"],
        filters=[("zone_id", "==", int(zone_id))],
    )

    history["datetime"] = pd.to_datetime(history["datetime"])
    forecast_datetime = pd.Timestamp(forecast_datetime)
    start_datetime = forecast_datetime - pd.Timedelta(hours=24)

    history = history[
        (history["datetime"] >= start_datetime)
        & (history["datetime"] < forecast_datetime)
    ].copy()

    return history.sort_values("datetime")


@st.cache_data
def load_predictions(forecast_datetime):
    return predict_for_datetime(forecast_datetime)


@st.cache_data
def load_scenario_predictions(
    forecast_datetime,
    temperature,
    humidity,
    precipitation,
    wind_speed,
    weather_code,
):
    weather_override = {
        "temperature": temperature,
        "humidity": humidity,
        "precipitation": precipitation,
        "wind_speed": wind_speed,
        "weather_code": weather_code,
    }

    return predict_for_datetime(
        forecast_datetime=forecast_datetime,
        weather_override=weather_override,
    )


# ==================================================
# ПОДГОТОВКА ПРОГНОЗОВ
# ==================================================

def filter_display_zones(predictions):
    """Исключает специальные зоны из пользовательского интерфейса."""
    return predictions[
        ~predictions["zone_id"].isin(EXCLUDED_ZONE_IDS)
    ].copy()


# ==================================================
# ИНТЕРАКТИВНАЯ КАРТА
# ==================================================

def create_demand_map(
    zones,
    predictions,
    selected_zone_id=None,
):
    """Создаёт карту прогнозируемого спроса."""

    map_data = zones.merge(
        predictions[
            ["zone_id", "predicted_demand"]
        ],
        on="zone_id",
        how="inner",
    )

    demand_map = folium.Map(
        location=[40.7128, -74.0060],
        zoom_start=10,
        tiles="CartoDB positron",
    )

    min_demand = map_data["predicted_demand"].min()
    max_demand = map_data["predicted_demand"].max()

    if min_demand == max_demand:
        max_demand += 1

    colormap = folium.LinearColormap(
        colors=[
            "#fff7bc",
            "#fec44f",
            "#d95f0e",
            "#993404",
        ],
        vmin=min_demand,
        vmax=max_demand,
        caption="Прогнозируемое количество поездок",
    )

    def style_function(feature):
        zone_id = int(feature["properties"]["zone_id"])
        demand = feature["properties"]["predicted_demand"]

        color = (
            colormap(demand)
            if demand is not None
            else "#bdbdbd"
        )

        if (
            selected_zone_id is not None
            and zone_id == selected_zone_id
        ):
            return {
                "fillColor": color,
                "color": "#0066ff",
                "weight": 4,
                "fillOpacity": 0.9,
            }

        return {
            "fillColor": color,
            "color": "#555555",
            "weight": 0.6,
            "fillOpacity": 0.75,
        }

    folium.GeoJson(
        map_data,
        name="Taxi zones",
        style_function=style_function,
        highlight_function=lambda feature: {
            "weight": 3,
            "color": "#0066ff",
            "fillOpacity": 0.9,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "zone_id",
                "zone_name",
                "borough",
                "predicted_demand",
            ],
            aliases=[
                "ID зоны:",
                "Название:",
                "Район:",
                "Прогноз поездок:",
            ],
            localize=True,
        ),
        popup=folium.GeoJsonPopup(
            fields=[
                "zone_id",
                "zone_name",
                "borough",
                "predicted_demand",
            ],
            aliases=[
                "ID зоны:",
                "Название:",
                "Район:",
                "Прогноз поездок:",
            ],
            localize=True,
        ),
    ).add_to(demand_map)

    colormap.add_to(demand_map)

    folium.LayerControl().add_to(demand_map)

    return demand_map


# ==================================================
# БЛОК НАСТРОЕК ПРОГНОЗИРОВАНИЯ
# ==================================================

def render_forecast_settings(available_datetimes):
    """Отображает единый блок настроек прогнозирования."""

    min_datetime = available_datetimes[0]
    max_datetime = available_datetimes[-1]

    with st.container(border=True):
        st.subheader("Настройки прогнозирования")

        date_col, time_col = st.columns(2)

        available_dates = sorted({
            value.date()
            for value in available_datetimes
        })

        with date_col:
            selected_date = st.date_input(
                "Дата прогнозирования",
                value=max_datetime.date(),
                min_value=min_datetime.date(),
                max_value=max_datetime.date(),
                format="DD.MM.YYYY",
            )

        times_for_date = sorted({
            value.strftime("%H:%M")
            for value in available_datetimes
            if value.date() == selected_date
        })

        with time_col:
            if times_for_date:
                selected_time = st.selectbox(
                    "Час прогнозирования",
                    options=times_for_date,
                )
            else:
                selected_time = None
                st.warning(
                    "Для выбранной даты нет доступных часов."
                )

        forecast_datetime = None

        if selected_time is not None:
            forecast_datetime = pd.Timestamp(
                f"{selected_date} {selected_time}"
            )

        st.divider()

        use_weather_scenario = st.checkbox(
            "Использовать пользовательский погодный сценарий",
            value=False,
            help=(
                "Если флажок включён, погодные признаки будут "
                "заменены пользовательскими значениями. "
                "Прогноз пересчитается для всех зон."
            ),
        )

        weather_settings = None

        if use_weather_scenario:
            st.markdown("**Погодные условия сценария**")

            c1, c2, c3 = st.columns(3)

            with c1:
                temperature = st.slider(
                    "Температура, °C",
                    min_value=-20.0,
                    max_value=40.0,
                    value=2.0,
                    step=0.5,
                )

                humidity = st.slider(
                    "Относительная влажность, %",
                    min_value=0,
                    max_value=100,
                    value=90,
                )

            with c2:
                precipitation = st.slider(
                    "Осадки, мм/ч",
                    min_value=0.0,
                    max_value=20.0,
                    value=5.0,
                    step=0.1,
                )

                wind_speed = st.slider(
                    "Скорость ветра, км/ч",
                    min_value=0.0,
                    max_value=60.0,
                    value=35.0,
                    step=0.5,
                )

            with c3:
                weather_code = st.selectbox(
                    "Погодное явление (код WMO)",
                    options=list(WEATHER_CODES.keys()),
                    index=list(WEATHER_CODES.keys()).index(65),
                    format_func=lambda code: (
                        f"{WEATHER_CODES[code]} ({code})"
                    ),
                    help=(
                        "Код WMO — международный числовой "
                        "идентификатор погодного явления. "
                        "Например, 0 — ясно, 65 — сильный дождь, "
                        "95 — гроза. Код передаётся модели как "
                        "числовой признак weather_code_prev_hour."
                    ),
                )

                st.caption(
                    "Коды соответствуют классификации WMO, "
                    "используемой Open-Meteo."
                )

            weather_settings = {
                "temperature": temperature,
                "humidity": humidity,
                "precipitation": precipitation,
                "wind_speed": wind_speed,
                "weather_code": weather_code,
            }

        st.caption(
            "После изменения даты, времени или погодных "
            "параметров прогноз и аналитика обновляются автоматически."
        )

    return (
        forecast_datetime,
        weather_settings,
    )


# ==================================================
# ОСНОВНОЕ ПРИЛОЖЕНИЕ
# ==================================================

def main():

    st.title("🚕 Taxi Demand Analytics")

    st.caption(
        "Интеллектуальная система прогнозирования "
        "пространственно-временного спроса на поездки такси"
    )

    # ----------------------------------------------
    # 1. Загрузка доступных временных срезов
    # ----------------------------------------------

    try:
        available_datetimes = load_available_datetimes()

        if not available_datetimes:
            st.error(
                "Нет доступных временных срезов для прогнозирования."
            )
            return

        zones = load_zones()

    except Exception as error:
        st.error(f"Ошибка загрузки данных: {error}")
        return

    # ----------------------------------------------
    # 2. Настройки
    # ----------------------------------------------

    (
        forecast_datetime,
        weather_settings,
    ) = render_forecast_settings(available_datetimes)

    if forecast_datetime is None:
        return

    # ----------------------------------------------
    # 3. Базовый прогноз
    # ----------------------------------------------

    try:
        with st.spinner("Формирование прогноза..."):
            baseline_all = load_predictions(
                forecast_datetime
            )

            baseline = filter_display_zones(
                baseline_all
            )

    except Exception as error:
        st.error(f"Ошибка формирования прогноза: {error}")
        return

    # ----------------------------------------------
    # 4. Сценарный прогноз
    # ----------------------------------------------

    scenario = None

    if weather_settings is not None:
        try:
            with st.spinner(
                "Пересчёт спроса с учётом погодного сценария..."
            ):
                scenario_all = load_scenario_predictions(
                    forecast_datetime,
                    weather_settings["temperature"],
                    weather_settings["humidity"],
                    weather_settings["precipitation"],
                    weather_settings["wind_speed"],
                    weather_settings["weather_code"],
                )

                scenario = filter_display_zones(
                    scenario_all
                )

        except Exception as error:
            st.error(
                f"Ошибка сценарного прогнозирования: {error}"
            )
            return

    # Активный набор прогнозов используется
    # одновременно для карты, топ-10 и карточки зоны.
    active_predictions = (
        scenario
        if scenario is not None
        else baseline
    )

    # ----------------------------------------------
    # 5. Общие показатели
    # ----------------------------------------------

    st.subheader("Общие показатели")

    if scenario is not None:
        forecast_mode = "Пользовательский сценарий"
    else:
        forecast_mode = "Базовый прогноз"

    st.caption(
        f"Режим: {forecast_mode} | "
        f"Время прогноза: "
        f"{forecast_datetime.strftime('%d.%m.%Y %H:%M')}"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Суммарный прогноз",
        f"{active_predictions['predicted_demand'].sum():,.0f}",
    )

    col2.metric(
        "Количество зон",
        f"{active_predictions['zone_id'].nunique()}",
    )

    col3.metric(
        "Максимальный спрос в зоне",
        f"{active_predictions['predicted_demand'].max():.1f}",
    )

    col4.metric(
        "Средний спрос на зону",
        f"{active_predictions['predicted_demand'].mean():.1f}",
    )

    st.divider()

    # ----------------------------------------------
    # 6. Карта и выбор зоны
    # ----------------------------------------------

    st.subheader("Пространственное распределение спроса")

    if "selected_zone_id" not in st.session_state:
        st.session_state.selected_zone_id = None

    left, right = st.columns([2, 1])

    with left:

        demand_map = create_demand_map(
            zones=zones,
            predictions=active_predictions,
            selected_zone_id=st.session_state.selected_zone_id,
        )

        map_result = st_folium(
            demand_map,
            height=650,
            use_container_width=True,
            returned_objects=["last_active_drawing"],
            key="demand_map",
        )

        # После выбора полигона сразу обновляем состояние
        # и перерисовываем карту, не ожидая следующего клика.
        if map_result:
            drawing = map_result.get("last_active_drawing")

            if drawing:
                properties = drawing.get("properties", {})

                selected_id = properties.get("zone_id")

                if selected_id is not None:
                    selected_id = int(selected_id)

                    if (
                        selected_id
                        != st.session_state.selected_zone_id
                    ):
                        st.session_state.selected_zone_id = selected_id
                        st.rerun()

    # ----------------------------------------------
    # 7. Информация о выбранной зоне
    # ----------------------------------------------

    with right:

        st.subheader("Информация о зоне")

        selected_id = st.session_state.selected_zone_id

        if selected_id is None:
            st.info("Нажмите на зону на карте.")
        else:
            selected = active_predictions[
                active_predictions["zone_id"] == selected_id
            ]

            if not selected.empty:
                zone = selected.iloc[0]

                st.markdown(f"### {zone['Zone']}")

                st.write(
                    f"**Боро:** {zone['Borough']}"
                )

                st.write(
                    f"**ID зоны:** {selected_id}"
                )

                st.metric(
                    "Прогнозируемый спрос",
                    f"{zone['predicted_demand']:.1f}",
                )

                st.metric(
                    "Спрос за предыдущий час",
                    f"{zone['previous_hour_demand']:.0f}",
                )

                if scenario is not None:
                    baseline_zone = baseline[
                        baseline["zone_id"] == selected_id
                    ]

                    if not baseline_zone.empty:
                        baseline_value = float(
                            baseline_zone["predicted_demand"].iloc[0]
                        )

                        scenario_value = float(
                            zone["predicted_demand"]
                        )

                        st.metric(
                            "Изменение относительно базы",
                            f"{scenario_value - baseline_value:+.1f}",
                            f"{(
                                (scenario_value - baseline_value)
                                / baseline_value * 100
                            ):+.2f}%" if baseline_value > 0 else None,
                        )

    # ----------------------------------------------
    # 8. Динамика спроса выбранной зоны
    # ----------------------------------------------

    st.divider()
    st.subheader("Динамика спроса в выбранной зоне")

    selected_id = st.session_state.selected_zone_id

    if selected_id is None:
        st.info("Выберите зону на карте, чтобы увидеть динамику спроса.")
    else:
        try:
            zone_history = load_zone_history(
                selected_id,
                forecast_datetime,
            )

            selected_prediction = active_predictions[
                active_predictions["zone_id"] == selected_id
            ]

            if zone_history.empty or selected_prediction.empty:
                st.info(
                    "Недостаточно исторических данных "
                    "для построения графика."
                )
            else:
                predicted_demand = float(
                    selected_prediction.iloc[0]["predicted_demand"]
                )

                fig_history = go.Figure()

                fig_history.add_trace(
                    go.Scatter(
                        x=zone_history["datetime"],
                        y=zone_history["trip_count"],
                        mode="lines+markers",
                        name="Фактический спрос",
                        line=dict(width=2),
                        marker=dict(size=5),
                    )
                )

                fig_history.add_trace(
                    go.Scatter(
                        x=[forecast_datetime],
                        y=[predicted_demand],
                        mode="markers",
                        name="Прогноз",
                        marker=dict(
                            size=13,
                            symbol="diamond",
                        ),
                    )
                )

                fig_history.update_layout(
                    title=(
                        f"Спрос: {selected_prediction.iloc[0]['Zone']} "
                        f"— предыдущие 24 часа и прогноз"
                    ),
                    xaxis_title="Время",
                    yaxis_title="Количество поездок",
                    hovermode="x unified",
                    legend_title="Показатель",
                    height=400,
                )

                st.plotly_chart(
                    fig_history,
                    use_container_width=True,
                )

                st.caption(
                    "На графике показаны фактические поездки за 24 часа "
                    "до выбранного момента и прогноз на выбранный час. "
                    "Фактическое значение прогнозируемого часа "
                    "не используется в графике."
                )

        except Exception as error:
            st.error(f"Не удалось построить график динамики спроса: {error}")

    # ----------------------------------------------
    # 9. Топ-10 зон
    # ----------------------------------------------

    st.subheader(
        "Топ-10 зон по прогнозируемому спросу"
    )

    top_zones = active_predictions.nlargest(
        10,
        "predicted_demand",
    ).sort_values(
        "predicted_demand"
    )

    fig = px.bar(
        top_zones,
        x="predicted_demand",
        y="Zone",
        orientation="h",
        color="predicted_demand",
        labels={
            "predicted_demand": "Количество поездок",
            "Zone": "Зона",
        },
        title=(
            "Топ-10 зон — "
            + forecast_mode.lower()
        ),
    )

    fig.update_layout(
        yaxis_title=None,
        xaxis_title="Прогнозируемое количество поездок",
        coloraxis_showscale=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # ----------------------------------------------
    # 10. Сравнение сценариев
    # ----------------------------------------------

    if scenario is not None:

        st.divider()

        st.subheader(
            "Сравнение базового и пользовательского прогноза"
        )

        comparison = baseline[
            [
                "zone_id",
                "Borough",
                "Zone",
                "predicted_demand",
            ]
        ].merge(
            scenario[
                [
                    "zone_id",
                    "predicted_demand",
                ]
            ],
            on="zone_id",
            suffixes=("_baseline", "_scenario"),
        )

        comparison["Изменение, поездок"] = (
            comparison["predicted_demand_scenario"]
            - comparison["predicted_demand_baseline"]
        )

        comparison["Изменение, %"] = np.where(
            comparison["predicted_demand_baseline"] > 0,
            (
                comparison["Изменение, поездок"]
                / comparison["predicted_demand_baseline"]
                * 100
            ),
            np.nan,
        )

        comparison = comparison.rename(
            columns={
                "predicted_demand_baseline": "Базовый прогноз",
                "predicted_demand_scenario": "Сценарный прогноз",
                "zone_id": "ID зоны",
                "Borough": "Боро",
                "Zone": "Название зоны",
            }
        )

        baseline_total = comparison[
            "Базовый прогноз"
        ].sum()

        scenario_total = comparison[
            "Сценарный прогноз"
        ].sum()

        total_difference = scenario_total - baseline_total

        total_percent = (
            total_difference / baseline_total * 100
            if baseline_total > 0
            else 0
        )

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Базовый суммарный спрос",
            f"{baseline_total:,.0f}",
        )

        m2.metric(
            "Сценарный суммарный спрос",
            f"{scenario_total:,.0f}",
        )

        m3.metric(
            "Общее изменение",
            f"{total_difference:+,.0f}",
            f"{total_percent:+.2f}%",
        )

        st.dataframe(
            comparison.sort_values(
                "Сценарный прогноз",
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Базовый прогноз": st.column_config.NumberColumn(
                    format="%.2f"
                ),
                "Сценарный прогноз": st.column_config.NumberColumn(
                    format="%.2f"
                ),
                "Изменение, поездок": st.column_config.NumberColumn(
                    format="%+.2f"
                ),
                "Изменение, %": st.column_config.NumberColumn(
                    format="%+.2f%%"
                ),
            },
        )

        st.caption(
            "Сценарное изменение отражает реакцию модели "
            "на замену погодных признаков. Оно не является "
            "оценкой причинного влияния погоды на спрос."
        )



    # ----------------------------------------------
    # 11. Информация о рабочей модели и мониторинг
    # ----------------------------------------------

    st.divider()

    with st.expander(
        "Информация о рабочей модели и мониторинг",
        expanded=False,
    ):
        try:
            registry = load_model_registry()
            active_version = registry["active_version"]
            active_model = registry["models"][active_version]

            st.subheader("Рабочая модель")

            info1, info2, info3 = st.columns(3)
            info1.metric("Алгоритм", active_model.get("model_name", "—"))
            info2.metric("Версия", active_version)
            info3.metric(
                "Количество признаков",
                len(active_model.get("features", [])),
            )

            st.caption(
                f"Experiment ID: {active_model.get('experiment_id', '—')}"
            )

            params = active_model.get("params", {})
            if params:
                st.markdown("**Гиперпараметры модели**")
                st.json(params, expanded=False)

            validation_metrics = active_model.get("validation_metrics", {})
            if validation_metrics:
                st.markdown("**Качество рабочей модели на Validation**")
                metric_cols = st.columns(5)
                metric_labels = [
                    ("MAE", "MAE"),
                    ("MSE", "MSE"),
                    ("RMSE", "RMSE"),
                    ("WAPE_percent", "WAPE, %"),
                    ("R2", "R²"),
                ]
                for column, (key, label) in zip(metric_cols, metric_labels):
                    value = validation_metrics.get(key)
                    column.metric(
                        label,
                        f"{float(value):.4f}" if value is not None else "—",
                    )

            st.divider()
            st.subheader("Демонстрационная проверка актуальности")

            st.caption(
                "Проверка использует исторические данные и искусственно "
                "смещённые признаки. Она демонстрирует обнаружение Data Drift, "
                "оценку качества и сценарий продвижения/отката модели. "
                "Рабочий реестр моделей не изменяется."
            )

            if st.button(
                "Проверить актуальность модели",
                type="primary",
                key="run_model_monitoring_demo",
                use_container_width=True,
            ):
                try:
                    with st.spinner(
                        "Выполняется проверка данных, качества и версий моделей..."
                    ):
                        demo_result = run_monitoring_demo()

                    st.session_state["last_monitoring_demo"] = demo_result
                    st.success("Демонстрационная проверка завершена.")
                except Exception as error:
                    st.error(f"Не удалось выполнить проверку: {error}")

            demo_result = st.session_state.get("last_monitoring_demo")

            if demo_result:
                st.markdown("### Результаты последней проверки")

                drift = demo_result["data_drift"]
                performance = demo_result["performance_drift"]
                lifecycle = demo_result["model_lifecycle"]

                result_col1, result_col2, result_col3 = st.columns(3)
                result_col1.metric(
                    "Проверено признаков",
                    drift["features_checked"],
                )
                result_col2.metric(
                    "Признаков с PSI ≥ 0.2",
                    len(drift["significant_features"]),
                )
                result_col3.metric(
                    "Максимальный PSI",
                    f"{drift['max_psi']:.3f}",
                )

                if drift["significant_features"]:
                    st.warning(
                        "Обнаружен искусственно смоделированный Data Drift: "
                        + ", ".join(drift["significant_features"])
                    )
                else:
                    st.success("Существенный Data Drift не обнаружен.")

                with st.expander("Подробнее о Data Drift"):
                    drift_table = pd.DataFrame(drift["records"])
                    st.dataframe(
                        drift_table,
                        use_container_width=True,
                        hide_index=True,
                    )

                st.markdown("**Проверка качества модели**")
                if performance["consecutive_performance_degradation"]:
                    st.error(performance["decision"])
                else:
                    st.success(performance["decision"])

                st.caption(
                    "Performance Drift оценивается на исторических "
                    "Validation-данных по месячным метрикам. "
                    "Это не проверка новых поступающих поездок."
                )

                with st.expander("Метрики по месяцам"):
                    performance_table = pd.DataFrame(
                        performance["monthly_metrics"]
                    )
                    st.dataframe(
                        performance_table,
                        use_container_width=True,
                        hide_index=True,
                    )

                st.markdown("**Демонстрация управления версиями**")
                for event in lifecycle["events"]:
                    action = event["action"]
                    if action == "candidate_evaluation":
                        st.write(
                            f"1. Кандидат {event['candidate_version']}: "
                            f"{'одобрен' if event['approved'] else 'отклонён'} "
                            f"на Validation."
                        )
                    elif action == "model_promotion":
                        st.write(
                            f"2. Продвижение: {event['from_version']} → "
                            f"{event['to_version']}."
                        )
                    elif action == "test_comparison":
                        st.write(
                            "3. Сравнение Test: "
                            f"MAE кандидата {event['candidate_MAE']:.4f} "
                            f"против {event['reference_MAE']:.4f}; "
                            f"WAPE кандидата "
                            f"{event['candidate_WAPE_percent']:.2f}% "
                            f"против {event['reference_WAPE_percent']:.2f}%."
                        )
                    elif action == "model_rollback":
                        st.write(
                            f"4. Откат: {event['from_version']} → "
                            f"{event['to_version']}."
                        )

                final_version = lifecycle["final_active_version"]
                st.success(
                    f"Итоговая рабочая версия после демонстрации: "
                    f"{final_version}"
                )
                st.caption(
                    f"ID запуска: {demo_result['run_id']} · "
                    f"Отчёт: {demo_result['output_directory']}"
                )
                st.info(
                    "Демонстрационный реестр сохранён отдельно. "
                    "Файл models/registry/model_registry.json "
                    "не изменялся."
                )

        except Exception as error:
            st.error(
                f"Не удалось загрузить информацию о рабочей модели: {error}"
            )


if __name__ == "__main__":
    main()
