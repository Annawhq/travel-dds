"""Travel DSS — единый интерфейс принятия решений (этап 8).

Запуск из корня репозитория:
    python -m streamlit run app.py

Зависимости: requirements.txt проекта + streamlit.
Все цены и тематические оценки в data/test/ являются учебными.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from src.dss.service import CITY_ROUTE_CONFIGS, DecisionResult, analyze_trip
from src.forecast.forecast import FORECAST_YEARS
from src.multicriteria.alternatives import CRITERIA_COLUMNS, load_alternatives
from src.recommendations.cities import DESTINATIONS_PATH
from src.recommendations.profile import THEMES, TravelProfile


CRITERION_LABELS = {
    "cost": "Стоимость поездки",
    "weather": "Погодные условия",
    "transport": "Транспортная доступность",
    "attractiveness": "Привлекательность",
    "interests": "Соответствие интересам",
}
# Предельное время одного интерактивного расчета маршрута.
# Тесты оптимизации без этого аргумента продолжают работать как прежде.
INTERACTIVE_ROUTE_TIMEOUT_SECONDS = 90.0

MONTHS = {
    1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
    5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
    9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь",
}


def rubles(value: float | int | None) -> str:
    """Форматирование рублей для отображения без изменения расчетов."""
    if value is None or pd.isna(value):
        return "Нет данных"
    return f"{float(value):,.0f}".replace(",", " ") + " ₽"


def clean_city_selection() -> None:
    """При изменении города убираем результаты прежнего направления."""
    st.session_state.pop("detail", None)
    st.session_state.pop("confirmed_city", None)


def build_profile(
    origin: str,
    days: int,
    total_budget: int,
    tickets_budget: int,
    activity_budget: int,
    interests: list[str],
    with_children: bool,
    first_visit: bool,
    weather: str,
    risk_attitude: str,
    custom_weights: bool,
    weights: dict[str, float],
) -> TravelProfile:
    """Один профиль используется во всех этапах системы."""
    return TravelProfile(
        origin=origin,
        days=int(days),
        total_budget=float(total_budget),
        tickets_budget=float(tickets_budget),
        activity_budget=float(activity_budget),
        interests=tuple(interests),
        with_children=with_children,
        first_visit=first_visit,
        weather=weather,
        risk_attitude=risk_attitude,
        weights=weights if custom_weights else None,
    )


def show_rankings(result: DecisionResult) -> None:
    """Сравнение всех направлений с отдельной проверкой финансовой доступности."""
    ranking = result.all_cities
    affordable = ranking.loc[ranking["budget_ok"].eq(True)]
    best_overall = str(ranking.iloc[0]["city"])
    best_affordable = str(affordable.iloc[0]["city"]) if not affordable.empty else None

    left, center, right = st.columns(3)
    left.metric("Город № 1 по предпочтениям", best_overall)
    center.metric("Доступных по смете", f"{len(affordable)} из {len(ranking)}")
    right.metric("Лучший в бюджете", best_affordable or "Не найден")

    plot = ranking.head(10).set_index("city")[["score"]].rename(
        columns={"score": "Оценка (1–5)"}
    )
    st.bar_chart(plot, height=290)

    table = ranking[[
        "city", "score", "estimated_total", "budget_status", "risk_std", "method",
    ]].copy()
    table["score"] = table["score"].round(3)
    table["risk_std"] = table["risk_std"].round(3)
    table["estimated_total"] = table["estimated_total"].map(rubles)
    st.dataframe(
        table.rename(columns={
            "city": "Город", "score": "Оценка", "estimated_total": "Смета",
            "budget_status": "Финансовая доступность", "risk_std": "Риск (СКО)",
            "method": "Критерий",
        }),
        hide_index=True, use_container_width=True,
    )
    st.caption(
        "Оценка города учитывает интересы, веса AHP и отношение к риску. "
        "Бюджет проверяется отдельно: он не изменяет баллы AHP."
    )


def show_budget(result: DecisionResult, profile: TravelProfile) -> None:
    """Общая смета со всеми составляющими и проверкой лимитов."""
    preliminary = result.city_budget
    if preliminary is None:
        st.warning("Для выбранного маршрута нет сметы поездки.")
        return

    data = [
        ("Билеты туда и обратно", preliminary["tickets"]),
        ("Проживание", preliminary["accommodation"]),
        ("Объекты и питание (предварительно)", preliminary["activities_and_food"]),
        ("Городской транспорт", preliminary["local_transport"]),
        ("Резерв", preliminary["reserve"]),
    ]
    c1, c2, c3 = st.columns(3)
    c1.metric("Предварительная стоимость", rubles(preliminary["total"]))
    c2.metric("Твой бюджет", rubles(profile.total_budget))
    c3.metric("Остаток / превышение", rubles(preliminary["remaining_total"]))

    st.dataframe(
        pd.DataFrame(data, columns=["Статья расходов", "Сумма, ₽"]).assign(
            **{"Сумма, ₽": lambda frame: frame["Сумма, ₽"].map(rubles)}
        ),
        hide_index=True, use_container_width=True,
    )
    if preliminary["fits_limits"]:
        st.success("Предварительная смета удовлетворяет установленным лимитам.")
    else:
        st.warning(
            "В предварительной смете превышены лимиты: "
            + ", ".join(preliminary["violations"])
            + ". Оптимизатор иногда подбирает более дешевый набор объектов."
        )
    st.caption(
        "Лимиты редактируются в панели параметров слева. После нажатия "
        "«Пересчитать рекомендации» обновятся и рейтинг, и сметы; "
        "старый маршрут будет сброшен."
    )


def show_weather(result: DecisionResult) -> None:
    """Сценарии и температура Казани, не погода на определенную дату."""
    w = result.weather
    if w is None:
        if result.selected_city != "Казань":
            st.info("Прогнозная модель этапа 6 подготовлена только для Казани.")
        else:
            st.info("Нет доступного прогноза для выбранного месяца.")
        return

    st.markdown(f"**Казань · {MONTHS[w.month].lower()} {w.year}**")
    c1, c2 = st.columns(2)
    c1.metric("Средняя температура месяца", f"{w.temperature_c:+.1f} °C")
    c2.metric(
        "80%-й интервал",
        f"{w.lower_80_c:+.1f}…{w.upper_80_c:+.1f} °C",
    )
    weather_table = pd.DataFrame({
        "Сценарий": ["Без дождя", "Дождь в части дней", "Дождь во все дни"],
        "Вероятность, %": [100 * w.p_better, 100 * w.p_normal, 100 * w.p_worse],
    })
    st.bar_chart(weather_table.set_index("Сценарий"), height=215)
    st.caption(
        f"Осадки: частоты {w.historical_windows} исторических окон 2011–2025 "
        f"для поездки на {w.days} дн. Это не прогноз на конкретные даты."
    )
    st.write(
        "Оценка Казани по Байесу: "
        f"{w.bayes_expert:.3f} (экспертные вероятности) → "
        f"{w.bayes_weather_data:.3f} (исторические частоты)."
    )
    st.caption(
        "Сравнение выполнено только для Казани и не изменяет общий рейтинг городов. "
        "При планировании маршрута используется выбранный пользователем погодный сценарий."
    )


def show_places(result: DecisionResult) -> None:
    if result.places.empty:
        st.info("Для выбранного города пока нет доступного рейтинга объектов.")
        return
    data = result.places[["name", "theme", "score", "price", "visit_min"]].copy()
    data["score"] = data["score"].round(2)
    data["price"] = data["price"].map(rubles)
    st.dataframe(
        data.rename(columns={
            "name": "Объект", "theme": "Тематика", "score": "Оценка",
            "price": "Цена", "visit_min": "Время, мин",
        }), hide_index=True, use_container_width=True,
    )
    with st.expander("Почему рекомендованы эти объекты"):
        for row in result.places.head(5).itertuples(index=False):
            st.markdown(f"**{row.name} — {row.score:.2f}/5**")
            st.caption(str(row.explanation))


def _route_map_data(result: DecisionResult) -> tuple[pd.DataFrame, list[dict]]:
    """Координаты объектов и схематические линии; не дорожная навигация."""
    if result.route is None or result.selected_city not in CITY_ROUTE_CONFIGS:
        return pd.DataFrame(), []
    config = CITY_ROUTE_CONFIGS[result.selected_city]
    catalog = pd.read_csv(Path(config.catalog_path))
    by_name = {
        str(row["name"]): (float(row["lat"]), float(row["lon"]))
        for _, row in catalog.iterrows()
    }
    by_id = catalog.set_index("id")
    if config.start_coordinates is not None:
        origin = config.start_coordinates
    elif config.start_object_id in by_id.index:
        point = by_id.loc[config.start_object_id]
        origin = (float(point["lat"]), float(point["lon"]))
    else:
        return pd.DataFrame(), []

    colors = [(27, 110, 171), (227, 135, 64), (69, 153, 110)]
    marker_rows: list[dict] = []
    paths: list[dict] = []
    for day, schedule in sorted(result.route.schedules.items()):
        points = [[float(origin[1]), float(origin[0])]]
        for name in schedule["Точка"].astype(str):
            if name == "Возвращение в гостиницу" or name not in by_name:
                continue
            lat, lon = by_name[name]
            points.append([lon, lat])
            marker_rows.append({
                "lat": lat, "lon": lon, "name": name, "day": int(day),
                "color": list(colors[(int(day) - 1) % len(colors)]),
            })
        points.append([float(origin[1]), float(origin[0])])
        if len(points) > 2:
            paths.append({
                "path": points,
                "color": list(colors[(int(day) - 1) % len(colors)]),
                "label": f"День {day}",
            })
    return pd.DataFrame(marker_rows), paths


def show_route_map(result: DecisionResult) -> None:
    """На карте — порядок точек по прямой, не фактические улицы/дороги."""
    markers, paths = _route_map_data(result)
    if markers.empty:
        st.info("Координаты маршрута не удалось загрузить; расписание доступно ниже.")
        return

    try:
        import pydeck as pdk

        layers = [
            pdk.Layer(
                "PathLayer", paths, get_path="path", get_color="color",
                width_min_pixels=4, pickable=True,
            ),
            pdk.Layer(
                "ScatterplotLayer", markers.to_dict("records"),
                get_position="[lon, lat]", get_fill_color="color",
                get_radius=120, radius_min_pixels=5, pickable=True,
            ),
        ]
        spread = max(
            markers["lat"].max() - markers["lat"].min(),
            markers["lon"].max() - markers["lon"].min(),
        )
        zoom = 12 if spread < 0.08 else 10 if spread < 0.35 else 8
        view = pdk.ViewState(
            latitude=float(markers["lat"].mean()),
            longitude=float(markers["lon"].mean()),
            zoom=zoom, pitch=0,
        )
        deck = pdk.Deck(
            layers=layers, initial_view_state=view,
            tooltip={"text": "{name} · день {day}"},
        )
        st.pydeck_chart(deck, use_container_width=True)
    except (ImportError, ValueError, KeyError) as exc:
        st.caption(f"Схематическая карта недоступна ({exc}); показаны точки объектов.")
        st.map(markers[["lat", "lon"]])
    st.caption(
        "Цвета разделяют дни. Линии соединяют координаты по прямой: "
        "это схема последовательности посещений, а не маршруты по дорогам. "
        "Начальная точка — условное место проживания."
    )


def _plan_text(result: DecisionResult, profile: TravelProfile) -> str:
    route = result.route
    if route is None:
        return "Маршрут еще не построен."
    lines = [
        "TRAVEL DSS — ПЛАН ПУТЕШЕСТВИЯ",
        f"Отправление: {profile.origin}",
        f"Направление: {result.selected_city}",
        f"Продолжительность: {profile.days} дн.",
        f"Интересы: {', '.join(profile.interests) or 'не указаны'}",
        f"Стоимость поездки (учебная): {rubles(route.final_budget['total'])}",
        f"Объекты и питание: {rubles(route.activities_cost)}",
        f"Приемов пищи в день: {route.meals_per_day}",
        "",
    ]
    for day, schedule in sorted(route.schedules.items()):
        lines += [f"ДЕНЬ {day}", schedule.to_string(index=False), ""]
    lines += [
        "Все суммы оценочные. Время и цены могут отличаться от фактических.",
        "Транспортные перемещения моделируются по приближенным расстояниям.",
    ]
    return "\n".join(lines)


def show_route(result: DecisionResult, profile: TravelProfile) -> None:
    route = result.route
    if route is None:
        st.warning("Маршрут не построен. Посмотри сообщения системы ниже.")
        return
    st.success(f"Маршрут на {profile.days} дн. построен для города {result.selected_city}.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Объектов в маршруте", len(route.plan))
    c2.metric("Посещения и питание", rubles(route.activities_cost))
    c3.metric("Итоговая смета", rubles(route.final_budget["total"]))
    st.caption(
        f"Персональная полезность: {route.personal_score:.2f} балла; "
        f"приемов пищи в день: {route.meals_per_day}."
    )
    if not route.final_budget["fits_limits"]:
        st.error("Внимание: итоговая смета превышает лимиты.")

    show_route_map(result)
    for day, schedule in sorted(route.schedules.items()):
        st.markdown(f"#### День {day}")
        st.dataframe(schedule, hide_index=True, use_container_width=True)

    st.download_button(
        "Скачать итоговый план (.txt)",
        data=_plan_text(result, profile).encode("utf-8-sig"),
        file_name="travel_dss_plan.txt", mime="text/plain",
    )


def main() -> None:
    st.set_page_config(page_title="Travel DSS", page_icon="🧭", layout="wide")
    st.markdown(
        """
        <style>
          .block-container {max-width: 1220px; padding-top: 2rem;}
          h1, h2, h3 {letter-spacing: -0.02em;}
          div[data-testid="stMetric"] {border: 1px solid rgba(120,130,145,.20);
              border-radius: 12px; padding: 14px 16px;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("🧭 Travel DSS")
    st.write("**Единая система выбора направления и планирования путешествия**")
    st.caption(
        "От личных предпочтений и бюджетных ограничений — к обоснованному "
        "выбору города и маршруту на 1–3 дня."
    )
    st.warning(
        "Исследовательский прототип: цены, тематические оценки и часть сведений "
        "об объектах являются учебными. Система не бронирует билеты или гостиницы."
    )

    available_cities, _ = load_alternatives(DESTINATIONS_PATH)
    today = date.today()
    year_default = today.year if today.year in FORECAST_YEARS else max(FORECAST_YEARS)
    if today.year in FORECAST_YEARS and today.month >= 11 and today.year + 1 in FORECAST_YEARS:
        year_default = today.year + 1

    with st.sidebar:
        st.header("Параметры путешествия")
        st.caption("Изменения вступают в силу после нажатия кнопки пересчёта.")
        with st.form("travel_profile"):
            origin = st.selectbox(
                "Город отправления", available_cities,
                index=available_cities.index("Москва") if "Москва" in available_cities else 0,
            )
            days = st.select_slider("Продолжительность", options=[1, 2, 3], value=2,
                                    format_func=lambda x: f"{x} дн.")
            total_budget = st.number_input(
                "Общий бюджет, ₽", min_value=0, max_value=1_000_000,
                value=30_000, step=1_000,
            )
            tickets_budget = st.number_input(
                "Максимум на билеты туда-обратно, ₽", min_value=0,
                max_value=1_000_000, value=15_000, step=1_000,
            )
            activity_budget = st.number_input(
                "Максимум на объекты и питание, ₽", min_value=0,
                max_value=1_000_000, value=3_000, step=500,
            )
            interests = st.multiselect(
                "Твои интересы", list(THEMES), default=["Культура и история"],
            )
            with_children = st.checkbox("Путешествую с детьми", value=False)
            first_visit = st.checkbox("Впервые в выбранном городе", value=True)
            weather = st.selectbox(
                "Предполагаемая погода для маршрута",
                ["обычная", "лучше", "хуже"],
                help="Этот сценарий влияет на отбор уличных объектов. "
                     "Он не подменяется автоматически месячным прогнозом.",
            )
            risk_attitude = st.selectbox(
                "Отношение к риску", ["нейтральный", "осторожный", "оптимист"],
            )
            st.markdown("**Месяц путешествия**")
            travel_year = st.selectbox(
                "Год", FORECAST_YEARS,
                index=FORECAST_YEARS.index(year_default),
                help="Прогноз температуры доступен только для Казани.",
            )
            travel_month = st.selectbox(
                "Месяц", list(MONTHS), index=6, format_func=lambda m: MONTHS[m],
            )
            with st.expander("Веса критериев AHP (дополнительно)"):
                custom = st.checkbox("Задать свои веса", value=False)
                st.caption("Если флажок выключен, применяются исходные веса AHP.")
                default_weights = [0.3584, 0.089, 0.089, 0.2318, 0.2318]
                weights = {}
                for key, value in zip(CRITERIA_COLUMNS, default_weights):
                    weights[key] = st.number_input(
                        CRITERION_LABELS[key],
                        min_value=0.0, max_value=1.0, value=float(value),
                        step=0.05, format="%.4f", key=f"ahp_{key}",
                    )
            applied = st.form_submit_button(
                "Пересчитать рекомендации", type="primary", use_container_width=True,
            )

    if applied:
        # В момент изменения профиля все предыдущие результаты должны устареть.
        st.session_state.pop("ranking", None)
        st.session_state.pop("detail", None)
        st.session_state.pop("confirmed_city", None)
        st.session_state.pop("active_profile", None)
        try:
            profile = build_profile(
                origin, days, total_budget, tickets_budget, activity_budget,
                interests, with_children, first_visit, weather, risk_attitude,
                custom, weights,
            )
            with st.spinner("Рассчитываем персональный рейтинг 15 городов..."):
                ranking = analyze_trip(profile, selected_city=None)
            st.session_state["ranking"] = ranking
            st.session_state["active_profile"] = profile
            st.session_state["active_year"] = int(travel_year)
            st.session_state["active_month"] = int(travel_month)
            available = ranking.affordable_cities
            preferred = (
                str(available.iloc[0]["city"])
                if not available.empty else str(ranking.all_cities.iloc[0]["city"])
            )
            st.session_state["destination_picker"] = preferred
        except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
            st.error(f"Не удалось рассчитать рекомендации: {exc}")

    if "active_profile" not in st.session_state:
        st.info(
            "**Начни с профиля путешествия.** Укажи параметры в панели слева "
            "и нажми «Пересчитать рекомендации»."
        )
        return

    profile = st.session_state["active_profile"]
    ranking = st.session_state["ranking"]
    month = st.session_state["active_month"]
    year = st.session_state["active_year"]
    st.caption(
        f"Примененный профиль: {profile.origin} · {profile.days} дн. · "
        f"{rubles(profile.total_budget)} · "
        f"{', '.join(profile.interests) or 'интересы не указаны'}"
    )

    st.divider()
    st.header("1. Рейтинг направлений")
    show_rankings(ranking)

    st.divider()
    st.header("2. Выбор города")
    all_ranked_cities = ranking.all_cities["city"].tolist()
    selected = st.selectbox(
        "Какой город рассматриваем для поездки?", all_ranked_cities,
        key="destination_picker", on_change=clean_city_selection,
        help="Можно выбрать не только лидера: решение остается за туристом.",
    )
    if st.button("Выбрать направление и показать детали", type="primary"):
        with st.spinner("Считаем бюджет, погоду и рекомендации объектов..."):
            try:
                detail = analyze_trip(
                    profile, selected_city=selected,
                    travel_year=year, travel_month=month, build_route=False,
                )
                st.session_state["detail"] = detail
                st.session_state["confirmed_city"] = selected
            except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
                clean_city_selection()
                st.error(f"Не удалось открыть город: {exc}")

    detail = st.session_state.get("detail")
    if detail is None or st.session_state.get("confirmed_city") != selected:
        st.info("Выбери город и нажми кнопку, чтобы увидеть смету и достопримечательности.")
        return

    selected_ranking = ranking.all_cities.set_index("city").loc[selected]
    with st.expander("Почему система оценивает это направление именно так"):
        st.write(selected_ranking["explanation"])

    st.divider()
    st.header("3. Бюджет путешествия")
    show_budget(detail, profile)

    st.divider()
    st.header("4. Погодная информация")
    show_weather(detail)

    st.divider()
    st.header("5. Достопримечательности")
    show_places(detail)

    st.divider()
    st.header("6. Оптимизация маршрута и итоговый план")
    if selected not in CITY_ROUTE_CONFIGS:
        st.info(
            "Для этого города пока не подготовлен каталог достопримечательностей. "
            "Рейтинг города и смета доступны, построение маршрута — нет."
        )
    elif not Path(CITY_ROUTE_CONFIGS[selected].catalog_path).is_file():
        st.warning("Для выбранного города не найден CSV-каталог объектов.")
    else:
        st.caption(
            "Расчёт использует весь допустимый каталог города, правила питания, "
            "интересы, ограничения бюджета и часы работы объектов. "
            "Оптимизация выполняется только после нажатия кнопки."
        )
        st.caption(
            "Интерактивный расчет ограничен 90 секундами; "
            "при сложных параметрах система может предложить повторить попытку."
        )
        if st.button("Построить персональный маршрут", type="primary"):
            # Результат со старым маршрутом не сохраняем при неудаче.
            st.session_state["detail"].route = None
            with st.spinner("Оптимизируем маршрут и проверяем расписание..."):
                try:
                    st.session_state["detail"] = analyze_trip(
                        profile, selected_city=selected, build_route=True,
                        travel_year=year, travel_month=month,
                        optimization_timeout_seconds=INTERACTIVE_ROUTE_TIMEOUT_SECONDS,
                    )
                except Exception as exc:
                    st.error(f"Оптимизатор завершился с ошибкой: {exc}")
        current_detail = st.session_state["detail"]
        if current_detail.route is not None:
            show_route(current_detail, profile)
        elif any("Маршрут не построен" in note or "Маршрут не строится" in note
                 for note in current_detail.notices):
            st.warning("Маршрут пока не получен; смотри пояснение ниже.")

    with st.expander("Примечания и ограничения системы"):
        for notice in st.session_state["detail"].notices:
            st.write("• " + notice)
        st.write(
            "• Карты показывают схематичные линии, не реальные автомобильные/пешеходные пути."
        )


if __name__ == "__main__":
    main()
