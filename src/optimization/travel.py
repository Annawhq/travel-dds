import pandas as pd
from src.optimization.objects import haversine_km

# Тестовые параметры передвижения (допущения модели)
DETOUR_FACTOR = 1.3     # реальный путь длиннее прямой линии
CITY_SPEED_KMH = 30     # средняя скорость такси / общественного транспорта
MIN_TRANSFER_MIN = 15   # минимальный переход между соседними объектами


def access_time(
    objects: pd.DataFrame,
    base_lat: float,
    base_lon: float,
    detour: float = DETOUR_FACTOR,
    speed_kmh: float = CITY_SPEED_KMH,
    min_transfer: int = MIN_TRANSFER_MIN,) -> pd.Series:
    """
    Время на дорогу к объекту (мин) в «звездной» аппроксимации:
    поездка от места проживания к объекту и обратно,
    но не меньше минимального перехода между соседними объектами
    """
    km = haversine_km(base_lat, base_lon, objects["lat"], objects["lon"])
    round_trip = 2 * km * detour / speed_kmh * 60
    return round_trip.clip(lower=min_transfer).round()
