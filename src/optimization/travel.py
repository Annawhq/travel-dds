import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from src.optimization.objects import haversine_km

# Тестовые параметры передвижения (допущения модели)
DETOUR_FACTOR = 1.3     # реальный путь длиннее прямой линии
CITY_SPEED_KMH = 30     # средняя скорость такси / общественного транспорта
MIN_TRANSFER_MIN = 15   # минимальный переход между соседними объектами
WALK_IN_CLUSTER_MIN = 10   # пеший переход между объектами внутри кластера
MAX_CLUSTER_DIAMETER_KM = 1.0


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


def distance_matrix_km(objects: pd.DataFrame) -> np.ndarray:
    """Матрица расстояний по прямой между всеми парами объектов (км)"""
    lat = objects["lat"].to_numpy()
    lon = objects["lon"].to_numpy()
    return haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])


def cluster_objects(
    objects: pd.DataFrame,
    max_diameter_km: float = MAX_CLUSTER_DIAMETER_KM,) -> pd.Series:
    """
    Объединяет объекты в кластеры пешей доступности
    Иерархическая кластеризация с полной связью (complete linkage):
    любые два объекта одного кластера находятся не дальше max_diameter_km
    Кластер называется по самому привлекательному объекту
    """
    distances = distance_matrix_km(objects)
    tree = linkage(squareform(distances, checks=False), method="complete")
    labels = pd.Series(
        fcluster(tree, t=max_diameter_km, criterion="distance"),
        index=objects.index,
    )
    names = {
        label: objects.loc[members, "attractiveness"].idxmax()
        for label, members in labels.groupby(labels).groups.items()
    }
    return labels.map(names)


def cluster_access_time(
    objects: pd.DataFrame,
    clusters: pd.Series,
    base_lat: float,
    base_lon: float,) -> pd.Series:
    """
    Время на дорогу от места проживания до центра кластера и обратно (мин)
    Индекс результата - названия кластеров
    """
    centers = objects.groupby(clusters)[["lat", "lon"]].mean()
    return access_time(centers, base_lat, base_lon)
