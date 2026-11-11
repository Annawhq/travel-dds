"""Интеграция этапа 6 с Travel DSS (без запуска оптимизатора PuLP/CBC).

Из корня проекта:
    python -m unittest discover -s tests -p "test_weather_integration.py" -v
"""
import unittest
from unittest.mock import patch

import pandas as pd

from src.dss.service import analyze_trip
from src.recommendations.profile import TravelProfile


class WeatherIntegrationTests(unittest.TestCase):
    def profile(self, days=2):
        return TravelProfile(
            origin="Москва", days=days, total_budget=100_000,
            tickets_budget=80_000, activity_budget=3_000,
            interests=("Культура и история",), risk_attitude="нейтральный",
        )

    def analyze(self, city="Казань", days=2, **kwargs):
        return analyze_trip(
            self.profile(days), selected_city=city, build_route=False,
            **kwargs,
        )

    def test_2027_july_forecast_and_probabilities(self):
        result = self.analyze(travel_year=2027, travel_month=7)
        self.assertIsNotNone(result.weather)
        w = result.weather
        self.assertEqual((w.city, w.year, w.month, w.days), ("Казань", 2027, 7, 2))
        self.assertAlmostEqual(w.temperature_c, 21.62, delta=0.2)
        self.assertLess(w.lower_80_c, w.temperature_c)
        self.assertLess(w.temperature_c, w.upper_80_c)
        self.assertAlmostEqual(w.p_better, 0.569, delta=0.005)
        self.assertAlmostEqual(w.p_normal, 0.293, delta=0.005)
        self.assertAlmostEqual(w.p_worse, 0.138, delta=0.005)
        self.assertAlmostEqual(w.p_better + w.p_normal + w.p_worse, 1.0, places=10)
        self.assertGreater(w.historical_windows, 0)
        self.assertNotAlmostEqual(w.bayes_expert, w.bayes_weather_data, places=5)
        # При нейтральном ЛПР показатель из старого рейтинга = исходный Байес.
        kazan = result.all_cities.set_index("city").loc["Казань", "score"]
        self.assertAlmostEqual(float(kazan), w.bayes_expert, places=10)
        self.assertIsNone(result.route)

    def test_trip_days_change_rain_probabilities(self):
        w1 = self.analyze(days=1, travel_year=2027, travel_month=7).weather
        w2 = self.analyze(days=2, travel_year=2027, travel_month=7).weather
        w3 = self.analyze(days=3, travel_year=2027, travel_month=7).weather
        self.assertEqual((w1.days, w2.days, w3.days), (1, 2, 3))
        self.assertEqual(w1.p_normal, 0.0)  # часть дней = невозможно для одного дня
        self.assertGreater(w1.p_better, w2.p_better)
        self.assertGreater(w2.p_better, w3.p_better)
        self.assertAlmostEqual(w1.temperature_c, w2.temperature_c, places=9)

    def test_forecast_does_not_change_15_city_ranking(self):
        without = self.analyze()
        with_weather = self.analyze(travel_year=2027, travel_month=7)
        pd.testing.assert_frame_equal(without.all_cities, with_weather.all_cities)
        self.assertIsNone(without.weather)
        self.assertIsNotNone(with_weather.weather)

    def test_other_cities_get_no_kazan_forecast(self):
        result = self.analyze("Нижний Новгород", travel_year=2027, travel_month=7)
        self.assertIsNone(result.weather)
        self.assertTrue(any("нет модели этапа 6" in note for note in result.notices))
        self.assertIsNotNone(result.city_budget)

    def test_missing_weather_file_does_not_break_city_selection(self):
        with patch("src.dss.service.load_weather", side_effect=FileNotFoundError("нет CSV")):
            result = self.analyze(travel_year=2027, travel_month=7)
        self.assertIsNone(result.weather)
        self.assertIsNotNone(result.city_budget)
        self.assertTrue(any("Погодная модель Казани недоступна" in n for n in result.notices))

    def test_invalid_date_rejected(self):
        invalid = [
            {"travel_year": 2027},
            {"travel_month": 7},
            {"travel_year": 2028, "travel_month": 7},
            {"travel_year": 2027, "travel_month": 0},
            {"travel_year": 2027, "travel_month": 13},
            {"travel_year": True, "travel_month": 7},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    self.analyze(**kwargs)


if __name__ == "__main__":
    unittest.main()
