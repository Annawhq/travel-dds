"""Проверки рекомендательного модуля этапа 7.

Запуск из корня репозитория:
    python -m unittest discover -s tests -p 'test_recommendations.py' -v

Данные data/test являются УЧЕБНЫМИ. Тесты проверяют алгоритмы и
согласованность данных, а не реальную стоимость путешествий.
"""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.recommendations.budget import BudgetLimits, TripCosts, calculate_budget
from src.recommendations.cities import load_city_themes, recommend_cities
from src.recommendations.city_budget import recommend_cities_with_budget
from src.recommendations.cost_catalog import load_cost_estimates
from src.recommendations.places import recommend_places
from src.recommendations.profile import THEMES, TravelProfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COST_CSV = PROJECT_ROOT / "data" / "test" / "city_costs.csv"
THEMES_CSV = PROJECT_ROOT / "data" / "test" / "city_themes.csv"
INTEREST_WEIGHTS = {
    "cost": 0.05,
    "weather": 0.05,
    "transport": 0.05,
    "attractiveness": 0.15,
    "interests": 0.70,
}


class RecommendationsTests(unittest.TestCase):
    """Проверяем весь этап 7 без запросов к интернету."""

    def test_profile_weights_are_normalized(self):
        profile = TravelProfile(weights={
            "cost": 1, "weather": 1, "transport": 1,
            "attractiveness": 1, "interests": 6,
        })
        self.assertAlmostEqual(float(profile.criterion_weights().sum()), 1.0)
        self.assertAlmostEqual(profile.criterion_weights()[-1], 0.6)
        self.assertAlmostEqual(float(TravelProfile().criterion_weights().sum()), 1.0)

    def test_profile_rejects_invalid_data(self):
        with self.assertRaises(ValueError):
            TravelProfile(days=0)
        with self.assertRaises(ValueError):
            TravelProfile(total_budget=-100)
        with self.assertRaises(ValueError):
            TravelProfile(interests=("Неизвестная тематика",))
        with self.assertRaises(ValueError):
            TravelProfile(weights={"cost": 1.0})

    def test_themes_cover_all_destinations(self):
        themes = load_city_themes(THEMES_CSV)
        ranking = recommend_cities(TravelProfile(), top_n=None)
        self.assertEqual(len(themes), 15)
        self.assertEqual(set(themes.index), set(ranking["city"]))
        self.assertEqual(set(themes.columns), set(THEMES.values()))
        self.assertTrue(themes.ge(1).all().all() and themes.le(5).all().all())

    def test_city_ranking_changes_with_interests(self):
        culture = recommend_cities(TravelProfile(
            interests=("Культура и история",), weights=INTEREST_WEIGHTS
        ), top_n=None)
        nature = recommend_cities(TravelProfile(
            interests=("Прогулки и природа",), weights=INTEREST_WEIGHTS
        ), top_n=None)
        self.assertEqual(len(culture), 15)
        self.assertEqual(len(nature), 15)
        self.assertEqual(culture.iloc[0]["city"], "Калининград")
        self.assertEqual(nature.iloc[0]["city"], "Сочи")
        self.assertNotEqual(culture.iloc[0]["city"], nature.iloc[0]["city"])
        self.assertTrue(culture["score"].between(1, 5).all())
        self.assertTrue(nature["score"].between(1, 5).all())
        self.assertTrue(culture["score"].is_monotonic_decreasing)
        self.assertTrue(culture["explanation"].str.len().gt(0).all())

    def test_risk_attitude_selects_correct_decision_rule(self):
        expected = {
            "осторожный": "Вальд",
            "нейтральный": "Байес",
            "оптимист": "Гурвиц (α=0.8)",
        }
        for risk, method in expected.items():
            with self.subTest(risk=risk):
                ranked = recommend_cities(TravelProfile(risk_attitude=risk))
                self.assertTrue(ranked["method"].eq(method).all())
                self.assertTrue(np.isfinite(ranked["risk_std"]).all())

    def test_places_change_with_user_interests(self):
        culture = recommend_places("Казань", TravelProfile(
            interests=("Культура и история",), activity_budget=3000
        ), top_n=None)
        food = recommend_places("Казань", TravelProfile(
            interests=("Гастрономия",), activity_budget=3000,
            first_visit=False,
        ), top_n=None)
        self.assertEqual(len(culture), 40)
        self.assertEqual(len(food), 40)
        self.assertNotEqual(culture.iloc[0]["object_id"], food.iloc[0]["object_id"])
        self.assertTrue(culture["score"].between(1, 5).all())
        self.assertTrue(food["score"].between(1, 5).all())
        self.assertTrue(food["score"].is_monotonic_decreasing)
        self.assertTrue(food["explanation"].str.len().gt(0).all())

    def test_places_respect_per_object_budget_and_rain(self):
        inexpensive = recommend_places("Казань", TravelProfile(
            interests=("Культура и история",), activity_budget=1000
        ), top_n=None)
        rainy = recommend_places("Казань", TravelProfile(
            interests=("Развлечения",), with_children=True,
            weather="хуже", activity_budget=5000,
        ), top_n=None)
        self.assertEqual(len(inexpensive), 36)
        self.assertTrue(inexpensive["price"].le(1000).all())
        self.assertEqual(len(rainy), 20)
        self.assertTrue(rainy["indoor"].all())
        with self.assertRaises(ValueError):
            recommend_places("Сочи", TravelProfile())

    def test_budget_total_and_limits(self):
        costs = TripCosts(
            round_trip_tickets=8000, price_per_night=3500,
            nights=1, activities_and_food=3000, local_transport=800,
        )
        within = calculate_budget(costs, BudgetLimits(
            total=17000, tickets=9000, activities_and_food=3000,
        ))
        self.assertAlmostEqual(within["reserve"], 1530.0)
        self.assertAlmostEqual(within["total"], 16830.0)
        self.assertTrue(within["fits_limits"])
        over = calculate_budget(costs, BudgetLimits(total=15000))
        self.assertFalse(over["fits_limits"])
        self.assertIn("общий бюджет", over["violations"])
        self.assertAlmostEqual(over["remaining_total"], -1830.0)

    def test_budget_rejects_negative_or_missing_prices(self):
        with self.assertRaises(ValueError):
            TripCosts(-1, 2000, 1, 1000, 500)
        with self.assertRaises((TypeError, ValueError)):
            TripCosts(None, 2000, 1, 1000, 500)
        with self.assertRaises(ValueError):
            BudgetLimits(total=-1)

    def test_all_city_pairs_exist_and_are_test_only(self):
        estimates = load_cost_estimates(COST_CSV)
        keys = {(e.origin, e.city, e.days) for e in estimates}
        all_cities = set(load_city_themes(THEMES_CSV).index)
        self.assertEqual(len(estimates), 225)
        self.assertEqual(len(keys), 225)
        self.assertEqual({e.origin for e in estimates}, all_cities)
        self.assertEqual({e.city for e in estimates}, all_cities)
        self.assertEqual({e.days for e in estimates}, {2})
        self.assertTrue(all(e.source.startswith("[ТЕСТ]") for e in estimates))
        self.assertTrue(all(e.checked_on is None for e in estimates))

    def test_origin_changes_estimated_costs(self):
        estimates = load_cost_estimates(COST_CSV)
        prices = []
        for origin in ("Москва", "Владивосток"):
            ranking = recommend_cities_with_budget(
                TravelProfile(origin=origin, days=2, total_budget=100000,
                              activity_budget=100000),
                estimates, top_n=None, only_affordable=False,
            )
            self.assertEqual(len(ranking), 15)
            self.assertFalse(ranking["estimated_total"].isna().any())
            prices.append(float(ranking.set_index("city").loc[
                "Казань", "estimated_total"
            ]))
        self.assertNotEqual(prices[0], prices[1])

    def test_missing_duration_is_not_treated_as_free(self):
        estimates = load_cost_estimates(COST_CSV)
        ranking = recommend_cities_with_budget(
            TravelProfile(origin="Москва", days=3),
            estimates, top_n=None, only_affordable=False,
        )
        self.assertTrue(ranking["estimated_total"].isna().all())
        self.assertTrue(ranking["budget_status"].eq("смета отсутствует").all())

    def test_catalog_rejects_duplicate_or_incomplete_prices(self):
        with COST_CSV.open(encoding="utf-8-sig", newline="") as handle:
            first_row = next(csv.DictReader(handle))
        columns = list(first_row)
        scenarios = [
            [first_row, first_row],
            [{**first_row, "round_trip_tickets": ""}],
            [{**first_row, "round_trip_tickets": "-100"}],
        ]
        for rows in scenarios:
            with self.subTest(rows=len(rows)):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "broken.csv"
                    with path.open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=columns)
                        writer.writeheader()
                        writer.writerows(rows)
                    with self.assertRaises(ValueError):
                        load_cost_estimates(path)


if __name__ == "__main__":
    unittest.main()
