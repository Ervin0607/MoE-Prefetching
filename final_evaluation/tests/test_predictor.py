import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_predictor import Predictor, evaluate


def record(pair, count=3):
    return {'input_ids': [1] + [10] * (count - 1),
            'exp_ids': [[[pair[0]] * 32, [pair[1]] * 32] for _ in range(count)]}


class PredictorTests(unittest.TestCase):
    def test_known_transition_and_counter_bounds(self):
        p = Predictor([record((3, 5))])
        self.assertEqual(p.predict(1, (3, 5), None), {3, 5})
        for _ in range(8):
            p.update(1, (3, 5), {3, 5})
        self.assertEqual(set(p.table[0][3].values()), {3})
        for _ in range(8):
            p.update(1, (3, 5), {1, 7})
        self.assertEqual(set(p.table[0][3]), {1, 7})
        self.assertTrue(all(0 <= v <= 3 for v in p.table[0][3].values()))

    def test_history_resets_between_records(self):
        p = Predictor([record((3, 5))])
        result = evaluate(p, [record((3, 5), 1), record((3, 5), 1)], 'temporal', 2)
        self.assertEqual(result['mean_predictions'], 0)

    def test_score_precedes_update(self):
        p = Predictor([record((3, 5))])
        result = evaluate(p, [record((6, 7), 1)], 'spatial', 2)
        self.assertEqual(result['recall'], 0)

    def test_metric_denominators_and_chance(self):
        p = Predictor([record((3, 5))])
        result = evaluate(p, [record((3, 5))], 'popular', 2, False)
        self.assertEqual(result['recall'], 1)
        self.assertEqual(result['precision'], 1)
        self.assertAlmostEqual(result['matched_random_any_hit'], 13 / 28)
        self.assertAlmostEqual(result['matched_random_recall'], 0.25)


if __name__ == '__main__':
    unittest.main()
