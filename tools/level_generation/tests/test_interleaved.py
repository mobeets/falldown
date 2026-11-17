"""Independent checks of schedule balance, physical paths, and policy labels."""
import importlib.util
from itertools import product
from pathlib import Path
import unittest
from collections import Counter

spec = importlib.util.spec_from_file_location(
    'interleaved', Path(__file__).resolve().parents[1] / 'generating_levels_interleaved.py')
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


class InterleavedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blocks = task.generate(300)

    def test_exact_packet_balance(self):
        for start in range(0, 300, 12):
            actual = Counter((b['design']['structure'], b['design']['condition'])
                             for b in self.blocks[start:start + 12])
            self.assertEqual(Counter({shape: sum(n for (s, _), n in actual.items() if s == shape) for shape in ('1-1-1', '1-2-1', '1-2-2-1')}), Counter({'1-1-1': 4, '1-2-1': 4, '1-2-2-1': 4}))
        self.assertEqual(len({str(b['levels']) for b in self.blocks}), 300)

    def test_physical_routes_and_policies(self):
        for block in self.blocks:
            levels, meta = block['levels'], block['design']
            routes = [(levels[0][0],)]
            # A legal directional move reaches only the nearest gap in that
            # direction. Enumerate from that rule, not a Cartesian assumption.
            for holes in levels[1:]:
                extended = []
                for route in routes:
                    x = route[-1]
                    left, right = [h for h in holes if h < x], [h for h in holes if h > x]
                    targets = ([max(left)] if left else []) + ([min(right)] if right else [])
                    if x in holes:
                        targets.append(x)
                    extended.extend(route + (h,) for h in targets)
                routes = extended
            self.assertEqual(set(routes), set(product(*levels)))
            costs = {p: sum(abs(b - a) for a, b in zip(p, p[1:])) for p in routes}
            self.assertEqual(len(routes), meta['n_paths'])
            self.assertEqual(min(costs.values()), meta['optimal_cost'])
            optimum = min(costs, key=costs.get)
            labels = []
            for k in range(1, len(levels) - 1):
                if len(levels[k]) == 2:
                    nearest = min(levels[k], key=lambda h: abs(h - optimum[k - 1]))
                    labels.append('agree' if nearest == optimum[k] else 'conflict')
            self.assertEqual('_'.join(labels) if labels else 'forced', meta['condition'])
            self.assertEqual([p['cost'] for p in meta['paths']],
                             [costs[tuple(p['holes'])] for p in meta['paths']])

    def test_horizon_predictions(self):
        directions = Counter()
        for block in self.blocks:
            lv, meta = block['levels'], block['design']
            for state in meta['horizon_states']:
                k, parent = state['level_index'], state['parent']
                for h in (1, 2, 3):
                    costs = {}
                    for route in product(*lv[k:k+h]):
                        cost = abs(route[0] - parent) + sum(abs(b-a) for a,b in zip(route, route[1:]))
                        costs[route[0]] = min(costs.get(route[0], float('inf')), cost)
                    policy = state['horizons'][str(h)]
                    self.assertEqual(set(policy['preferred_holes']), {a for a,c in costs.items() if c == min(costs.values())})
                    self.assertEqual(policy['action_excess_cost'], {str(a): c-min(costs.values()) for a,c in costs.items()})
            predictions = meta['horizon_predictions']
            for annotation in meta['route_horizon_annotations']:
                compatible = []
                for h in (1, 2, 3):
                    if all(h in choice['compatible_horizons'] for choice in annotation['choices']):
                        compatible.append(h)
                self.assertEqual(annotation['compatible_horizons'], compatible)
                self.assertEqual(compatible, [h for h in (1,2,3) if annotation['holes'] in predictions[str(h)]])
            if meta['n_paths'] == 4:
                self.assertTrue(all(len(r) == 1 for r in predictions.values()))
                self.assertEqual(len({tuple(r[0]) for r in predictions.values()}), 3)
                route = predictions['3'][0]
                directions['left' if route[1] < route[0] else 'right'] += 1
        self.assertEqual(directions, Counter(left=50, right=50))

    def test_alignment_and_spawn_balance(self):
        for n in (12, 600):
            blocks = task.generate(n)
            for start in range(0, n, 12):
                groups = {}
                for block in blocks[start:start+12]:
                    lv, m = block['levels'], block['design']
                    self.assertFalse(any(set(a) & set(b) for a,b in zip(lv,lv[1:])))
                    self.assertNotIn(m['spawn_segment'], sum(lv, []))
                    self.assertEqual(m['spawn_offset_segments'], m['spawn_segment'] - lv[0][0])
                    self.assertTrue(0 <= m['spawn_segment'] < 12)
                    self.assertEqual(m['intended_entry_direction'], 'left' if m['spawn_segment'] > lv[0][0] else 'right')
                    group = (m['structure'], m['condition'])
                    if m['n_paths'] == 4:
                        predictions = m['horizon_predictions']
                        self.assertTrue(all(len(v) == 1 for v in predictions.values()))
                        self.assertEqual(len({tuple(v[0]) for v in predictions.values()}), 3)
                        r = predictions['3'][0]
                        group = (m['structure'], 'left' if r[1] < r[0] else 'right')
                    groups.setdefault(group, []).append(1 if m['spawn_offset_segments'] > 0 else -1)
                for sides in groups.values():
                    self.assertEqual(sum(sides), 0)

    def test_reproducibility_and_invalid_counts(self):
        self.assertEqual(self.blocks, task.generate(300))
        self.assertNotEqual(self.blocks, task.generate(300, seed=42))
        for n in (0, -12, 500):
            with self.assertRaises(ValueError):
                task.generate(n)


if __name__ == '__main__':
    unittest.main()
