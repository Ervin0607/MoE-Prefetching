"""Causal ST-MoE-inspired routing evaluation; Python standard library only.

Uses Eq. (1)'s additive history score (2), rather than Algorithm 2's
conflicting assignment. No neural training; profiling counts and online
2-bit counters only. This evaluates predictions, not latency or energy.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAYERS, EXPERTS, K = 32, 8, 2


def load_records(path):
    records = [json.loads(line) for line in path.open() if line.strip()]
    for record in records:
        if len(record['input_ids']) != len(record['exp_ids']):
            raise ValueError('Token IDs and routing lengths differ')
        for token in record['exp_ids']:
            if len(token) != K or any(len(rank) != LAYERS for rank in token):
                raise ValueError('Expected [tokens, 2, 32] routing shape')
            for layer in range(LAYERS):
                ids = [token[r][layer] for r in range(K)]
                if len(set(ids)) != K or any(type(e) is not int or not 0 <= e < EXPERTS for e in ids):
                    raise ValueError('Invalid routed expert IDs')
    return records


def routes(token):
    return [tuple(token[r][layer] for r in range(K)) for layer in range(LAYERS)]


class Predictor:
    def __init__(self, profile):
        counts = [[[0] * EXPERTS for _ in range(EXPERTS)] for _ in range(LAYERS - 1)]
        popularity = [[0] * EXPERTS for _ in range(LAYERS)]
        for record in profile:
            for token in record['exp_ids']:
                active = routes(token)
                for layer, selected in enumerate(active):
                    for expert in selected:
                        popularity[layer][expert] += 1
                    if layer:
                        for source in active[layer - 1]:
                            for target in selected:
                                counts[layer - 1][source][target] += 1
        self.popular = [sorted(range(EXPERTS), key=lambda e: (-row[e], e))[:K] for row in popularity]
        self.table = [[{e: 2 for e in sorted(range(EXPERTS), key=lambda e: (-row[e], e))[:K]}
                       for row in layer] for layer in counts]

    def predict(self, layer, current, history, mode='combined', budget=2):
        """Called before target layer routes are revealed; layer is target index."""
        if mode == 'popular':
            return set(self.popular[layer])
        scores = [0] * EXPERTS
        if mode in ('spatial', 'combined'):
            for source in current:
                for target, confidence in self.table[layer - 1][source].items():
                    scores[target] += confidence
        if mode in ('temporal', 'combined'):
            for expert in history or ():
                scores[expert] += 2
        selected = sorted((e for e in range(EXPERTS) if scores[e] >= 2),
                          key=lambda e: (-scores[e], e))
        return set(selected[:budget] if budget else selected)

    def update(self, layer, current, actual):
        for source in current:
            row = self.table[layer - 1][source]
            for expert in list(row):
                if expert in actual:
                    row[expert] = min(3, row[expert] + 1)
                elif row[expert] > 0:
                    row[expert] -= 1
                else:
                    replacements = sorted(set(actual) - set(row))
                    if replacements:
                        del row[expert]
                        row[replacements[0]] = 2


def evaluate(initial, records, mode, budget, online=True):
    totals = dict(events=0, predicted=0, correct=0, any_hit=0, full_coverage=0,
                  exact_set=0, random_any=0.0, random_correct=0.0)
    for record in records:
        predictor = copy.deepcopy(initial)  # Independent sequence evaluation.
        history = [None] * LAYERS
        for token_id, token in zip(record['input_ids'], record['exp_ids']):
            if token_id == 1:  # Packed document BOS: do not carry temporal state.
                history = [None] * LAYERS
            active = routes(token)
            for layer in range(1, LAYERS):
                predicted = predictor.predict(layer, active[layer - 1], history[layer], mode, budget)
                actual = set(active[layer])
                hits, n = len(predicted & actual), len(predicted)
                totals['events'] += 1
                totals['predicted'] += n
                totals['correct'] += hits
                totals['any_hit'] += hits > 0
                totals['full_coverage'] += actual <= predicted
                totals['exact_set'] += actual == predicted
                totals['random_correct'] += n * K / EXPERTS
                totals['random_any'] += 1 - (math.comb(EXPERTS - K, n) / math.comb(EXPERTS, n) if n <= EXPERTS - K else 0)
                if online:
                    predictor.update(layer, active[layer - 1], actual)
            history = active
    events, predicted = totals['events'], totals['predicted']
    return dict(mode=mode, budget=budget or 'threshold', online=online,
                events=events, mean_predictions=predicted / events,
                precision=totals['correct'] / predicted if predicted else 0,
                recall=totals['correct'] / (K * events),
                any_hit=totals['any_hit'] / events,
                full_coverage=totals['full_coverage'] / events,
                exact_set=totals['exact_set'] / events,
                matched_random_recall=totals['random_correct'] / (K * events),
                matched_random_any_hit=totals['random_any'] / events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, default=ROOT / 'mid_evaluation/data/hf_mixtral_c4_routing.jsonl')
    parser.add_argument('--profile-sequences', type=int, default=4)
    parser.add_argument('--output', type=Path, default=ROOT / 'final_evaluation/results')
    args = parser.parse_args()
    records = load_records(args.trace)
    n = args.profile_sequences
    if not 0 < n < len(records):
        parser.error('Require disjoint nonempty profile and evaluation sequences')
    profile, evaluation = records[:n], records[n:]
    predictor = Predictor(profile)
    settings = [('popular', 2, False), ('temporal', 2, False), ('spatial', 2, False),
                ('spatial', 2, True), ('combined', 2, True), ('combined', 4, True), ('combined', 0, True)]
    rows = [evaluate(predictor, evaluation, *setting) for setting in settings]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'prediction_metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = dict(trace=str(args.trace.resolve()), sha256=hashlib.sha256(args.trace.read_bytes()).hexdigest(),
                    profile_sequence_ids=list(range(n)), evaluation_sequence_ids=list(range(n, len(records))),
                    profile_tokens=sum(len(r['exp_ids']) for r in profile),
                    evaluation_tokens=sum(len(r['exp_ids']) for r in evaluation),
                    scope='C4 routing replay; autoregressive decode provenance unverified',
                    layers_scored='1..31; layer 0 has no preceding layer',
                    history_reset='Every record and input token ID 1 (BOS)',
                    cct_reset='Profile snapshot at each evaluation record; online updates after scoring',
                    tie_break='Lower expert ID', history_score=2,
                    algorithm='ST-MoE-inspired Eq.1 additive history; not hardware reproduction')
    (args.output / 'prediction_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for row in rows:
        print(f"{row['mode']:9} budget={str(row['budget']):9} online={str(row['online']):5} "
              f"recall={row['recall']:.3%} precision={row['precision']:.3%} predictions={row['mean_predictions']:.2f}")


if __name__ == '__main__':
    main()
