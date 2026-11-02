"""Paper-rule numerical/structural tests; all fixtures are synthetic."""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sparsedvfs.paper_core import (Frequency, OrderedDAG, greedy_partition, relative_frequency_similarity,
                                   timeline_seconds, thermal_power_w, minimum_model_energy)
from sparsedvfs.paper_calibration import predicted_candidates, coordination_vectors
from sparsedvfs.governor_reference import GovernorReference
import paper_demo


class Equations(unittest.TestCase):
    def test_eq1_compute_memory_and_effective_storage_are_separate(self):
        self.assertEqual(timeline_seconds(100, 100, 0, 0, 10, 20, 1), 11)
        self.assertEqual(timeline_seconds(100, 100, .9, 0, 10, 20, 1), 6)
        self.assertAlmostEqual(timeline_seconds(100, 100, .9, .9, 10, 20, 1), 2)

    def test_eq2_shared_thermal_coefficients_give_written_linear_term(self):
        cold = thermal_power_w(1e-9, 1.2, 1e9, .1, .5, 40)
        hot = thermal_power_w(1e-9, 1.2, 1e9, .1, .5, 60)
        self.assertAlmostEqual(cold, 6.84)
        self.assertAlmostEqual(hot-cold, 2.4)

    def test_nonfinite_negative_and_invalid_frequency_inputs(self):
        for work in (float('nan'), float('inf'), -1, True):
            with self.assertRaises(ValueError):
                timeline_seconds(work, 1, 0, 0, 1, 1, 0)
        for vector in ([0, 1, 2], [1, 2, 0], [True, 1, 2], [1.0, 2, 3]):
            with self.assertRaises(ValueError):
                Frequency.of(vector)

    def test_optimizer_requires_budget_and_rejects_infeasible_candidates(self):
        candidates = [([1, 2, 3], .1, 3), ([2, 2, 3], .08, 5)]
        self.assertEqual(minimum_model_energy(candidates, .11)['triplet'], [1, 2, 3])
        with self.assertRaises(ValueError):
            minimum_model_energy(candidates, .05)


class AlgorithmOne(unittest.TestCase):
    def test_dag_validation_accepts_diamond_with_explicit_order(self):
        graph = OrderedDAG(('a', 'b', 'c', 'd'), (('a', 'b'), ('a', 'c'), ('b', 'd'), ('c', 'd')))
        blocks, _ = greedy_partition(graph, lambda n: [1, 2, 3], lambda ns, f: len(ns)*.1,
                                      relative_frequency_similarity(0), .01, 5)
        self.assertEqual(blocks[0]['nodes'], ['a', 'b', 'c', 'd'])

    def test_duplicate_unknown_cyclic_and_invalid_topological_order_fail(self):
        bad = [OrderedDAG(('a', 'a'), ()), OrderedDAG(('a',), (('a', 'x'),)),
               OrderedDAG(('a', 'b'), (('a', 'b'), ('b', 'a'))), OrderedDAG(('b', 'a'), (('a', 'b'),))]
        for graph in bad:
            with self.assertRaises(ValueError):
                graph.validate()

    def test_short_or_similar_merge_and_componentwise_max(self):
        freq = dict(a=[1, 2, 3], b=[2, 1, 4], c=[1, 1, 1])
        blocks, trace = greedy_partition(OrderedDAG.chain(freq), freq.__getitem__, lambda ns, f: .1*len(ns),
                                         relative_frequency_similarity(0), .03, 5)
        self.assertEqual([b['nodes'] for b in blocks], [['a', 'b'], ['c']])
        self.assertEqual(blocks[0]['triplet'], [2, 2, 4])
        self.assertTrue(trace[0]['short'])
        self.assertFalse(trace[0]['similar'])
        self.assertFalse(blocks[-1]['amortized_strict_gt'])

    def test_similarity_can_merge_an_already_long_block(self):
        blocks, trace = greedy_partition(OrderedDAG.chain(['a', 'b']), lambda n: [1, 2, 3],
            lambda ns, f: .2*len(ns), relative_frequency_similarity(0), .01, 5)
        self.assertEqual(len(blocks), 1)
        self.assertFalse(trace[0]['short'])
        self.assertTrue(trace[0]['similar'])

    def test_equality_and_unconditional_tail_match_pseudocode(self):
        f = dict(a=[1, 2, 3], b=[2, 3, 4])
        blocks, _ = greedy_partition(OrderedDAG.chain(f), f.__getitem__, lambda ns, v: .5,
                                     relative_frequency_similarity(0), .1, 5)
        self.assertEqual(len(blocks), 2)
        self.assertTrue(all(not b['amortized_strict_gt'] for b in blocks))

    def test_current_block_duration_uses_merged_frequency(self):
        f = dict(a=[1, 2, 3], b=[2, 1, 4], c=[4, 4, 4])
        calls = []
        def time(nodes, frequency):
            calls.append((nodes, frequency.values))
            return .1 if len(nodes) == 1 else .7
        greedy_partition(OrderedDAG.chain(f), f.__getitem__, time, relative_frequency_similarity(0), .1, 5)
        self.assertIn((('a', 'b'), (2, 2, 4)), calls)

    def test_invalid_duration_and_empty_graph_fail(self):
        for value in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                greedy_partition(OrderedDAG.chain(['a']), lambda n: [1, 2, 3], lambda ns, f: value,
                                 relative_frequency_similarity(0), .1, 5)
        with self.assertRaises(ValueError):
            OrderedDAG.chain([])


class Calibration(unittest.TestCase):
    def setUp(self):
        data = paper_demo.fixture()
        self.profile, self.cal = data['profile'], data['calibration']

    def test_fixture_predictions_are_not_measurements(self):
        rows = predicted_candidates(self.profile, self.cal, allow_synthetic=True)
        self.assertEqual(len(rows), 8)
        self.assertEqual(len(rows[0]['operators']), 4)
        self.assertEqual(rows[0]['operators'][0]['power_attribution'], 'Eq2_scalar_prediction_not_measurement')
        with self.assertRaisesRegex(ValueError, 'Synthetic'):
            predicted_candidates(self.profile, self.cal)

    def test_peak_cannot_depend_on_emc(self):
        key = list(self.cal['opps'])[1]
        self.cal['opps'][key]['peak_flops'] *= 2
        with self.assertRaisesRegex(ValueError, 'only on CPU/GPU'):
            predicted_candidates(self.profile, self.cal, True)

    def test_bandwidth_cannot_depend_on_cpu_gpu(self):
        key = list(self.cal['opps'])[2]
        self.cal['opps'][key]['bandwidth_Bps'] *= 2
        with self.assertRaisesRegex(ValueError, 'only on EMC'):
            predicted_candidates(self.profile, self.cal, True)

    def test_per_opp_alpha_or_thermal_freedom_is_rejected(self):
        c = copy.deepcopy(self.cal)
        c['operators']['synthetic/matmul_a']['alpha_by_opp'] = {}
        with self.assertRaisesRegex(ValueError, 'not alpha_by_opp'):
            predicted_candidates(self.profile, c, True)
        self.cal['opps'][next(iter(self.cal['opps']))]['k1'] = 0
        with self.assertRaisesRegex(ValueError, 'shared device coefficients'):
            predicted_candidates(self.profile, self.cal, True)

    def test_activity_point_and_effective_sparsity_evidence_required(self):
        c = copy.deepcopy(self.cal)
        c['operators']['synthetic/vector_work']['activity_s_comp'] = 0
        with self.assertRaisesRegex(ValueError, 'Activity factor'):
            predicted_candidates(self.profile, c, True)
        self.cal['operators']['synthetic/vector_work']['effective_sparsity_verified'] = False
        with self.assertRaisesRegex(ValueError, 'verified skip'):
            predicted_candidates(self.profile, self.cal, True)

    def test_raw_zeros_do_not_change_dense_effective_sparsity(self):
        data = paper_demo.fixture(False)
        data['calibration']['operators']['synthetic/vector_work']['raw_zero_fraction'] = .9
        rows = predicted_candidates(data['profile'], data['calibration'], True)
        self.assertEqual(rows[0]['operators'][1]['s_comp'], 0)

    def test_dense_coordination_selects_minimum_sufficient_emc(self):
        indexed = {(1, 2, e): {('p', 'a'): dict(compute_ms=1, memory_ms=m)}
                   for e, m in [(3, 2), (4, .5), (5, .2)]}
        allowed, _ = coordination_vectors(indexed, ('p', 'a'), 'compute_bound')
        self.assertEqual(allowed, {(1, 2, 4)})

    def test_dense_memory_fallback_and_sparse_maximum_are_explicit(self):
        indexed = {(1, 2, e): {('p', 'a'): {}} for e in (3, 4)}
        allowed, rule = coordination_vectors(indexed, ('p', 'a'), 'effective_sparse_memory_bound')
        self.assertEqual(allowed, {(1, 2, 4)})
        self.assertTrue(coordination_vectors(indexed, ('p', 'a'), 'dense_memory_bound')[1].startswith('PROPOSED'))


class Governor(unittest.TestCase):
    def setUp(self):
        self.a = dict(block_id=0, triplet=[1, 2, 3], estimated_ms=2, amortized_strict_gt=True)
        self.b = dict(block_id=1, triplet=[2, 3, 4], estimated_ms=2, amortized_strict_gt=True)
        self.g = GovernorReference([self.a['triplet'], self.b['triplet'], [2, 2, 3]], self.a['triplet'])

    def started(self, upcoming=None):
        t = self.g.begin_submission(self.a, upcoming or self.b, 2, 100, 0)
        self.g.gpu_boundary(t, 10)
        self.g.observe_start(t, 20)
        return t

    def test_old_cpu_release_does_not_lower_new_submitter_frequency(self):
        t = self.started()
        other = self.g.begin_submission(self.b, None, 2, 100, 30)
        self.assertEqual(self.g.end_submission(t, 40), 'stale')
        self.assertEqual(self.g.cpu_requested, 2)
        self.assertEqual(self.g.cpu_owner, other)

    def test_must_finish_cpu_submission_before_lookahead(self):
        t = self.started()
        due = self.g.tickets[t].due_ns
        self.assertEqual(self.g.poll(t, due, False), 'waiting')
        self.g.end_submission(t, 30)
        self.assertEqual(self.g.poll(t, due, False), 'request_issued_end_pending')
        self.assertEqual(self.g.cpu_requested, 1)  # GPU/EMC look-ahead does not touch CPU.

    def test_old_gpu_request_is_stale_after_a_new_boundary(self):
        t = self.started()
        self.g.end_submission(t, 30)
        other = self.g.begin_submission(self.b, None, 2, 100, 40)
        self.g.gpu_boundary(other, 50)
        self.assertEqual(self.g.poll(t, self.g.tickets[t].due_ns, False), 'stale')
        self.assertEqual(self.g.resources_requested, (3, 4))

    def test_cpu_only_vector_change_is_not_a_resource_write(self):
        upcoming = dict(self.b, triplet=[2, 2, 3])
        t = self.started(upcoming)
        self.g.end_submission(t, 30)
        self.assertEqual(self.g.poll(t, self.g.tickets[t].due_ns, False), 'no_change')
        self.assertEqual(self.g.events[-1]['kind'], 'lookahead_noop_same_target')

    def test_end_ready_is_missed_not_issued(self):
        t = self.started()
        self.g.end_submission(t, 30)
        self.assertEqual(self.g.poll(t, self.g.tickets[t].due_ns, True), 'complete')
        self.assertEqual(self.g.resources_requested, (2, 3))
        self.assertIn('lookahead_missed_end_ready', [r['kind'] for r in self.g.events])

    def test_post_write_completion_is_ambiguous(self):
        t = self.started()
        self.g.end_submission(t, 30)
        self.assertEqual(self.g.poll(t, self.g.tickets[t].due_ns, False, after_write_end_ready=True), 'ambiguous_after_write')

    def test_restore_disables_late_requests_but_is_not_hardware_evidence(self):
        t = self.started()
        self.g.end_submission(t, 30)
        self.assertFalse(self.g.restore(40)['actual_hardware_restored'])
        self.assertEqual(self.g.poll(t, self.g.tickets[t].due_ns, False), 'closed')

    def test_short_tail_never_enables_lookahead(self):
        short = dict(self.a, estimated_ms=.02, amortized_strict_gt=False)
        t = self.g.begin_submission(short, self.b, 2, 100, 0)
        self.g.gpu_boundary(t, 10)
        self.g.observe_start(t, 20)
        self.g.end_submission(t, 30)
        self.assertEqual(self.g.poll(t, 40, False), 'waiting')


class SyntheticEntry(unittest.TestCase):
    def test_both_scenarios_are_executed_and_non_deployable(self):
        for sparse in (False, True):
            output = paper_demo.run(paper_demo.fixture(sparse))
            self.assertTrue(output['synthetic'])
            self.assertFalse(output['deployable'])
            self.assertEqual(len(output['blocks']), 4)
            self.assertFalse(output['blocks'][-1]['amortized_strict_gt'])
            self.assertEqual(output['evidence']['hardware'], 'NOT EXECUTED')


if __name__ == '__main__':
    unittest.main()
