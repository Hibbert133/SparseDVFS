import copy
import itertools
import unittest
from rtx5090.request_budget_partition import plan, select, fixture
from rtx5090.partition_dryrun import partition


def case():
    return dict(schema='pair-partition-input-v1', synthetic=True, scope='software_fixture',
                nodes=['a','b'], edges=[['a','b']], switch_s=.001, switch_energy_j=0,
                initial_mhz=2527, request_budget=dict(reference_s=.1,max_increase=.05,fixed_overhead_s=0),
                candidates={
                    'a': {'2250':dict(pair_latency_s=.012,pair_energy_j=1),
                          '2527':dict(pair_latency_s=.010,pair_energy_j=10)},
                    'b': {'2250':dict(pair_latency_s=.105,pair_energy_j=100),
                          '2527':dict(pair_latency_s=.090,pair_energy_j=10)}})


class BudgetTests(unittest.TestCase):
    def test_global_budget_allows_locally_slow_operator(self):
        d=case()
        self.assertEqual(partition(d)['status'],'blocked_missing_inputs')
        r=plan(d)
        self.assertEqual(r['before_partition']['choices']['a'],2250)
        self.assertTrue(r['request_budget_met'])
        self.assertAlmostEqual(r['modeled_request_s'],.104)
        self.assertFalse(r['deployable'])

    def test_matches_bruteforce_additive_objective(self):
        d=fixture(); nodes=d['nodes']; table=d['candidates']
        for deadline in (.025,.04,.06):
            rows=[]
            for path in itertools.product((2250,2527),repeat=len(nodes)):
                changes=sum(a!=b for a,b in zip((2527,)+path,path))
                t=sum(table[n][str(f)]['pair_latency_s'] for n,f in zip(nodes,path))+.001*changes
                e=sum(table[n][str(f)]['pair_energy_j'] for n,f in zip(nodes,path))+.01*changes
                if t<=deadline: rows.append((e,t,path))
            actual=select(nodes,table,deadline,.001,.01,2527)
            if rows:
                self.assertAlmostEqual(actual['energy_j'],min(rows)[0])
            else: self.assertIsNone(actual)

    def test_rechecks_after_max_frequency_merge(self):
        d=case(); d.update(initial_mhz=2250)
        d['request_budget']=dict(reference_s=.0035,max_increase=0,fixed_overhead_s=0)
        d['candidates']['a']['2250']['pair_latency_s']=.001
        d['candidates']['a']['2527']['pair_latency_s']=.004
        d['candidates']['b']['2527']['pair_latency_s']=.001
        r=plan(d)
        self.assertEqual(r['status'],'postmerge_budget_failed')
        self.assertFalse(r['request_budget_met'])

    def test_missing_inputs_and_fixed_overhead(self):
        d=case(); del d['switch_energy_j']
        self.assertEqual(plan(d)['status'],'blocked_missing_inputs')
        d=case(); d['request_budget']['fixed_overhead_s']=.2
        self.assertEqual(plan(d)['status'],'no_feasible_additive_sequence')
        d=case(); d['candidates']['a']['2250']=None
        self.assertEqual(plan(d)['status'],'blocked_missing_inputs')

    def test_retains_short_tail_no_deployment(self):
        r=plan(fixture())
        self.assertFalse(r['deployable'])
        self.assertEqual([n for b in r['partition']['blocks'] for n in b['nodes']],fixture()['nodes'])


if __name__=='__main__': unittest.main()
