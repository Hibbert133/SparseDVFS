import copy
import unittest
from rtx5090.partition_dryrun import partition,demo
from rtx5090.stage_freeze_plan import screen


class PartitionTests(unittest.TestCase):
    def test_algorithm_trace_tail_and_dual_rank_order(self):
        d=demo(); r=partition(d)
        self.assertFalse(r['deployable'])
        self.assertEqual(r['blocks'][0]['nodes'],['a','b'])
        self.assertEqual(r['blocks'][0]['triplet'],[2527,2527])
        self.assertTrue(r['decisions'][0]['short'])
        self.assertTrue(r['tail_short_or_equal'])
        self.assertFalse(r['eq3_all_blocks_passed'])
        self.assertEqual([n for b in r['blocks'] for n in b['rank_0_nodes']],d['nodes'])
        self.assertEqual([b['rank_0_nodes'] for b in r['blocks']],[b['rank_1_nodes'] for b in r['blocks']])

    def test_missing_data_no_fabrication(self):
        d=demo(); d['candidates']['a']['2250']=None; d['switch_s']=None
        r=partition(d)
        self.assertEqual(r['status'],'blocked_missing_inputs')
        self.assertEqual(r['blocks'],[])
        self.assertEqual(len(r['blockers']),2)

    def test_eq3_equality_and_reordered_edges(self):
        d=demo(); d['nodes']=['c','tail']; d['edges']=[('c','tail')]
        d['candidates']={n:d['candidates'][n] for n in d['nodes']}
        d['candidates']['c']['2250']['pair_latency_s']=.005
        r=partition(d)
        self.assertFalse(r['decisions'][0]['short'])
        self.assertFalse(r['blocks'][0]['amortized_strict_gt'])
        d['edges']=[('tail','c')]
        with self.assertRaises(ValueError): partition(d)

    def test_holdout_does_not_select(self):
        def w(t,e): return [dict(latency_s=t,energy_j=e)]
        train={'current':w(1,100),'2527':w(1.01,80),'2250':w(1.04,70)}
        held=copy.deepcopy(train); held['2250']=w(2,200)
        r=screen(train,held)
        self.assertEqual(r['selected_requested_setting_mhz'],2250)
        self.assertFalse(r['holdout_comparisons'][1]['latency_within_5_percent'])
        train['2250']=w(1.2,70)
        self.assertEqual(screen(train,held)['selected_requested_setting_mhz'],2527)


if __name__=='__main__': unittest.main()
