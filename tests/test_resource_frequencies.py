import copy
import unittest
from rtx5090.resource_frequencies import resolve_rates,holdout_rates,merge_training


def source(rates):
    rows=[]
    for rate in rates:
        for rank in (0,1):
            for shape in ([1,1024,4096],[512,1024,4096]):
                for repeat in range(3):
                    rows.append(dict(id=f'{rate}/{rank}/{shape}/{repeat}',rank=rank,family='linear',shape=shape,
                        split='train',identity_fingerprint='same',requested_mhz=rate,
                        data_semantics='logical_tensor_bytes',memory_mhz=13801))
    return dict(status='resource_collection_complete',collection_role='train',rows=rows,
        identity_fingerprint='same',requested_window_s=30,restoration=dict(verified=True))


class FrequencyTests(unittest.TestCase):
    def test_common_nearest_is_bounded_and_unique(self):
        rates,mapping=resolve_rates([1800,2025,2250,2527],[1800,2025,2250,2520])
        self.assertEqual(rates,[1800,2025,2250,2520]);self.assertEqual(mapping['2527'],2520)
        for requested,common in (([1800],[1600]),([1800,1801],[1800]),([0],[0])):
            with self.assertRaises(ValueError):resolve_rates(requested,common)

    def test_holdout_does_not_overlap_or_extrapolate(self):
        holdout_rates([1950,2385],[1800,2025,2250,2527])
        for rates in ([1800],[1700],[2600]):
            with self.assertRaises(ValueError):holdout_rates(rates,[1800,2025,2250,2527])

    def test_merge_preserves_sources_and_all_rows(self):
        high,low=source([2250,2527]),source([1800,2025]);before=copy.deepcopy(high)
        result=merge_training([high,low]);self.assertEqual(high,before)
        self.assertEqual(result['training_requested_mhz'],[1800,2025,2250,2527])
        self.assertEqual(len(result['rows']),48)

    def test_refuse_partial_holdout_overlap_changed_protocol(self):
        for mutation in (lambda c:c.update(status='failed'),lambda c:c.update(collection_role='holdout'),
                         lambda c:c.update(identity_fingerprint='other'),lambda c:c.update(requested_window_s=60),
                         lambda c:c['restoration'].update(verified=False),lambda c:c['rows'].pop()):
            low=source([1800,2025]);mutation(low)
            with self.assertRaises(ValueError):merge_training([source([2250,2527]),low])
        with self.assertRaises(ValueError):merge_training([source([2250,2527]),source([2250,2527])])

if __name__=='__main__':unittest.main()
