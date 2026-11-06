import copy
import unittest
from rtx5090.independent_capacity import estimate, compare


def rows():
    return [dict(name='launch_proxy', seconds=[1.,1.,1.]),
            dict(name='gemm_train', seconds=[11.,11.,11.], work_flop=100.),
            dict(name='copy_train', seconds=[6.,6.,6.], logical_data_bytes=100.)]


class IndependentTests(unittest.TestCase):
    def test_hand_calculation(self):
        p = estimate(rows())
        self.assertEqual((p['peak_flops'],p['bandwidth_Bps'],p['overhead_s']), (10,20,1))
        self.assertFalse(p['same_path_verified'])
        r = compare(p,[dict(name='heldout',work_flop=50,logical_data_bytes=40,seconds=[6,6,6])])
        self.assertEqual(r['rows'][0]['predicted_s'],6)
        self.assertEqual(r['scores']['wape'],0)

    def test_target_labels_cannot_fit_parameters(self):
        train = rows()
        p = estimate(train)
        train.append(dict(name='MLP_label',seconds=[1e-9],work_flop=1e20))
        self.assertEqual(p, estimate(train))
        before = copy.deepcopy(p)
        compare(p,[dict(name='heldout',work_flop=50,logical_data_bytes=40,seconds=[99])])
        self.assertEqual(p,before)

    def test_inconsistent_overhead_rejected(self):
        for bad in ([.5,.5,.5], [float('nan')]*3):
            r = rows(); r[1]['seconds'] = bad
            with self.assertRaises(ValueError): estimate(r)

    def test_zero_label_rejected(self):
        with self.assertRaises((ValueError, ZeroDivisionError)):
            compare(estimate(rows()),[dict(name='bad',work_flop=1,logical_data_bytes=1,seconds=[0])])


if __name__ == '__main__': unittest.main()
