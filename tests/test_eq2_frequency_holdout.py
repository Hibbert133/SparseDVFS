import copy
import unittest
from rtx5090.eq2_frequency_holdout import require_manifest, NAMES
from rtx5090.observe_core import digest


class HoldoutTests(unittest.TestCase):
    def test_frozen_binding_and_restoration_required(self):
        frozen=dict(identity_fingerprint='id',parameters=[1,2,3])
        valid=dict(status='complete',restoration=dict(verified=True),identity_fingerprint='id',
                   frozen_sha256=digest(frozen),windows=NAMES)
        require_manifest(valid,frozen)
        for field,value in [('status','failed'),('restoration',{'verified':False}),
                            ('identity_fingerprint','other'),('frozen_sha256','changed'),('windows',[])]:
            with self.subTest(field=field):
                bad=copy.deepcopy(valid); bad[field]=value
                with self.assertRaises(ValueError): require_manifest(bad,frozen)
        frozen['parameters'][0]=9
        with self.assertRaises(ValueError): require_manifest(valid,frozen)


if __name__=='__main__': unittest.main()
