import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stdout
from io import StringIO
from rtx5090.stage_freeze_plan import run
from rtx5090.service_core import write_new,read
from rtx5090.observe_core import digest,GPU_UUIDS
from rtx5090.eager_core import UNITS


class FreezeTests(unittest.TestCase):
    def test_full_offline_fixture_freezes_without_authorizing_hardware(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            dirs={n:root/n for n in ('latency','power','training','holdout','repair')}
            identity='fixture'
            def put(path,value): write_new(path,value); return value
            def manifest(windows,frozen=None):
                return dict(status='complete',identity_fingerprint=identity,windows=windows,
                            frozen_sha256=digest(frozen) if frozen else None,restoration=dict(verified=True))
            obs=put(dirs['latency']/'train/observation.json',dict(coverage=dict(passed=True,units=UNITS,
                contexts=['prefill'],identity_fingerprint=identity)))
            model=put(dirs['latency']/'model/model.json',dict(identity_fingerprint=identity,source_sha256=digest(obs)))
            checked=put(dirs['latency']/'check/observation.json',dict(identity_fingerprint=identity))
            diag=dirs['latency']/'diagnostic.json'
            put(diag,dict(model_sha256=digest(model),observation_sha256=digest(checked),numeric_limits_met=True))
            put(dirs['power']/'reference.json',dict(identity_fingerprint=identity,status='host_envelope_reference_complete'))
            pm=put(dirs['power']/'power-model.json',dict(identity_fingerprint=identity))
            put(dirs['power']/'power-validation.json',dict(passed=True,model_sha256=digest(pm)))
            audit=put(dirs['training']/'audit/audit.json',dict(identity_fingerprint=identity))
            frozen=put(dirs['holdout']/'frozen.json',dict(identity_fingerprint=identity,audit_sha256=digest(audit)))
            names=['requested-2527','requested-2250']
            put(dirs['training']/'collection/collection.json',manifest(names))
            m=put(dirs['holdout']/'collection/collection.json',manifest(names,frozen))
            replacement=put(dirs['repair']/'collection/collection.json',manifest(names[:1],frozen))
            put(dirs['repair']/'validation/validation.json',dict(frozen_sha256=digest(frozen),collection_sha256=digest(m),
                replacement_2527_manifest_sha256=digest(replacement),center_numeric_limits_met=True,restoration_verified=True))
            folders=[dirs['power']/(s+'-'+str(i)) for s in ('reference','holdout') for i in (1,2,3)]
            folders += [dirs['training']/'collection'/n for n in names]
            folders += [dirs['repair']/'collection'/names[0],dirs['holdout']/'collection'/names[1]]
            for index,folder in enumerate(folders):
                put(folder/'window.json',dict(identity_fingerprint=identity,status='loaded_host_sampling_passed',
                    cache_status='verified_miss',replies=[dict(latency_s=1,usage=dict(prompt_tokens=512,completion_tokens=64))]*20))
                put(folder/'summary.json',dict(status='valid_host_envelope',energy_by_uuid_j={u:10 for u in GPU_UUIDS}))
                put(folder/'requests.json',[dict(prompt_hash='%s-%s'%(index,j)) for j in range(20)])
            args=SimpleNamespace(**{k:str(v) for k,v in dirs.items()},latency_diagnostic=str(diag),out=str(root/'out'))
            with redirect_stdout(StringIO()): run(args)
            self.assertFalse(read(root/'out/stage-freeze.json')['deployable'])
            self.assertEqual(read(root/'out/module-partition-audit.json')['status'],'blocked_missing_inputs')
            self.assertEqual(len(read(root/'out/algorithm1-software-example.json')['blocks']),3)


if __name__=='__main__': unittest.main()
