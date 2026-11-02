"""Synthetic correlation fixtures and fake-worker dispatcher invariants."""
import copy
import unittest
from types import SimpleNamespace as NS
from test_v3_worker import Model, fake_torch
from sparsedvfs_vllm.contracts import digest_json
from sparsedvfs_vllm.contracts import validate_plan
from sparsedvfs_vllm.operator_contracts import (build_certificate, correlate, validate_certificate,
                                              layout_entry, context_key)
from sparsedvfs_vllm.operators import OperatorSpans
from sparsedvfs.graph_partitioner import build_plan
from sparsedvfs.governor_reference import GovernorReference
from v5_demo import synthetic_trace, synthetic_fit_fixture


class CorrelationTests(unittest.TestCase):
    def setUp(self):
        self.trace, self.events = synthetic_trace()

    def test_three_requests_fusion_and_host_only_units(self):
        cert = build_certificate(self.trace, self.events, synthetic=True)
        validate_certificate(cert, self.trace['identity'], allow_synthetic=True)
        self.assertEqual(cert['request_count'], 3)
        self.assertEqual(cert['context_order'], ['prefill:0:2','decode:2:1'])
        case = cert['cases']['prefill:0:2']
        self.assertEqual(len(case['layout']), 3)
        self.assertEqual(len(case['units']), 2)
        self.assertTrue(case['layout'][1]['opaque_fused'])

    def test_unattributed_gpu_kernel_is_rejected(self):
        self.events.append(dict(cat='kernel', ph='X', name='outside dispatcher', ts=1,dur=2,
                                args=dict(correlation=99999,stream=7,device=0)))
        with self.assertRaisesRegex(ValueError,'coverage failed'):
            build_certificate(self.trace,self.events,synthetic=True)

    def test_auxiliary_stream_is_rejected(self):
        next(e for e in self.events if e.get('cat')=='kernel')['args']['stream']=8
        with self.assertRaisesRegex(ValueError,'streams='):
            build_certificate(self.trace,self.events,synthetic=True)

    def test_launch_on_another_thread_does_not_inherit_host_scope(self):
        next(e for e in self.events if e.get('cat')=='cuda_runtime')['tid']=101
        with self.assertRaisesRegex(ValueError,'ambiguous CUDA launch'):
            build_certificate(self.trace,self.events,synthetic=True)

    def test_missing_correlation_does_not_fall_back_to_timestamp_overlap(self):
        del next(e for e in self.events if e.get('cat')=='kernel')['args']['correlation']
        with self.assertRaisesRegex(ValueError,'unattributed GPU'):
            build_certificate(self.trace,self.events,synthetic=True)

    def test_changed_layout_across_requests_is_rejected(self):
        self.trace['operator_steps'][-1]['calls'][0]['layout']['metadata']['shape'][0]=9
        with self.assertRaisesRegex(ValueError,'layout/GPU ownership changed'):
            build_certificate(self.trace,self.events,synthetic=True)

    def test_truncated_decode_is_not_a_complete_request(self):
        self.trace['operator_steps']=self.trace['operator_steps'][:-1]
        expected={c['marker'] for s in self.trace['operator_steps'] for c in s['calls']}
        # Remove all events for the last step, so failure is request coverage,
        # rather than merely a stray activity in the profiler fixture.
        self.events=self.events[:-7]
        with self.assertRaises(ValueError):
            build_certificate(self.trace,self.events,synthetic=True)

    def test_native_output_comparison_required(self):
        self.trace['native_output_sha256']='different'
        with self.assertRaisesRegex(ValueError,'native smoke'):
            build_certificate(self.trace,self.events,synthetic=True)

    def test_synthetic_tampering_and_identity_gates(self):
        cert=build_certificate(self.trace,self.events,synthetic=True)
        with self.assertRaisesRegex(ValueError,'Synthetic'):
            validate_certificate(cert,self.trace['identity'])
        changed=copy.deepcopy(cert)
        changed['cases']['prefill:0:2']['units'].pop()
        with self.assertRaisesRegex(ValueError,'checksum'):
            validate_certificate(changed,self.trace['identity'],allow_synthetic=True)
        with self.assertRaisesRegex(ValueError,'identity'):
            validate_certificate(cert,{'backend':'other'},allow_synthetic=True)

    def test_partitioned_operator_blocks_use_existing_ticket_governor(self):
        profile,_,_,_=synthetic_fit_fixture()
        plan=build_plan(profile,100,factor=5,tolerance=0,max_slowdown=.05,allow_synthetic=True)
        governor=GovernorReference([c['triplet'] for c in profile['candidates']],profile['reference'])
        now=0
        for block in plan['blocks']:
            ticket=governor.begin_submission(block,None,2000,100,now)
            governor.end_submission(ticket,now+1)
            governor.gpu_boundary(ticket,now+2)
            governor.observe_start(ticket,now+3)
            self.assertEqual(governor.poll(ticket,now+10000000,True),'complete')
            now+=20000000
        governor.restore(now)
        self.assertTrue(governor.closed)


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.torch=fake_torch()
        self.torch.Tensor=type('NeverTensor',(),{})
        self.model=Model()
        self.schedule=NS(start_forward=lambda *a: self.log.append(('start',a)),
                         before=lambda *a:self.log.append(('before',a)),
                         after=lambda *a:self.log.append(('after',a)))
        self.log=[]
        self.trace=OperatorSpans(self.model,self.torch,lambda:self.schedule)
        self.context=dict(phase='prefill',step=0,request_id='r',num_computed_tokens=0,num_scheduled_tokens=2)
        self.calls=0
        outer=self
        class Function:
            def __str__(self):return '_C.fused.default'
            def __call__(self,*a,**k):
                outer.calls+=1
                return outer.model
        self.func=Function()

    def tearDown(self):
        self.trace.remove()

    def configure(self):
        entry=layout_entry('__worker__',0,str(self.func),{'args':[7],'kwargs':{}})
        self.trace.configure({'certificate_sha256':'low-level test fixture, not a deployable certificate',
                              'cases':{'prefill:0:2':{'layout':[entry],'units':[entry['name']]}}})

    def test_original_called_once_and_return_identity_preserved(self):
        self.configure()
        self.trace.start(self.context,'test')
        value=self.trace.dispatch(self.func,(7,),{})
        self.assertIs(value,self.model)
        self.assertEqual(self.calls,1)
        self.assertEqual([k for k,_ in self.log],['start','before','after'])
        self.assertEqual(self.log[1][1][0],'prefill:0:2')
        self.model.forward()
        self.trace.begin('sample');self.trace.end('sample');self.trace.finish()
        self.assertEqual(len(self.trace.export()['operator_steps']),1)

    def test_signature_rejected_before_original_or_frequency_boundary(self):
        self.configure();self.trace.start(self.context,'test')
        with self.assertRaisesRegex(RuntimeError,'signature'):
            self.trace.dispatch(self.func,(8,),{})
        self.assertEqual(self.calls,0)
        self.assertEqual([k for k,_ in self.log],['start'])

    def test_unknown_context_rejected_before_operator_execution(self):
        self.configure()
        context=dict(self.context,num_scheduled_tokens=3)
        with self.assertRaisesRegex(RuntimeError,'Uncertified'):
            self.trace.start(context,'test')
        self.assertEqual(self.calls,0)

    def test_missing_dispatch_is_rejected_even_with_all_modules(self):
        self.configure();self.trace.start(self.context,'test')
        self.model.forward();self.trace.begin('sample');self.trace.end('sample')
        with self.assertRaisesRegex(RuntimeError,'Incomplete dispatcher'):
            self.trace.finish()


class DeploymentContractTests(unittest.TestCase):
    def setUp(self):
        # In-memory trusted-evidence protocol fixture only. Actual shipped demo
        # outputs remain synthetic/nondeployable; this does not run any clocks.
        trace,events=synthetic_trace()
        cert=build_certificate(trace,events,synthetic=True)
        cert.update(synthetic=False,deployable=True)
        cert.pop('certificate_sha256')
        cert['certificate_sha256']=digest_json(cert)
        vectors=[[1000,1000000000,1000000000],[2000,2000000000,1000000000]]
        evidences={}
        for vector in vectors:
            evidence=copy.deepcopy(cert)
            evidence['applied_triplet']=vector
            evidence.pop('certificate_sha256')
            evidence['certificate_sha256']=digest_json(evidence)
            evidences[','.join(map(str,vector))]=evidence
        self.plan=dict(schema=5,granularity='operator',identity=trace['identity'],selector='measured',
                       coverage_certificate=cert,opp_coverage_certificates=evidences,
                       units={k:c['units'] for k,c in cert['cases'].items()},switch_us=100,
                       blocks=[dict(phase=key,block_id=i,operators=cert['cases'][key]['units'],
                                    triplet=vectors[i],estimated_ms=2,amortized_strict_gt=True)
                               for i,key in enumerate(cert['context_order'])])

    def test_used_vectors_require_matching_per_opp_diagnostic_evidence(self):
        validate_plan(self.plan)
        self.plan['opp_coverage_certificates'].pop(next(iter(self.plan['opp_coverage_certificates'])))
        with self.assertRaisesRegex(ValueError,'used OPP'):
            validate_plan(self.plan)

    def test_frequency_evidence_cannot_be_relabelled_without_validation(self):
        evidence=next(iter(self.plan['opp_coverage_certificates'].values()))
        evidence['applied_triplet']=[1,2,3]
        with self.assertRaisesRegex(ValueError,'checksum'):
            validate_plan(self.plan)

    def test_granularity_and_old_schema_cannot_be_mixed(self):
        self.plan['granularity']='module'
        with self.assertRaisesRegex(ValueError,'granularity'):
            validate_plan(self.plan)
        self.plan['granularity']='operator';self.plan['schema']=4
        with self.assertRaisesRegex(ValueError,'schema'):
            validate_plan(self.plan)
