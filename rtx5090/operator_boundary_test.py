"""One 512/64 no-clock-write operator rendezvous test on resident TP2 workers."""
import argparse
import asyncio
import json
from pathlib import Path
import tempfile
import time
import uuid
from . import eager,service_model as service,service_client
from .eager_ipc import LocalServer,Rendezvous
from .operator_pair_contract import audit,CONTEXTS
from .observe_core import digest
from .service_core import read,write_new
from .performance import safe_error


def boundaries(data):
    report=audit(data)
    if report['mismatches'] or [c['context'] for c in report['candidates']]!=list(CONTEXTS):
        raise ValueError('Four matching candidate boundaries required')
    ops=data['windows'][0]['operators']
    for c in report['candidates']:
        for w in data['windows']:
            for rank in (0,1):
                matches=[key for key,row in w['operators'].items()
                    if key.startswith(f"{rank}:{c['context']}/") and row['parent']==c['parent']
                    and row['shape']['overload']==c['operator']]
                if len(matches)!=1:raise ValueError('Semantic boundary is not unique in source windows')
    return [dict(c,contracts=[ops[k]['shape'] for k in c['before_keys']]) for c in report['candidates']]


async def run(a,identity,out):
    source=read(a.collection)
    selected=boundaries(source)
    source_hash=digest(source); source_identity=source['identity_fingerprint']
    del source
    if any(r.get('operator_boundary_version',0)<2 for r in identity['ranks']):
        raise ValueError('Normal worker restart required: operator_boundary_version=2 absent')
    result=dict(status='failed',clock_writes=False,restoration_required=False,
                identity_fingerprint=identity['fingerprint'],source_collection_sha256=source_hash,
                source_identity_fingerprint=source_identity,
                scope='four_selected_operator_boundaries_not_full_DAG_or_DVFS_acceptance')
    write_new(out/'identity.json',identity);write_new(out/'boundaries.json',selected)
    journal=[];request_id=None
    def release(key):
        if not isinstance(key,list) or len(key)!=3 or type(key[2]) is not int:
            raise ValueError('Invalid boundary key')
        seq=key[2]
        if not 0<=seq<len(selected) or key[0]!=request_id or key[1]!=selected[seq]['context']:
            raise ValueError('Unexpected request/context/sequence')
        journal.append(dict(key=key,released_ns=time.monotonic_ns()))
        write_new(out/f'release-{seq}.json',journal[-1])
        return dict(released=True,seq=seq,clock_writes=False)
    gate=Rendezvous(release,identity['fingerprint'],timeout=10)
    def receive(message):
        if message.get('op')=='abort':
            if message.get('fingerprint')!=identity['fingerprint']:raise ValueError('Wrong identity')
            gate.abort(message.get('reason','Rank abort'));return dict(aborted=True)
        return gate.exchange(message)
    with tempfile.TemporaryDirectory(prefix='sdvfs-boundary-') as runtime:
        address=Path(runtime)/'c.sock'
        server=LocalServer(address,receive,timeout=15,on_disconnect=gate.abort)
        try:
            async with service_client.BenchClient(identity['base_url'],service.model_id(identity)) as client:
                request=client.prepare(512,a.seed,'eager-'+uuid.uuid4().hex)
                isolation=await eager.isolate_observation_cache(identity,request)
                write_new(out/'request.json',request)
                # vLLM prefixes HTTP request IDs internally; bind the exact paired ID
                # on first arrival while still requiring both ranks to agree.
                original_action=gate.action
                def first_bound_action(key):
                    nonlocal request_id
                    if request_id is None:
                        if not isinstance(key[0],str) or request['request_id'] not in key[0]:
                            raise ValueError('Unrelated request boundary')
                        request_id=key[0]
                    return original_action(key)
                gate.action=first_bound_action
                eager.rank_commands(identity,dict(op='arm',mode='operator_boundary',out=str(out),
                    request_id=request['request_id'],prompt_hash=request['prompt_hash'],
                    fingerprint=identity['fingerprint'],coordinator=str(address),boundaries=selected))
                result['reply']=await asyncio.wait_for(client.run(request),timeout=120)
                result['coverage']=eager.coverage(out,identity)
                if gate.error or len(journal)!=4:raise RuntimeError(gate.error or 'Incomplete rendezvous')
                for rank in (0,1):
                    row=read(out/f'rank-{rank}-boundary.json')
                    if row['rank']!=rank or row['status']!='complete' or row['contexts']!=list(CONTEXTS) or row['identity_sha256']!=digest(identity['ranks'][rank]):
                        raise ValueError('Worker completion evidence mismatch')
                if result['reply']['usage']['prompt_tokens']!=512 or result['reply']['usage']['completion_tokens']!=64:
                    raise ValueError('Request usage mismatch')
                result.update(status='no_write_operator_rendezvous_passed',cache_isolation=isolation,
                              runtime_boundary_verified=True)
        except BaseException as exc:
            result['error']=safe_error(exc);gate.abort(result['error']);raise
        finally:
            gate.abort('Experiment ended')
            try:eager.rank_commands(identity,dict(op='arm',mode='off'))
            except BaseException as exc:result['disarm_error']=safe_error(exc);result['status']='failed'
            server.close();result['releases']=journal;write_new(out/'boundary-test.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','collection','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--base-url',default=service_client.BASE_URL);p.add_argument('--seed',type=int,default=11100200)
    a=p.parse_args();out=service.fresh(a.out)
    try:
        import fcntl
        launch=read(Path(a.run)/'launch.json')
        with (Path(launch['runtime'])/'client.lock').open('a') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            identity=eager.inspect(a.run,a.base_url)
            result=asyncio.run(run(a,identity,out))
            print(json.dumps({k:result.get(k) for k in ('status','clock_writes','runtime_boundary_verified','disarm_error')}))
            if result['status']!='no_write_operator_rendezvous_passed':raise SystemExit(1)
    except Exception as exc:
        write_new(out/'client-error.json',dict(error=safe_error(exc),clock_writes=False))
        print(safe_error(exc));raise SystemExit(1)
    finally:print('Saved:',out)

if __name__=='__main__':main()
