// Independent reference adapter for paper sections 4.3.1/4.3.3 and 5.
// No CUDA API calls occur in the boundary host function. Sysfs writes are
// serialized by the core. CUDA Graph capture and unrelated streams are rejected.
#include "CUDAFUSE.h"
#include "FUSE.h"
#include <cuda_runtime_api.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using U=uint64_t;
using Clock=std::chrono::steady_clock;
thread_local std::string api_error;
U now_ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
void cuda_ok(cudaError_t code,const char* where){
    if(code!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(code));
}
struct CallbackState {
    std::mutex mu; std::string error; std::atomic<unsigned> pending{0};
    void fail(const std::string& s){std::lock_guard<std::mutex> lock(mu);if(error.empty())error=s;}
    std::string message(){std::lock_guard<std::mutex> lock(mu);return error;}
};
struct BoundaryTiming {
    U queued_ns=now_ns();
    std::atomic<U> started_ns{0},returned_ns{0};
};
struct Payload {
    void* core; U ticket,gpu,emc; std::shared_ptr<CallbackState> state;
    std::shared_ptr<BoundaryTiming> timing;
    Payload(void* c,U t,U g,U e,std::shared_ptr<CallbackState> s,std::shared_ptr<BoundaryTiming> b):core(c),ticket(t),gpu(g),emc(e),state(std::move(s)),timing(std::move(b)){
        fuse_retain(core);state->pending.fetch_add(1);
    }
    ~Payload(){state->pending.fetch_sub(1);fuse_release(core);}
};
void CUDART_CB boundary(void* opaque) noexcept {
    std::unique_ptr<Payload> p(static_cast<Payload*>(opaque));
    p->timing->started_ns.store(now_ns());
    try{
        // This call only performs CPU/sysfs work. In particular it must never
        // call cudaEventQuery, cudaStreamSynchronize, or another CUDA function.
        if(fuse_gpu_boundary(p->core,p->ticket,p->gpu,p->emc)<0)p->state->fail(fuse_error());
    }catch(const std::exception& e){p->state->fail(e.what());}catch(...){p->state->fail("boundary host function failed");}
    p->timing->returned_ns.store(now_ns());
}
struct Ticket {
    U id=0,block=0,estimated_us=0,switch_us=0,next_gpu=0,next_emc=0;
    cudaEvent_t start=nullptr,end=nullptr;
    std::atomic<bool> sealed{false};
    bool seen=false,issued=false,want_lookahead=false;
    U start_seen_ns=0,due_ns=0;
    std::shared_ptr<BoundaryTiming> timing=std::make_shared<BoundaryTiming>();
    ~Ticket(){if(start)cudaEventDestroy(start);if(end)cudaEventDestroy(end);}
};
struct Adapter {
    void* core;int device;unsigned poll_us;
    std::shared_ptr<CallbackState> state=std::make_shared<CallbackState>();
    std::mutex mu,log_mu;
    std::condition_variable cv;
    std::map<U,std::shared_ptr<Ticket>> tickets;
    cudaStream_t stream=nullptr;bool stream_set=false,stop=false,closing=false;
    std::atomic<bool> allow_lookahead{true};
    std::thread worker;std::ofstream log;
    Adapter(void* c,int d,const char* path,unsigned poll):core(c),device(d),poll_us(poll){
        if(!core || poll_us<10 || poll_us>10000)throw std::runtime_error("invalid core or poll_us (10..10000)");
        int current=-1;cuda_ok(cudaGetDevice(&current),"cudaGetDevice");
        if(current!=device)throw std::runtime_error("adapter device differs from the caller CUDA device");
        log.open(path);if(!log)throw std::runtime_error("cannot open CUDA timing trace");
        log<<"ticket,block_id,kind,host_ns,start_seen_ns,predicted_due_ns,gpu_elapsed_ms,boundary_queued_ns,boundary_started_ns,boundary_returned_ns,callback_duration_ns,start_observation_lag_ns,poll_late_ns\n";
        fuse_retain(core);
        try{worker=std::thread([this]{run();});}catch(...){fuse_release(core);throw;}
    }
    void check(){auto e=state->message();if(!e.empty())throw std::runtime_error(e);if(fuse_status(core)<0)throw std::runtime_error(fuse_error());}
    void record(const Ticket& t,const char* kind,double elapsed=-1){
        std::lock_guard<std::mutex> lock(log_mu);
        const U now=now_ns(),beg=t.timing->started_ns.load(),ret=t.timing->returned_ns.load();
        log<<t.id<<','<<t.block<<','<<kind<<','<<now<<','<<t.start_seen_ns<<','<<t.due_ns<<','<<elapsed
           <<','<<t.timing->queued_ns<<','<<beg<<','<<ret<<','<<(ret>=beg?ret-beg:0)
           <<','<<(t.start_seen_ns>=ret && ret?t.start_seen_ns-ret:0)
           <<','<<(t.due_ns && now>=t.due_ns?now-t.due_ns:0)<<'\n';
    }
    bool observe(const std::shared_ptr<Ticket>& t){
        if(!t->seen){
            auto rc=cudaEventQuery(t->start);
            if(rc==cudaErrorNotReady)return false;
            cuda_ok(rc,"query block start");
            t->seen=true;t->start_seen_ns=now_ns();
            const U delay=t->estimated_us>t->switch_us?t->estimated_us-t->switch_us:0;
            t->due_ns=t->start_seen_ns+delay*1000;
            record(*t,"stream_boundary_callback_observed");
            record(*t,"gpu_start_observed");
        }
        if(!t->sealed.load())return false;
        auto end_state=cudaEventQuery(t->end);
        if(end_state==cudaSuccess){
            if(t->want_lookahead && !t->issued)record(*t,"lookahead_missed_end_ready");
            float elapsed=0;cuda_ok(cudaEventElapsedTime(&elapsed,t->start,t->end),"block elapsed");
            record(*t,"gpu_end_observed",elapsed);return true;
        }
        cuda_ok(end_state==cudaErrorNotReady?cudaSuccess:end_state,"query block end");
        if(t->want_lookahead && !t->issued && now_ns()>=t->due_ns && allow_lookahead.load()){
            int rc=fuse_gpu_lookahead(core,t->id,t->next_gpu,t->next_emc);
            if(rc<0)throw std::runtime_error(fuse_error());
            t->issued=true;
            if(rc==1){record(*t,"lookahead_skipped_stale");return false;}
            if(rc==2){record(*t,"lookahead_noop_same_target");return false;}
            auto after=cudaEventQuery(t->end);
            if(after==cudaErrorNotReady)record(*t,"lookahead_written_end_pending");
            else {cuda_ok(after,"post-write end query");record(*t,"lookahead_written_end_complete_postcheck");}
        }
        return false;
    }
    void run(){
        try{
            cuda_ok(cudaSetDevice(device),"worker cudaSetDevice");
            for(;;){
                std::vector<std::shared_ptr<Ticket>> current;
                {std::unique_lock<std::mutex> lock(mu);
                    if(stop)break;
                    for(auto& item:tickets)current.push_back(item.second);
                    if(current.empty()){cv.wait_for(lock,std::chrono::microseconds(poll_us));continue;}
                }
                for(auto& t:current){
                    if(observe(t)){std::lock_guard<std::mutex> lock(mu);tickets.erase(t->id);cv.notify_all();}
                }
                std::unique_lock<std::mutex> lock(mu);
                cv.wait_for(lock,std::chrono::microseconds(poll_us),[this]{return stop;});
            }
        }catch(const std::exception& e){state->fail(e.what());cv.notify_all();}
    }
    U begin(cudaStream_t s,U block,U boost,U run,U g,U e,U estimate,U lead,U next_g,U next_e,bool lookahead){
        check();
        if(!estimate || (lookahead && estimate<=lead))throw std::runtime_error("invalid block estimate/lookahead lead");
        if(fuse_validate_vector(core,boost,g,e)<0)throw std::runtime_error(fuse_error());
        if(lookahead && fuse_validate_vector(core,boost,next_g,next_e)<0)throw std::runtime_error(fuse_error());
        cudaStreamCaptureStatus capture;
        cuda_ok(cudaStreamIsCapturing(s,&capture),"cudaStreamIsCapturing");
        if(capture!=cudaStreamCaptureStatusNone)throw std::runtime_error("CUDA Graph capture is unsupported by frequency host boundaries");
        {std::lock_guard<std::mutex> lock(mu);
            if(closing)throw std::runtime_error("adapter closing");
            if(stream_set && stream!=s)throw std::runtime_error("one ordered main stream is required; join auxiliary work before the end marker");
            stream=s;stream_set=true;
        }
        auto t=std::make_shared<Ticket>();t->block=block;t->estimated_us=estimate;t->switch_us=lead;
        t->next_gpu=next_g;t->next_emc=next_e;t->want_lookahead=lookahead;
        cuda_ok(cudaEventCreate(&t->start),"create start event");
        cuda_ok(cudaEventCreate(&t->end),"create end event");
        t->id=fuse_submission_begin(core,boost,run);
        if(!t->id)throw std::runtime_error(fuse_error());
        t->timing->queued_ns=now_ns();
        auto* p=new Payload(core,t->id,g,e,state,t->timing);
        auto rc=cudaLaunchHostFunc(s,boundary,p);
        if(rc!=cudaSuccess){delete p;cuda_ok(rc,"queue boundary host function");}
        cuda_ok(cudaEventRecord(t->start,s),"record GPU start");
        {std::lock_guard<std::mutex> lock(mu);tickets[t->id]=t;}
        // The worker can now see this ticket; do not read worker-owned fields here.
        cv.notify_all();return t->id;
    }
    void end(U id){
        check();std::shared_ptr<Ticket> t;
        {std::lock_guard<std::mutex> lock(mu);auto it=tickets.find(id);if(it==tickets.end())throw std::runtime_error("unknown CUDA ticket");t=it->second;}
        if(t->sealed.load())throw std::runtime_error("CUDA ticket submitted twice");
        cuda_ok(cudaEventRecord(t->end,stream),"record GPU end");
        if(fuse_submission_end(core,id)<0)throw std::runtime_error(fuse_error());
        t->sealed.store(true);cv.notify_all();
    }
    void drain(){
        check();
        {std::lock_guard<std::mutex> lock(mu);for(const auto& p:tickets)if(!p.second->sealed.load())throw std::runtime_error("unclosed block scope");}
        if(stream_set)cuda_ok(cudaStreamSynchronize(stream),"drain FUSE main stream");
        std::unique_lock<std::mutex> lock(mu);
        while(!tickets.empty()){
            lock.unlock();check();lock.lock();
            cv.wait_for(lock,std::chrono::milliseconds(1));
        }
        lock.unlock();check();
    }
    std::string shutdown() noexcept {
        allow_lookahead.store(false);
        {std::lock_guard<std::mutex> lock(mu);closing=true;}
        // Never synchronize CUDA while holding the core writer lock or state mutex.
        if(stream_set){auto rc=cudaStreamSynchronize(stream);if(rc!=cudaSuccess)state->fail(std::string("close stream: ")+cudaGetErrorString(rc));}
        {std::lock_guard<std::mutex> lock(mu);stop=true;cv.notify_all();}
        if(worker.joinable())worker.join();
        {std::lock_guard<std::mutex> lock(mu);
            // Finish successful timing records after stopping the competing poller.
            for(auto& p:tickets){try{observe(p.second);}catch(const std::exception& e){state->fail(e.what());}}
            tickets.clear();
        }
        return state->message();
    }
    ~Adapter(){if(worker.joinable())shutdown();fuse_release(core);}
};
}
extern "C" {
const char* fuse_cuda_error(){return api_error.c_str();}
int fuse_cuda_reference_abi_version(){return 5;}
int fuse_cuda_runtime_version(){
    int version=0;auto rc=cudaRuntimeGetVersion(&version);
    if(rc!=cudaSuccess){api_error=cudaGetErrorString(rc);return -1;}return version;
}
void* fuse_cuda_open(void* core,int device,const char* trace,unsigned poll){
    try{return new Adapter(core,device,trace,poll);}catch(const std::exception& e){api_error=e.what();return nullptr;}
}
U fuse_cuda_begin(void* handle,void* stream,U block,U boost,U run,U g,U e,U estimate,U lead,U ng,U ne,int lookahead){
    try{if(!handle)throw std::runtime_error("null CUDA adapter");return static_cast<Adapter*>(handle)->begin(static_cast<cudaStream_t>(stream),block,boost,run,g,e,estimate,lead,ng,ne,lookahead!=0);}
    catch(const std::exception& e){api_error=e.what();return 0;}
}
int fuse_cuda_end(void* handle,U ticket){
    try{if(!handle)throw std::runtime_error("null CUDA adapter");static_cast<Adapter*>(handle)->end(ticket);return 0;}
    catch(const std::exception& e){api_error=e.what();return -1;}
}
int fuse_cuda_drain(void* handle){
    try{if(!handle)throw std::runtime_error("null CUDA adapter");static_cast<Adapter*>(handle)->drain();return 0;}
    catch(const std::exception& e){api_error=e.what();return -1;}
}
int fuse_cuda_close(void* handle){
    if(!handle)return 0;auto* a=static_cast<Adapter*>(handle);auto error=a->shutdown();delete a;
    if(!error.empty()){api_error=error;return -1;}return 0;
}
}
