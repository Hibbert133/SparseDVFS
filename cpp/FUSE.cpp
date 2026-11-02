// Reference implementation written for this demo, NOT author-released SparseDVFS code.
// FUSE = Frequency Unified Scaling Engine (not the FUSE filesystem library).
// Paper: arXiv:2603.21908v1, sections 4.3 and 5. CUDAFUSE.cpp provides explicit
// stream boundaries; no transparent CUDA/TensorRT API interception is claimed.
#include "FUSE.h"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

namespace {
using U = std::uint64_t;
using Clock = std::chrono::steady_clock;
thread_local std::string api_error;
U stamp() { return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(); }
std::string read(const std::string& path) {
    std::ifstream in(path); std::string s;
    if (!in || !std::getline(in,s)) throw std::runtime_error("read failed: " + path);
    return s;
}
U number(const std::string& path) { return std::stoull(read(path)); }
std::vector<std::string> split(const std::string& s) {
    std::istringstream in(s); std::vector<std::string> out; std::string x;
    while (in >> x) out.push_back(x);
    return out;
}
std::set<U> rates(const std::string& s) {
    std::set<U> r; for(const auto& v:split(s)) r.insert(std::stoull(v)); return r;
}
struct Node {
    std::string path, original; int fd=-1; bool mock=false;
    Node(std::string p, bool apply, bool mock_mode):path(std::move(p)),original(read(path)),mock(mock_mode) {
        if(apply) { fd=::open(path.c_str(),O_WRONLY|O_CLOEXEC); if(fd<0) throw std::runtime_error("open failed: "+path+": "+std::strerror(errno)); }
    }
    ~Node(){ if(fd>=0) ::close(fd); }
    void write(U value) {
        if(fd<0) return;
        auto s=std::to_string(value);
        if(::lseek(fd,0,SEEK_SET)<0) throw std::runtime_error("seek failed: "+path);
        if(mock && ::ftruncate(fd,0)<0) throw std::runtime_error("mock truncate failed");
        if(::write(fd,s.data(),s.size())!=static_cast<ssize_t>(s.size())) throw std::runtime_error("write failed: "+path+": "+std::strerror(errno));
    }
};
struct Domain {
    std::unique_ptr<Node> lo,hi; U orig_lo,orig_hi,cur_lo,cur_hi;
    Domain(const std::string& root, bool cpu, bool apply, bool mock) {
        lo=std::make_unique<Node>(root+(cpu?"/scaling_min_freq":"/min_freq"),apply,mock);
        hi=std::make_unique<Node>(root+(cpu?"/scaling_max_freq":"/max_freq"),apply,mock);
        orig_lo=cur_lo=std::stoull(lo->original); orig_hi=cur_hi=std::stoull(hi->original);
    }
    void range(U low,U high) {
        // Widen the permitted interval before narrowing; never temporarily min > max.
        if(cur_hi!=0 && low>cur_hi) { hi->write(high); cur_hi=high; lo->write(low); cur_lo=low; }
        else { lo->write(low); cur_lo=low; hi->write(high); cur_hi=high; }
    }
    void lock(U f){ if(cur_lo!=f || cur_hi!=f) range(f,f); }
    void restore(){ range(orig_lo,orig_hi); }
};
struct Job { Clock::time_point due; U cpu,gpu,emc,queued; };
using GpuCallback = int (*)(void*, unsigned, U);
struct Engine {
    GpuCallback callback=nullptr; void* callback_context=nullptr; U requested_gpu=0;
    std::atomic<unsigned> references{1};
    std::vector<std::unique_ptr<Domain>> cpus;
    std::unique_ptr<Domain> gpu,emc_dev;
    std::unique_ptr<Node> emc_rate,emc_lock,emc_halt;
    std::set<U> cpu_rates,gpu_rates,emc_rates;
    bool apply=false,mock=false,stopping=false,restored=false,dirty=false;
    int device_lock=-1;
    std::mutex mu; std::condition_variable cv; std::deque<Job> jobs; std::thread worker;
    std::string error; std::ofstream log; double last_us=0; U last_emc=0;
    U submission_ticket=0, cpu_run_khz=0, gpu_ticket=0;
    explicit Engine(const std::string& config,int mode,GpuCallback cb=nullptr,void* context=nullptr):callback(cb),callback_context(context) {
        // Construction performs no frequency writes. Incomplete construction is safe.
        std::ifstream in(config); if(!in) throw std::runtime_error("cannot read config");
        std::map<std::string,std::string> cfg; std::string line;
        while(std::getline(in,line)) {
            if(line.empty() || line[0]=='#') continue;
            auto p=line.find('='); if(p==std::string::npos) throw std::runtime_error("bad config line");
            cfg[line.substr(0,p)]=line.substr(p+1);
        }
        if(callback) {
            if(mode!=1 && mode!=2) throw std::runtime_error("GPU callback mode must be real or mock");
            apply=true; mock=mode==2; cpu_rates={0}; gpu_rates=rates(cfg.at("gpu_rates"));
            if(gpu_rates.empty()) throw std::runtime_error("empty GPU frequency grid");
            for(auto f:gpu_rates) if(!f || f%1000000) throw std::runtime_error("GPU frequency must be whole MHz in Hz");
            if(!cfg["log_path"].empty()) { log.open(cfg.at("log_path")); if(!log) throw std::runtime_error("trace open failed"); log<<"queued_ns,start_ns,returned_ns,cpu_khz,gpu_hz,emc_hz,kind,ticket\n"; }
            worker=std::thread([this]{run();}); return;
        }
        if(mode<0 || mode>2) throw std::runtime_error("mode must be 0=dry,1=hardware,2=temporary mock");
        apply=mode!=0; mock=mode==2;
        if(mode==1 && ::geteuid()!=0) throw std::runtime_error("--apply requires root on Thor");
        std::vector<std::string> roots=split(cfg.at("cpu_paths")); roots.push_back(cfg.at("gpu_path"));
        if(cfg.at("emc_backend")!="off") roots.push_back(cfg.at("emc_path"));
        if(!cfg["bwmgr_halt"].empty()) roots.push_back(cfg["bwmgr_halt"]);
        if(mock) {
            auto root=std::filesystem::canonical(cfg.at("mock_root"));
            if(root==root.root_path() || root.string().rfind("/sys",0)==0) throw std::runtime_error("mock_root must be a temporary directory");
            for(const auto& r:roots) {
                auto rel=std::filesystem::relative(std::filesystem::canonical(r),root);
                if(rel.empty() || rel.is_absolute() || *rel.begin()=="..") throw std::runtime_error("mock path escapes mock_root");
            }
        }
        cpu_rates=rates(cfg.at("cpu_rates")); gpu_rates=rates(cfg.at("gpu_rates")); emc_rates=rates(cfg.at("emc_rates"));
        if(cpu_rates.empty() || gpu_rates.empty()) throw std::runtime_error("empty CPU/GPC OPP list");
        if(mode==1) {
            device_lock=::open("/run/lock/sparsedvfs-thor.lock",O_CREAT|O_RDWR|O_CLOEXEC,0600);
            if(device_lock<0) throw std::runtime_error("cannot open device lock");
            if(::flock(device_lock,LOCK_EX|LOCK_NB)<0) { ::close(device_lock); device_lock=-1; throw std::runtime_error("another FUSE process owns the clocks"); }
        }
        try {
            for(const auto& p:split(cfg.at("cpu_paths"))) cpus.emplace_back(std::make_unique<Domain>(p,true,apply,mock));
            if(cpus.empty()) throw std::runtime_error("no CPU policies");
            gpu=std::make_unique<Domain>(cfg.at("gpu_path"),false,apply,mock);
            auto backend=cfg.at("emc_backend");
            if(backend=="devfreq") emc_dev=std::make_unique<Domain>(cfg.at("emc_path"),false,apply,mock);
            else if(backend=="bpmp") {
                if(emc_rates.empty()) throw std::runtime_error("explicit validated EMC rates required");
                emc_rate=std::make_unique<Node>(cfg.at("emc_path")+"/rate",apply,mock);
                emc_lock=std::make_unique<Node>(cfg.at("emc_path")+"/mrq_rate_locked",apply,mock);
                emc_halt=std::make_unique<Node>(cfg.at("bwmgr_halt"),apply,mock);
            } else if(backend!="off") throw std::runtime_error("unknown EMC backend");
            if(!cfg["log_path"].empty()) { log.open(cfg["log_path"]); if(!log) throw std::runtime_error("cannot open FUSE trace"); log<<"queued_ns,start_ns,returned_ns,cpu_khz,gpu_hz,emc_hz,kind,ticket\n"; }
            worker=std::thread([this]{run();});
        } catch(...) { if(device_lock>=0){::close(device_lock);device_lock=-1;} throw; }
    }
    void valid(U c,U g,U e) {
        if(!cpu_rates.count(c) || !gpu_rates.count(g) || (!emc_rates.empty()&&!emc_rates.count(e)) || (emc_rates.empty()&&e!=0)) throw std::runtime_error("frequency is outside the probed OPP grid");
    }
    void check(){ if(!error.empty()) throw std::runtime_error(error); if(restored) throw std::runtime_error("engine already restored"); }
    void apply_cpu(U c){ if(!cpu_rates.count(c)) throw std::runtime_error("invalid CPU rate"); if(apply)dirty=true; for(auto& p:cpus) p->lock(c); }
    void valid_resources(U g,U e) {
        if(!gpu_rates.count(g) || (!emc_rates.empty()&&!emc_rates.count(e)) || (emc_rates.empty()&&e!=0))
            throw std::runtime_error("GPC/EMC frequency is outside the probed OPP grid");
    }
    void resources(U g,U e) {
        valid_resources(g,e); if(apply)dirty=true;
        if(emc_halt && (last_emc!=e || last_emc==0)) { emc_halt->write(1); emc_lock->write(1); emc_rate->write(e); last_emc=e; }
        if(emc_dev) emc_dev->lock(e);
        if(callback) { if(requested_gpu!=g) { if(callback(callback_context,0,g/1000000)) throw std::runtime_error("GPU backend set failed"); requested_gpu=g; } }
        else gpu->lock(g);
    }
    bool resources_change_needed(U g,U e) const {
        if(callback) return requested_gpu!=g;
        if(gpu->cur_lo!=g || gpu->cur_hi!=g)return true;
        if(emc_dev && (emc_dev->cur_lo!=e || emc_dev->cur_hi!=e))return true;
        if(emc_halt && (last_emc==0 || last_emc!=e))return true;
        return false; // Cached requested constraints, not physical clock readback.
    }
    void record(U start,U c,U g,U e,const char* kind,U ticket=0,U queued=0) {
        if(log)log<<(queued?queued:start)<<','<<start<<','<<stamp()<<','<<c<<','<<g<<','<<e<<','<<kind<<','<<ticket<<'\n';
    }
    void set(U c,U g,U e,U queued,const char* kind) {
        valid(c,g,e); U start=stamp();
        // Single writer serializes the coupled vector. Sysfs writes are NOT atomic.
        apply_cpu(c);
        resources(g,e);
        U end=stamp(); last_us=(end-start)/1000.0;
        record(start,c,g,e,kind,0,queued);
    }
    void run() {
        std::unique_lock<std::mutex> lock(mu);
        while(!stopping) {
            if(jobs.empty()){ cv.wait(lock,[this]{return stopping||!jobs.empty();}); continue; }
            if(Clock::now()<jobs.front().due){auto due=jobs.front().due;cv.wait_until(lock,due);continue;}
            auto j=jobs.front();jobs.pop_front();
            try{set(j.cpu,j.gpu,j.emc,j.queued,"lookahead");}catch(const std::exception& e){error=e.what();jobs.clear();}
            cv.notify_all();
        }
    }
    void restore_all() {
        if(restored) return;
        if(!dirty){restored=true;return;}
        jobs.clear(); cv.notify_all(); std::string failures;
        auto attempt=[&](auto fn){try{fn();}catch(const std::exception& e){failures+=std::string(e.what())+"; ";}};
        if(callback) attempt([&]{if(callback(callback_context,1,0)) throw std::runtime_error("GPU backend reset failed");});
        if(emc_rate) {
            attempt([&]{emc_lock->write(1);emc_rate->write(std::stoull(emc_rate->original));});
            attempt([&]{emc_lock->write(std::stoull(emc_lock->original));});
            attempt([&]{emc_halt->write(std::stoull(emc_halt->original));});
        }
        if(emc_dev) attempt([&]{emc_dev->restore();});
        if(gpu) attempt([&]{gpu->restore();});
        for(auto& p:cpus) attempt([&]{p->restore();});
        restored=true; if(!failures.empty()) throw std::runtime_error("restore incomplete: "+failures);
    }
    ~Engine() {
        {std::lock_guard<std::mutex> lock(mu);stopping=true;jobs.clear();cv.notify_all();}
        if(worker.joinable())worker.join();
        try{restore_all();}catch(...){} // explicit fuse_close returns restoration failures.
        if(device_lock>=0)::close(device_lock);
    }
};
template<class F> int guarded(void* h,F fn){
    try{if(!h)throw std::runtime_error("null FUSE handle");auto& e=*static_cast<Engine*>(h);std::lock_guard<std::mutex> lock(e.mu);e.check();fn(e);return 0;}
    catch(const std::exception& e){api_error=e.what();return -1;}
}
}
extern "C" {
int fuse_reference_abi_version(){return 5;}
int fuse_gpu_backend_abi_version(){return 1;}
void* fuse_open_gpu(const char* config,int mode,GpuCallback cb,void* context){try{if(!cb)throw std::runtime_error("null GPU callback");return new Engine(config,mode,cb,context);}catch(const std::exception& e){api_error=e.what();return nullptr;}}
const char* fuse_error(){return api_error.c_str();}
void* fuse_open(const char* config,int mode){try{return new Engine(config,mode);}catch(const std::exception& e){api_error=e.what();return nullptr;}}
int fuse_set(void* h,U c,U g,U e){return guarded(h,[&](Engine& x){x.set(c,g,e,stamp(),"boundary");});}
int fuse_cpu(void* h,U c){return guarded(h,[&](Engine& x){auto start=stamp();x.apply_cpu(c);x.record(start,c,0,0,"cpu_submit_complete");});}
int fuse_enqueue(void* h,U c,U g,U e,U delay_us){return guarded(h,[&](Engine& x){
    x.valid(c,g,e);if(x.jobs.size()>=64)throw std::runtime_error("FUSE queue full");
    Job j{Clock::now()+std::chrono::microseconds(delay_us),c,g,e,stamp()};
    auto it=std::upper_bound(x.jobs.begin(),x.jobs.end(),j,[](const Job& a,const Job& b){return a.due<b.due;});x.jobs.insert(it,j);x.cv.notify_all();
});}
int fuse_cancel(void* h){return guarded(h,[&](Engine& x){x.jobs.clear();x.cv.notify_all();});}
int fuse_status(void* h){return guarded(h,[](Engine&){});}
int fuse_validate_vector(void* h,U c,U g,U e){return guarded(h,[&](Engine& x){x.valid(c,g,e);});}
U fuse_submission_begin(void* h,U boost,U run) {
    U result=0;
    int rc=guarded(h,[&](Engine& x){
        if(!x.cpu_rates.count(run))throw std::runtime_error("invalid CPU run rate");
        if(boost<run)throw std::runtime_error("CPU submission frequency must not be below CPU run frequency");
        auto start=stamp();x.apply_cpu(boost);x.cpu_run_khz=run;result=++x.submission_ticket;
        x.record(start,boost,0,0,"cpu_submission_boost",result);
    });
    return rc==0?result:0;
}
int fuse_submission_end(void* h,U ticket) {
    bool stale=false;
    int rc=guarded(h,[&](Engine& x){
        if(!ticket || ticket!=x.submission_ticket){stale=true;return;}
        auto start=stamp();x.apply_cpu(x.cpu_run_khz);x.record(start,x.cpu_run_khz,0,0,"cpu_submission_release",ticket);
    });return rc==0?(stale?1:0):rc;
}
int fuse_gpu_boundary(void* h,U ticket,U g,U e) {
    return guarded(h,[&](Engine& x){
        if(!ticket || ticket<=x.gpu_ticket || ticket>x.submission_ticket)throw std::runtime_error("GPU boundaries are not in submission order");
        auto start=stamp();x.resources(g,e);x.gpu_ticket=ticket;x.record(start,0,g,e,"gpu_stream_boundary",ticket);
    });
}
int fuse_gpu_lookahead(void* h,U ticket,U g,U e) {
    int outcome=0;
    int rc=guarded(h,[&](Engine& x){
        if(!ticket || ticket!=x.gpu_ticket){outcome=1;return;}
        x.valid_resources(g,e);
        if(!x.resources_change_needed(g,e)){
            outcome=2;x.record(stamp(),0,g,e,"gpu_event_lookahead_unchanged",ticket);return;
        }
        auto start=stamp();x.resources(g,e);x.record(start,0,g,e,"gpu_event_lookahead",ticket);
    });return rc==0?outcome:rc;
}
double fuse_last_write_us(void* h){if(!h)return -1;auto& x=*static_cast<Engine*>(h);std::lock_guard<std::mutex> l(x.mu);return x.last_us;}
void fuse_retain(void* h){if(h)static_cast<Engine*>(h)->references.fetch_add(1);}
void fuse_release(void* h){if(h && static_cast<Engine*>(h)->references.fetch_sub(1)==1)delete static_cast<Engine*>(h);}
int fuse_close(void* h){
    if(!h)return 0;auto* x=static_cast<Engine*>(h);int result=0;
    try{std::lock_guard<std::mutex> lock(x->mu);x->restore_all();}catch(const std::exception& e){api_error=e.what();result=-1;}
    fuse_release(x);return result;
}
}
