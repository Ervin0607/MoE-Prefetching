#include "hardware/expert_offload.h"
#include <cassert>
#include <filesystem>
using namespace llm_system;
int main(int argc,char**argv) {
  std::string d=argv[1];std::filesystem::create_directories(d);
  std::ofstream f(d+"/trace");
  for(int t=0;t<2;t++){f<<(t==0);for(int l=0;l<32;l++)f<<" 0 1";f<<'\n';}f.close();
  OffloadConfig c;c.expert_bytes=32;c.cache_bytes=64;c.hbm_bytes=1000;c.reserved_bytes=0;
  c.startup_ns=0;c.predictor_ns=0;c.mode="streaming";c.budget=0;
  ExpertOffload engine(c,d+"/trace",d+"/trace",d+"/events");
  double epoch=0;
  for(int t=0;t<2;t++) {
    engine.begin(t,epoch);double now=0;
    for(int l=0;l<32;l++) {auto a=engine.route(now);assert(a[0]==0 && a[1]==1);}
    assert(std::abs(now-64)<1e-8);engine.end(now,1,now);epoch+=now;
  }
  engine.write(d+"/summary");
  c.cache_bytes=32;bool failed=false;
  try {ExpertOffload bad(c,d+"/trace",d+"/trace",d+"/bad");} catch(const std::runtime_error&) {failed=true;}
  assert(failed);
  c.cache_bytes=64;c.mode="unknown";failed=false;
  try {ExpertOffload bad(c,d+"/trace",d+"/trace",d+"/bad");} catch(const std::runtime_error&) {failed=true;}
  assert(failed);
}
