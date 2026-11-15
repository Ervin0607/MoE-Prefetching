"""Drive the integrated LLMSimulator offloader (no averaged GPU timing)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import yaml
from evaluate_predictor import ROOT, load_records, routes


def write_trace(path, records, limit=0):
    with path.open('w') as f:
        for record in records:
            pairs=record['exp_ids'][:limit] if limit else record['exp_ids']
            for i,t in enumerate(pairs):
                f.write(' '.join(map(str,[int(i==0 or record['input_ids'][i]==1)]+[e for pair in routes(t) for e in pair]))+'\n')


def run_one(build, dest, config):
    if not sys.platform.startswith('linux'):
        raise RuntimeError('Native LLMSimulator requires Linux/WSL; run this command in Ubuntu.')
    dest.mkdir(parents=True,exist_ok=True)
    config['log']['output_directory']=str(dest)+'/'
    config['expert_offload']['events']=str(dest/'events.csv')
    config['expert_offload']['summary']=str(dest/'summary.csv')
    cfg=dest/'config.yaml';cfg.write_text(yaml.safe_dump(config))
    proc=subprocess.run([str(build/'run'),str(cfg)],cwd=build,capture_output=True,text=True,timeout=600)
    (dest/'run.log').write_text(proc.stdout+'\n'+proc.stderr)
    proc.check_returncode()
    with (dest/'summary.csv').open() as f: row=next(csv.DictReader(f))
    return {k:(v if k=='mode' else float(v)) for k,v in row.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokens',type=int,default=128,help='Tokens per sequence; 0 means all')
    p.add_argument('--sequences',type=int,default=8)
    p.add_argument('--bandwidths',type=float,nargs='+',default=[32])
    p.add_argument('--cache-gb',type=float,nargs='+',default=[32])
    p.add_argument('--budgets',type=int,nargs='+',default=[2])
    p.add_argument('--modes',nargs='+',default=['streaming','cache','temporal','spatial','combined','random','oracle'])
    p.add_argument('--ramulator',action='store_true',help='Enable detailed GPU HBM timing (host link stays analytical)')
    p.add_argument('--build',type=Path,default=ROOT/'LLMSimulator/build-final')
    p.add_argument('--output',type=Path,default=ROOT/'final_evaluation/results/native')
    p.add_argument('--background-w',type=float,default=127)
    p.add_argument('--link-pj-byte',type=float,default=0)
    p.add_argument('--predictor-w',type=float,default=0)
    a=p.parse_args()
    if not sys.platform.startswith('linux'): p.error('Native LLMSimulator requires Linux/WSL; run in Ubuntu')
    if a.tokens<0 or not 1<=a.sequences<=8: p.error('Invalid tokens/sequences')
    if a.output.exists(): p.error('Output directory already exists; choose a fresh --output for reproducibility')
    a.output=a.output.resolve();a.build=a.build.resolve();a.output.mkdir(parents=True)
    trace=ROOT/'mid_evaluation/data/hf_mixtral_c4_routing.jsonl'
    records=load_records(trace)
    write_trace(a.output/'profile.tsv',records[:4])
    base=yaml.safe_load((ROOT/'final_evaluation/configs/native_base.yaml').read_text())
    base['system']['num_device']=1;base['system']['processor_type']='GPU'
    base['system']['optimization']['use_ramulator']=a.ramulator
    base['serving']['max_batch_size']=1
    base['simulation']['output_len']=max(a.tokens,2048)+2
    base['simulation']['exit_out_of_memory']=True
    rows=[]
    for sid in range(4,4+a.sequences):
        trace_path=a.output/f'trace_{sid}.tsv';write_trace(trace_path,[records[sid]],a.tokens)
        for bw in a.bandwidths:
            for gb in a.cache_gb:
                for mode in a.modes:
                    budgets=[0] if mode in ('streaming','cache') else ([2] if mode=='oracle' else a.budgets)
                    for budget in budgets:
                        cfg=json.loads(json.dumps(base))
                        cfg['expert_offload']=dict(mode=mode,budget=budget,cache_bytes=int(gb*1e9),link_GBs=bw,
                            profile=str(a.output/'profile.tsv'),trace=str(trace_path),
                            background_w=a.background_w,link_pj_byte=a.link_pj_byte,predictor_w=a.predictor_w)
                        dest=a.output/f'seq{sid}_{mode}_b{budget}_bw{bw:g}_cache{gb:g}'
                        row=run_one(a.build,dest,cfg)
                        row.update(sequence_id=sid,cache_bytes=int(gb*1e9),link_gbps=bw,budget=budget)
                        # expertFFN stamp encloses native wait and predictor delay.
                        row['expert_execution_s']=row['expert_region_s']-row['exposed_stall_s']-(row['tokens']*31*1e-6 if mode not in ('cache','streaming') and budget!=0 else 0)
                        rows.append(row)
                        print(f"seq {sid} {mode} b={budget}: {row['ms_per_token']:.3f} ms/token",flush=True)
    for r in rows:
        same=[x for x in rows if all(x[k]==r[k] for k in ('sequence_id','cache_bytes','link_gbps'))]
        for control in ('streaming','cache'):
            ref=next((x for x in same if x['mode']==control),None)
            r[f'speedup_vs_{control}']=ref['elapsed_s']/r['elapsed_s'] if ref else ''
            r[f'energy_saving_vs_{control}']=1-r['total_j']/ref['total_j'] if ref else ''
    with (a.output/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    hashes={}
    for path in [trace,a.build/'run',a.build/'libllm_system.so',ROOT/'SSD-offloading/patch/ssd_offloading.patch']+[ROOT/p for p in ['LLMSimulator/src/hardware/expert_offload.h','LLMSimulator/src/hardware/device.h','LLMSimulator/src/hardware/cluster.cpp','LLMSimulator/src/module/route.cpp','LLMSimulator/eval/test.cpp']]:
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (a.output/'manifest.json').write_text(json.dumps(dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},hashes=hashes,
        energy='SSD-offloading DDR5 9.2pJ/bit and 127W background; original LLMSimulator HBM writes 0.891nJ/32B; link and predictor power explicitly excluded by default, sensitivity options available',
        scope='Native per-layer GPU graph, batch-one FP16 Mixtral, cold record starts, causal predictor; analytical serialized host transfers; C4 provenance not verified as autoregressive decode'),indent=2)+'\n')
    # Existing plotter expects these derived metrics.
    print('Completed:',a.output)
if __name__=='__main__':main()
