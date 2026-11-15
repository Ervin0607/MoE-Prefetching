"""Tests a clean-built LLMSimulator binary, not the Python approximation."""
import csv
import os
import json
from pathlib import Path
import sys
import tempfile
import unittest
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluate_predictor import ROOT, Predictor, evaluate
from run_native import run_one, write_trace

BUILD=Path(os.environ.get('MOE_NATIVE_BUILD', str(ROOT/'LLMSimulator/build-final'))).resolve()

@unittest.skipUnless(sys.platform.startswith('linux') and (BUILD/'run').exists(),
                         'Native tests require Linux/WSL and a built simulator; set MOE_NATIVE_BUILD')
class NativeIntegrationTests(unittest.TestCase):
    def execute(self,mode,budget=2,cache_gb=32):
        record={'input_ids':[1,10,11], 'exp_ids': [[[0]*32,[1]*32],[[2]*32,[3]*32],[[0]*32,[1]*32]]}
        profile={'input_ids':[1,10], 'exp_ids': [[[0]*32,[1]*32],[[2]*32,[3]*32]]}
        with tempfile.TemporaryDirectory(prefix='moe-native-test-') as d:
            d=Path(d);write_trace(d/'trace',[record]);write_trace(d/'profile',[profile])
            cfg=yaml.safe_load((ROOT/'final_evaluation/configs/native_base.yaml').read_text())
            cfg['system']['num_device']=1;cfg['system']['processor_type']='GPU';cfg['serving']['max_batch_size']=1
            cfg['simulation']['output_len']=20
            cfg['expert_offload']=dict(mode=mode,budget=budget,cache_bytes=int(cache_gb*1e9),link_GBs=32,profile=str(d/'profile'),trace=str(d/'trace'))
            result=run_one(BUILD,d/'run',cfg)
            with (d/'run/events.csv').open() as f: events=list(csv.DictReader(f))
            raw=list((d/'run').glob('*decode.csv'))
            with raw[0].open() as f: nrows=len(list(csv.DictReader(f)))
            return result,events,nrows,profile,record

    def test_native_graph_and_energy_conservation(self):
        r,events,nrows,_,_=self.execute('combined')
        self.assertEqual(nrows,3)
        self.assertEqual(sum(x['event']=='wait' for x in events),3*32)
        copies=[e for e in events if e['event']=='copy']
        self.assertAlmostEqual(r['host_bytes'],len(copies)*352321536)
        ordered=sorted(copies,key=lambda e:float(e['start_ns']))
        for a,b in zip(ordered,ordered[1:]):self.assertLessEqual(float(a['end_ns']),float(b['start_ns'])+1e-6)
        self.assertLessEqual(r['peak_cache_bytes'],32e9)
        self.assertAlmostEqual(r['total_j'],sum(r[k] for k in ('gpu_dynamic_j','host_read_j','hbm_write_j','link_j','background_j','predictor_j')))
        self.assertGreater(r['gpu_dynamic_j'],0)
        self.assertAlmostEqual(r['host_read_j'],r['host_bytes']*73.6e-12)
        self.assertAlmostEqual(r['hbm_write_j'],r['host_bytes']*27.84375e-12)
        self.assertAlmostEqual(r['background_j'],r['elapsed_s']*127)
        self.assertGreater(r['expert_region_s']-r['exposed_stall_s']-3*31e-6,0)

    def test_native_predictor_matches_python_reference(self):
        r,_,_,profile,record=self.execute('combined')
        ref=evaluate(Predictor([profile]),[record],'combined',2)
        self.assertAlmostEqual(r['recall'],ref['recall'])
        self.assertAlmostEqual(r['precision'],ref['precision'])

    def test_zero_budget_matches_native_cache(self):
        a,*_=self.execute('cache',0)
        b,*_=self.execute('combined',0)
        for key in ('elapsed_s','gpu_dynamic_j','host_bytes','total_j'):self.assertAlmostEqual(a[key],b[key])

    def test_streaming_loads_every_actual_expert(self):
        r,*_=self.execute('streaming',0)
        self.assertEqual(r['host_bytes'],3*32*2*352321536)
        self.assertEqual(r['ready_hits'],0)

if __name__=='__main__':unittest.main()
