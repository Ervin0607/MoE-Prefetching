import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from simulator import Simulator, Config, EXPERT_BYTES

class SimulatorTests(unittest.TestCase):
    def make(self,slots=8):
        return Simulator(Config(cache_bytes=slots*EXPERT_BYTES,predictor_us=0))
    def test_cache_reuse(self):
        s=self.make();s.layer([(0,0),(0,1)],[],0,0);copies=s.copies
        s.layer([(0,0),(0,1)],[],0,0)
        self.assertEqual(s.copies,copies);self.assertEqual(s.hits,2)
    def test_late_prefetch_merges_transfer(self):
        s=self.make();s.layer([(0,0),(0,1)],[(1,0),(1,1)],0,0)
        s.layer([(1,0),(1,1)],[],0,0)
        self.assertEqual(s.copies,4);self.assertEqual(s.late,2)
        self.assertAlmostEqual(s.time,s.link)
    def test_ready_prefetch_hides_wait(self):
        s=self.make();s.layer([(0,0),(0,1)],[(1,0),(1,1)],0,1)
        stall=s.stall;s.layer([(1,0),(1,1)],[],0,0)
        self.assertEqual(s.stall,stall);self.assertEqual(s.hits,2)
    def test_capacity_pins_and_layer_identity(self):
        s=self.make(2);s.layer([(0,0),(0,1)],[(1,0)],0,0)
        self.assertEqual(s.copies,2)
        s.layer([(1,0),(1,1)],[],0,0)
        self.assertEqual(s.copies,4);self.assertLessEqual(s.peak,2)
    def test_wrong_prefetch_and_drain_charged(self):
        s=self.make();s.layer([(0,0),(0,1)],[(1,4)],0,0)
        r=s.finish(1,1)
        self.assertEqual(r['unused_speculative_bytes'],EXPERT_BYTES)
        self.assertGreater(r['drain_s'],0)
        self.assertAlmostEqual(r['total_j'],sum(r[x] for x in ['gpu_dynamic_j','transfer_j','background_j','predictor_j']))
    def test_capacity_validation(self):
        with self.assertRaises(ValueError): Config(cache_bytes=79_000_000_000).validate()
    def test_zero_budget_matches_cache(self):
        from run_evaluation import replay
        from test_predictor import record
        timing=dict(other_layer_s=.001,expert_layer_s=.001,dynamic_j_per_token=1)
        cfg=Config(cache_bytes=8*EXPERT_BYTES)
        a=replay([record((0,1))],[record((0,1))],'cache',0,cfg,timing,2)[0]
        b=replay([record((0,1))],[record((0,1))],'combined',0,cfg,timing,2)[0]
        for key in ('elapsed_s','host_bytes','total_j'): self.assertEqual(a[key],b[key])

if __name__=='__main__':unittest.main()
