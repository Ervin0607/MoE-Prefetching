"""Finite HBM cache and single FIFO non-preemptive host-copy link, time in seconds."""
from collections import OrderedDict
from dataclasses import dataclass

EXPERT_BYTES = 3 * 4096 * 14336 * 2

@dataclass
class Config:
    cache_bytes: int = 32_000_000_000
    hbm_bytes: int = 80_000_000_000
    reserved_bytes: int = 12_000_000_000
    link_gbps: float = 32
    host_gbps: float = 100
    startup_us: float = 2
    background_w: float = 50
    host_pj_byte: float = 100
    link_pj_byte: float = 100
    hbm_write_pj_byte: float = 20
    predictor_us: float = 1
    predictor_w: float = 1

    def validate(self):
        if self.cache_bytes < 2 * EXPERT_BYTES or self.cache_bytes + self.reserved_bytes > self.hbm_bytes:
            raise ValueError('Cache must hold top-2 and fit HBM after reservations')
        if min(self.link_gbps, self.host_gbps) <= 0:
            raise ValueError('Bandwidth must be positive')
        if any(v < 0 for v in vars(self).values()):
            raise ValueError('Negative configuration value')

class Simulator:
    def __init__(self, cfg, streaming=False):
        cfg.validate()
        self.cfg, self.streaming = cfg, streaming
        self.slots = cfg.cache_bytes // EXPERT_BYTES
        self.cache = OrderedDict()
        self.time = self.link = self.stall = 0.0
        self.copies = self.speculative = self.unused = self.hits = self.late = 0
        self.evictions = self.requests = self.peak = 0
        self.prediction_time = 0.0

    def evict(self, key):
        entry = self.cache.pop(key)
        self.unused += entry['spec'] and not entry['used']
        self.evictions += 1

    def issue(self, key, now, pins, speculative):
        if key in self.cache:
            return now
        while len(self.cache) >= self.slots:
            victim = next((k for k,v in self.cache.items() if k not in pins and v['ready'] <= now), None)
            if victim is not None:
                self.evict(victim)
                break
            if speculative:
                return now  # Admission failure; never evict pinned/in-flight weights.
            pending = [v['ready'] for k,v in self.cache.items() if k not in pins]
            if not pending:
                raise RuntimeError('Insufficient capacity for pinned demands')
            now = max(now, min(pending))
        duration = self.cfg.startup_us * 1e-6 + EXPERT_BYTES / (min(self.cfg.link_gbps, self.cfg.host_gbps) * 1e9)
        self.link = max(self.link, now) + duration
        self.cache[key] = dict(ready=self.link, spec=speculative, used=False)
        self.copies += 1
        self.speculative += speculative
        self.peak = max(self.peak, len(self.cache))
        return now

    def layer(self, actual, predicted, other_s, expert_s, prediction_enabled=False):
        self.time += other_s
        arrival = self.time
        pins = set(actual)
        for key in actual:
            self.requests += 1
            if key in self.cache:
                self.hits += self.cache[key]['ready'] <= arrival
                self.late += self.cache[key]['spec'] and self.cache[key]['ready'] > arrival
            self.time = self.issue(key, self.time, pins, False)
        self.time = max([self.time] + [self.cache[k]['ready'] for k in actual])
        self.stall += self.time - arrival
        for key in actual:
            self.cache[key]['used'] = True
            self.cache.move_to_end(key)
        if prediction_enabled:
            overhead = self.cfg.predictor_us * 1e-6
            self.time += overhead
            self.prediction_time += overhead
        # Predictions issued after current demands are ready. This conservative
        # policy overlaps next-layer copies with current FFN and next attention.
        for key in predicted:
            self.issue(key, self.time, pins, True)
        self.time += expert_s
        if self.streaming:
            for key in actual:
                self.evict(key)

    def finish(self, tokens, gpu_dynamic_j):
        elapsed = max(self.time, self.link)  # Charge final speculative drain.
        unused = self.unused + sum(v['spec'] and not v['used'] for v in self.cache.values())
        traffic = self.copies * EXPERT_BYTES
        transfer_j = traffic * (self.cfg.host_pj_byte + self.cfg.link_pj_byte + self.cfg.hbm_write_pj_byte) * 1e-12
        background_j = elapsed * self.cfg.background_w
        prediction_j = self.prediction_time * self.cfg.predictor_w
        energy = gpu_dynamic_j + transfer_j + background_j + prediction_j
        return dict(tokens=tokens, elapsed_s=elapsed, decode_s=self.time, drain_s=elapsed-self.time,
                    ms_per_token=elapsed/tokens*1000, tokens_per_s=tokens/elapsed,
                    exposed_stall_s=self.stall, transfer_service_s=self.copies*(self.cfg.startup_us*1e-6+EXPERT_BYTES/(min(self.cfg.link_gbps,self.cfg.host_gbps)*1e9)),
                    host_bytes=traffic, speculative_copies=self.speculative, unused_speculative_bytes=unused*EXPERT_BYTES,
                    ready_hits=self.hits, late_prefetches=self.late, requests=self.requests,
                    evictions=self.evictions, peak_cache_bytes=self.peak*EXPERT_BYTES,
                    gpu_dynamic_j=gpu_dynamic_j, transfer_j=transfer_j, background_j=background_j,
                    predictor_j=prediction_j, total_j=energy, joules_per_token=energy/tokens,
                    energy_delay_product=energy*elapsed)
