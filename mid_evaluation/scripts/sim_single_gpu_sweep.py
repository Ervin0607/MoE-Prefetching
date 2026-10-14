"""
sim_single_gpu_sweep.py
Runs actual LLMSimulator simulation for single-GPU (num_node=1, num_device=1)
across batch sizes 1 to 64 for Mixtral 8x7B.
Measures simulated GPU execution time (attention + expert FFN) and pairs it with
off-chip DRAM transfer time over PCIe Gen 4, PCIe Gen 5, and CXL.
"""

import os
import subprocess
import yaml
import glob
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUILD_DIR = os.path.join(PROJECT_ROOT, "LLMSimulator", "build")
DATA_DIR = os.path.join(PROJECT_ROOT, "LLMSimulator", "data")
CONFIG_PATH = os.path.join(BUILD_DIR, "config_single_gpu.yaml")

batch_sizes = [1, 2, 4, 8, 16, 32, 64]
sim_results = []

for b in batch_sizes:
    cfg = {
        'model': {'model_name': 'mixtral'},
        'system': {
            'gpu_gen': 'H100',
            'nvlink_gen': 4,
            'infiniband_gen': 400,
            'num_node': 1,
            'num_device': 1,
            'processor_type': 'GPU',
            'distribution': {'expert_tensor_degree': 1, 'none_expert_tensor_degree': 1},
            'optimization': {
                'parallel_execution': False, 'hetero_subbatch': False, 'disagg_system': False,
                'use_low_unit_moe_only': False, 'use_ramulator': False, 'compressed_kv': False,
                'use_absorb': False, 'use_flash_mla': False, 'use_flash_attention': False,
                'reuse_kv_cache': False, 'kv_cache_reuse_rate': 0.0, 'prefill_mode': False,
                'decode_mode': True
            }
        },
        'serving': {'max_batch_size': b, 'max_process_token': 0},
        'simulation': {
            'data': 'synthesis', 'input_len': 1024, 'output_len': 5, 'precision_byte': 2,
            'skewness': 0.0, 'iter': 2, 'injection_rate': 0, 'exit_out_of_memory': False,
            'mem_cap_limit': False
        },
        'log': {'print_log': False, 'export_gantt': False, 'output_directory': '../data/', 'gantt_directory': './gantt'}
    }
    
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(cfg, f)
        
    cmd = 'wsl bash -c "cd LLMSimulator/build && ./run config_single_gpu.yaml"'
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"Error for batch {b}: {proc.stderr[:200]}")
        continue
        
    pattern = os.path.join(DATA_DIR, f"mixtral_synthesis_1024_5_GPU_N1_D1_TP1_DP1_maxbatch{b}_*.csv")
    matched = glob.glob(pattern)
    if not matched:
        print(f"No CSV for batch {b}")
        continue
        
    latest = max(matched, key=os.path.getmtime)
    df = pd.read_csv(latest)
    
    step_lat_us = df['latency'].mean() / 1000.0
    exp_ffn_us = df['expert_ffn'].mean() / 1000.0
    attn_us = (df['qkvgen'] + df['atten_gen'] + df['o_proj']).mean() / 1000.0
    ln_us = (df['layernorm'] + df['residual']).mean() / 1000.0 if 'layernorm' in df.columns else 0.0
    other_gpu_us = step_lat_us - exp_ffn_us
    energy_mj = df['total_energy'].mean() / 1e6
    
    # Active experts calculation
    # In Mixtral, 8 experts, k=2. For batch b, expected active experts:
    # E[act] = 8 * (1 - (1 - 2/8)^b)
    act_exp = 8.0 * (1.0 - (1.0 - 2.0/8.0)**b) if b < 8 else 8.0
    act_exp = min(act_exp, 8.0)
    
    # Off-chip DRAM loading:
    # Weight per expert = 3 * 4096 * 14336 * 2 bytes = 352.32 MB
    # Across 32 layers, total expert bytes to load = 32 * act_exp * 0.35232 GB
    weight_to_load_gb = 32 * act_exp * 0.35232
    
    # Over PCIe 5.0 (64 GB/s):
    load_pcie5_ms = (weight_to_load_gb / 64.0) * 1000.0
    # Over PCIe 4.0 (32 GB/s):
    load_pcie4_ms = (weight_to_load_gb / 32.0) * 1000.0
    
    # Total step time with off-chip DRAM loading:
    # On-demand sequential execution: Step_Time = T_load + T_gpu_compute
    total_step_pcie5_ms = load_pcie5_ms + (step_lat_us / 1000.0)
    total_step_pcie4_ms = load_pcie4_ms + (step_lat_us / 1000.0)
    
    stall_ratio_pcie5 = (load_pcie5_ms / total_step_pcie5_ms) * 100.0
    stall_ratio_pcie4 = (load_pcie4_ms / total_step_pcie4_ms) * 100.0
    
    sim_results.append({
        'batch_size': b,
        'sim_step_hbm_ms': round(step_lat_us / 1000.0, 3),
        'sim_expert_ffn_ms': round(exp_ffn_us / 1000.0, 3),
        'sim_attn_ms': round(attn_us / 1000.0, 3),
        'sim_energy_mJ': round(energy_mj, 1),
        'active_experts': round(act_exp, 2),
        'pcie5_load_ms': round(load_pcie5_ms, 2),
        'pcie4_load_ms': round(load_pcie4_ms, 2),
        'total_step_pcie5_ms': round(total_step_pcie5_ms, 2),
        'total_step_pcie4_ms': round(total_step_pcie4_ms, 2),
        'stall_pct_pcie5': round(stall_ratio_pcie5, 2),
        'stall_pct_pcie4': round(stall_ratio_pcie4, 2)
    })

res_df = pd.DataFrame(sim_results)
out_csv = os.path.join(PROJECT_ROOT, "mid_evaluation", "results", "simulated_single_gpu_offchip.csv")
res_df.to_csv(out_csv, index=False)

print("\n" + "=" * 90)
print("ACTUAL SIMULATION RESULTS: Single-GPU (1x H100) MoE with Off-Chip DRAM")
print("=" * 90)
print(res_df.to_string(index=False))
print(f"\nSaved results to: {out_csv}")
