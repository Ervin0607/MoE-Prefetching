"""
generate_run_logs.py
Executes all LLMSimulator configurations and Python benchmark models,
capturing full stdout/stderr verbatim logs into mid_evaluation/logs/.
"""

import os
import subprocess
import yaml
import glob
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(PROJECT_ROOT, "mid_evaluation", "logs")
LLM_SIM_DIR = os.path.join(PROJECT_ROOT, "LLMSimulator")
BUILD_DIR = os.path.join(LLM_SIM_DIR, "build")
CONFIG_PATH = os.path.join(BUILD_DIR, "config.yaml")

os.makedirs(LOG_DIR, exist_ok=True)

def get_base_config():
    return {
        'model': {
            'model_name': 'mixtral'
        },
        'system': {
            'gpu_gen': 'H100',
            'nvlink_gen': 4,
            'infiniband_gen': 400,
            'num_node': 1,
            'num_device': 4,
            'processor_type': 'GPU',
            'distribution': {
                'expert_tensor_degree': 1,
                'none_expert_tensor_degree': 1
            },
            'optimization': {
                'parallel_execution': False,
                'hetero_subbatch': False,
                'disagg_system': False,
                'use_low_unit_moe_only': False,
                'use_ramulator': False,
                'compressed_kv': False,
                'use_absorb': False,
                'use_flash_mla': False,
                'use_flash_attention': False,
                'reuse_kv_cache': False,
                'kv_cache_reuse_rate': 0.0,
                'prefill_mode': False,
                'decode_mode': True
            }
        },
        'serving': {
            'max_batch_size': 32,
            'max_process_token': 0
        },
        'simulation': {
            'data': 'synthesis',
            'input_len': 1024,
            'output_len': 5,
            'precision_byte': 2,
            'skewness': 0.0,
            'iter': 3,
            'injection_rate': 0,
            'exit_out_of_memory': False,
            'mem_cap_limit': False
        },
        'log': {
            'print_log': True,
            'export_gantt': False,
            'output_directory': '../data/',
            'gantt_directory': './gantt'
        }
    }

def run_and_save_log(config_dict, log_filename, description):
    print(f"\n>>> Running: {description} -> {log_filename}")
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)
        
    cmd = 'wsl bash -c "cd LLMSimulator/build && ./run config.yaml"'
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    log_path = os.path.join(LOG_DIR, log_filename)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"=== Simulation Log: {description} ===\n")
        f.write(f"=== Command: cd LLMSimulator/build && ./run config.yaml ===\n\n")
        f.write("--- STDOUT ---\n")
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(proc.stderr)
            
    print(f"    Saved: {log_path} ({os.path.getsize(log_path)} bytes)")

def run_command_log(cmd_str, log_filename, description):
    print(f"\n>>> Running: {description} -> {log_filename}")
    proc = subprocess.run(cmd_str, shell=True, capture_output=True, text=True, cwd=PROJECT_ROOT)
    log_path = os.path.join(LOG_DIR, log_filename)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"=== Script Execution Log: {description} ===\n")
        f.write(f"=== Command: {cmd_str} ===\n\n")
        f.write("--- STDOUT ---\n")
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(proc.stderr)
    print(f"    Saved: {log_path} ({os.path.getsize(log_path)} bytes)")

if __name__ == "__main__":
    # 1. Baseline GPU Analytic
    cfg1 = get_base_config()
    cfg1['system']['processor_type'] = 'GPU'
    cfg1['system']['optimization']['parallel_execution'] = False
    cfg1['system']['optimization']['use_ramulator'] = False
    run_and_save_log(cfg1, "01_baseline_4xh100_analytic.log", "Baseline 4x H100 GPU (Analytic Memory)")

    # 2. Duplex Analytic
    cfg2 = get_base_config()
    cfg2['system']['processor_type'] = 'GPU+LOGIC'
    cfg2['system']['optimization']['parallel_execution'] = False
    cfg2['system']['optimization']['use_ramulator'] = False
    run_and_save_log(cfg2, "02_duplex_4xh100_analytic.log", "Duplex GPU + Logic-PIM (Analytic Memory)")

    # 3. Duplex + PE Analytic
    cfg3 = get_base_config()
    cfg3['system']['processor_type'] = 'GPU+LOGIC'
    cfg3['system']['optimization']['parallel_execution'] = True
    cfg3['system']['optimization']['use_ramulator'] = False
    run_and_save_log(cfg3, "03_duplex_pe_4xh100_analytic.log", "Duplex + PE Co-Processing (Analytic Memory)")

    # 4. Baseline GPU Ramulator
    cfg4 = get_base_config()
    cfg4['system']['processor_type'] = 'GPU'
    cfg4['system']['optimization']['parallel_execution'] = False
    cfg4['system']['optimization']['use_ramulator'] = True
    run_and_save_log(cfg4, "04_baseline_4xh100_ramulator.log", "Baseline 4x H100 GPU (Ramulator 2.0 Cycle-Accurate)")

    # 5. Duplex Ramulator
    cfg5 = get_base_config()
    cfg5['system']['processor_type'] = 'GPU+LOGIC'
    cfg5['system']['optimization']['parallel_execution'] = False
    cfg5['system']['optimization']['use_ramulator'] = True
    run_and_save_log(cfg5, "05_duplex_4xh100_ramulator.log", "Duplex GPU + Logic-PIM (Ramulator 2.0 Cycle-Accurate)")

    # 6. Single-GPU MoE Off-chip model log
    run_command_log("python mid_evaluation/scripts/model_offchip_moe.py", 
                    "06_single_gpu_offchip_model.log", 
                    "Deliverable 2 & 3: Single-GPU Off-Chip DRAM Modeling")

    # 7. Interconnect decode benchmarks log
    run_command_log("python mid_evaluation/scripts/exp2_interconnect_decode_benchmarks.py", 
                    "07_interconnect_benchmarks.log", 
                    "Deliverable 2 & 3: Interconnect Scaling & Decode Step Breakdown")

    # 8. Correlation analysis log
    run_command_log("python mid_evaluation/scripts/correlation_analysis.py", 
                    "08_correlation_analysis.log", 
                    "Deliverable 4: Spatio-Temporal Routing Correlation Analysis")

    print("\n[SUCCESS] All execution and simulation logs saved to mid_evaluation/logs/")
