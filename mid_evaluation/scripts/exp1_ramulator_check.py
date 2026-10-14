import os
import subprocess
import yaml
import pandas as pd
import glob

LLM_SIM_DIR = os.path.abspath("LLMSimulator")
BUILD_DIR = os.path.join(LLM_SIM_DIR, "build")
DATA_DIR = os.path.join(LLM_SIM_DIR, "data")
CONFIG_PATH = os.path.join(BUILD_DIR, "config.yaml")

def run_simulation(config_dict, label=""):
    print(f"\n==================================================")
    print(f"Running simulation: {label}")
    print(f"==================================================")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)
        
    cmd = 'wsl bash -c "cd LLMSimulator/build && ./run config.yaml"'
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if proc.returncode != 0:
        print(f"Error running simulation {label}:")
        print(proc.stderr[:1000])
        return None
        
    # Save raw simulation log
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    clean_label = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    log_path = os.path.join(log_dir, f"ramulator_{clean_label}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"=== Ramulator Fidelity Check Simulation Log: {label} ===\n\n")
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n--- STDERR ---\n" + proc.stderr)
    print(f"Saved run log: {log_path}")

    # Get latest CSV
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    latest_csv = max(csv_files, key=os.path.getmtime)
    print(f"Latest CSV: {latest_csv}")
    
    df = pd.read_csv(latest_csv)
    lat_us = df['latency'].mean() / 1000.0
    lat_ms = lat_us / 1000.0
    eng_mJ = df['total_energy'].mean() / 1e6
    exp_us = df['expert_ffn'].mean() / 1000.0 if 'expert_ffn' in df.columns else 0
    moe_dram_mJ = df['moe_dram'].mean() / 1e6 if 'moe_dram' in df.columns else 0
    
    return {
        'label': label,
        'latency_ms': round(lat_ms, 2),
        'latency_us': round(lat_us, 1),
        'expert_ffn_us': round(exp_us, 1),
        'energy_mJ': round(eng_mJ, 1),
        'moe_dram_mJ': round(moe_dram_mJ, 1),
        'csv_path': latest_csv
    }

def run_ramulator_check():
    """
    Exp 1: Ramulator 2.0 Check vs Analytic Model.
    Evaluates GPU and Duplex (GPU+Logic-PIM) under both:
    1. Analytic mode (use_ramulator: false)
    2. Ramulator 2.0 cycle-accurate mode (use_ramulator: true)
    """
    base_config = {
        'model': {'model_name': 'mixtral'},
        'system': {
            'gpu_gen': 'H100',
            'nvlink_gen': 4,
            'infiniband_gen': 400,
            'num_node': 1,
            'num_device': 4,
            'processor_type': 'GPU',
            'distribution': {'expert_tensor_degree': 1, 'none_expert_tensor_degree': 1},
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
        'serving': {'max_batch_size': 32, 'max_process_token': 0},
        'simulation': {
            'data': 'synthesis', 'input_len': 1024, 'output_len': 5,
            'precision_byte': 2, 'skewness': 0.0, 'iter': 3,
            'injection_rate': 0, 'exit_out_of_memory': False, 'mem_cap_limit': False
        },
        'log': {'print_log': True, 'export_gantt': False, 'output_directory': '../data/', 'gantt_directory': './gantt'}
    }

    # 1. GPU Analytic
    cfg_gpu_ana = yaml.safe_load(yaml.dump(base_config))
    cfg_gpu_ana['system']['processor_type'] = 'GPU'
    cfg_gpu_ana['system']['optimization']['use_ramulator'] = False
    res_gpu_ana = run_simulation(cfg_gpu_ana, "GPU (Analytic)")

    # 2. GPU Ramulator
    cfg_gpu_ram = yaml.safe_load(yaml.dump(base_config))
    cfg_gpu_ram['system']['processor_type'] = 'GPU'
    cfg_gpu_ram['system']['optimization']['use_ramulator'] = True
    res_gpu_ram = run_simulation(cfg_gpu_ram, "GPU (Ramulator 2.0)")

    # 3. Duplex Analytic
    cfg_dup_ana = yaml.safe_load(yaml.dump(base_config))
    cfg_dup_ana['system']['processor_type'] = 'GPU+LOGIC'
    cfg_dup_ana['system']['optimization']['use_ramulator'] = False
    res_dup_ana = run_simulation(cfg_dup_ana, "Duplex (Analytic)")

    # 4. Duplex Ramulator
    cfg_dup_ram = yaml.safe_load(yaml.dump(base_config))
    cfg_dup_ram['system']['processor_type'] = 'GPU+LOGIC'
    cfg_dup_ram['system']['optimization']['use_ramulator'] = True
    res_dup_ram = run_simulation(cfg_dup_ram, "Duplex (Ramulator 2.0)")

    # Build comparison summary table
    table = [
        {
            'Mode': 'GPU',
            'Step time, analytic': f"{res_gpu_ana['latency_ms']:.2f} ms",
            'Step time, Ramulator': f"{res_gpu_ram['latency_ms']:.2f} ms",
            'Latency Ratio': f"{(res_gpu_ram['latency_ms'] / res_gpu_ana['latency_ms']):.2f}x",
            'Energy/token, analytic': f"{res_gpu_ana['energy_mJ']:.0f} mJ",
            'Energy/token, Ramulator': f"{res_gpu_ram['energy_mJ']:.0f} mJ",
            'Energy Ratio': f"{(res_gpu_ram['energy_mJ'] / res_gpu_ana['energy_mJ']):.2f}x"
        },
        {
            'Mode': 'GPU + Logic-PIM',
            'Step time, analytic': f"{res_dup_ana['latency_ms']:.2f} ms",
            'Step time, Ramulator': f"{res_dup_ram['latency_ms']:.2f} ms",
            'Latency Ratio': f"{(res_dup_ram['latency_ms'] / res_dup_ana['latency_ms']):.2f}x",
            'Energy/token, analytic': f"{res_dup_ana['energy_mJ']:.0f} mJ",
            'Energy/token, Ramulator': f"{res_dup_ram['energy_mJ']:.0f} mJ",
            'Energy Ratio': f"{(res_dup_ram['energy_mJ'] / res_dup_ana['energy_mJ']):.2f}x"
        }
    ]

    df_table = pd.DataFrame(table)
    out_csv = os.path.abspath("mid_evaluation/results/ramulator_check.csv")
    df_table.to_csv(out_csv, index=False)
    
    speedup_ana = res_gpu_ana['latency_ms'] / res_dup_ana['latency_ms']
    speedup_ram = res_gpu_ram['latency_ms'] / res_dup_ram['latency_ms']

    print("\n" + "=" * 60)
    print("EXPERIMENT 1: RAMULATOR 2.0 CHECK SUMMARY")
    print("=" * 60)
    print(df_table.to_string(index=False))
    print("-" * 60)
    print(f"Duplex Speedup (Analytic):  {speedup_ana:.2f}x")
    print(f"Duplex Speedup (Ramulator): {speedup_ram:.2f}x")
    print(f"MoE DRAM Energy Duplex (Analytic):  {res_dup_ana['moe_dram_mJ']:.1f} mJ")
    print(f"MoE DRAM Energy Duplex (Ramulator): {res_dup_ram['moe_dram_mJ']:.1f} mJ (Ratio: {res_dup_ram['moe_dram_mJ']/res_dup_ana['moe_dram_mJ']:.2f}x)")
    print(f"\nSaved results to {out_csv}")

if __name__ == "__main__":
    run_ramulator_check()
