import os
import subprocess
import yaml
import pandas as pd
import glob
import re

LLM_SIM_DIR = os.path.abspath("LLMSimulator")
BUILD_DIR = os.path.join(LLM_SIM_DIR, "build")
DATA_DIR = os.path.join(LLM_SIM_DIR, "data")
CONFIG_PATH = os.path.join(BUILD_DIR, "config.yaml")

def run_simulation(config_dict, run_label="sim"):
    print(f"\n==================================================")
    print(f"Running simulation: {run_label}")
    print(f"==================================================")
    
    # Write config
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)
        
    # Convert windows path to wsl path for running
    cmd = 'wsl bash -c "cd LLMSimulator/build && ./run config.yaml"'
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if proc.returncode != 0:
        print(f"Error running simulation {run_label}:")
        print(proc.stderr[:1000])
        return None
        
    print(f"Simulation {run_label} completed successfully.")
    
    # Save raw simulation log
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    clean_label = re.sub(r'[^a-zA-Z0-9_-]', '_', run_label.lower())
    log_path = os.path.join(log_dir, f"{clean_label}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"=== Simulation Log: {run_label} ===\n\n")
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n--- STDERR ---\n" + proc.stderr)
    print(f"Saved run log: {log_path}")

    # Find generated CSV in data directory
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print("No CSV generated in data directory.")
        return None, proc.stdout
        
    latest_csv = max(csv_files, key=os.path.getmtime)
    print(f"Latest CSV generated: {latest_csv}")
    return latest_csv, proc.stdout

def extract_summary_metrics(csv_path):
    """Parse metrics directly from the simulation output CSV."""
    metrics = {}
    if csv_path is None or not os.path.exists(csv_path):
        print(f"Warning: CSV path not found: {csv_path}")
        return metrics
    
    df = pd.read_csv(csv_path)
    
    # Latency per token (nanoseconds -> microseconds)
    if 'latency' in df.columns:
        metrics['per_token_latency_us'] = round(df['latency'].mean() / 1000.0, 1)
        
    # MoE Expert FFN latency per token (nanoseconds -> microseconds)
    if 'expert_ffn' in df.columns:
        metrics['expert_ffn_latency_us'] = round(df['expert_ffn'].mean() / 1000.0, 1)
    
    # End timestamp: last row's cumulative 'time' column, in nanoseconds -> convert to microseconds
    if 'time' in df.columns:
        metrics['end_timestamp_us'] = round(df['time'].iloc[-1] / 1000.0, 1)
    
    # Total energy: cluster.cpp outputs energy_nJ (nanoJoules)
    # Total energy across sample iterations in Joules (nJ / 1e9)
    if 'total_energy' in df.columns:
        metrics['total_energy_J'] = round(df['total_energy'].sum() / 1e9, 4)
        metrics['per_token_energy_mJ'] = round(df['total_energy'].mean() / 1e6, 2)
    
    return metrics

def run_duplex_reproduction():
    """
    Reproduce the baseline GPU vs Duplex comparison from the MICRO 2024 paper.
    
    Three configurations are tested:
    1. Baseline GPU (4x H100): Standard GPU-only processing
    2. Duplex (GPU + Logic-PIM): Layer-wise engine selection, no co-processing
    3. Duplex + PE (Co-processing): Expert and attention co-processing enabled
    
    NOTE: For Mixtral on 4 devices with expert parallelism (the default), each
    device processes only 2 experts. The paper (Section VII-A) notes that 
    co-processing (Duplex+PE) shows only ~1.04x average improvement over Duplex
    because few experts per device limits the benefit. The full benefit requires
    tensor parallelism for MoE layers (Duplex+PE+ET), which increases experts
    per device. With the current config, Duplex and Duplex+PE may produce
    nearly identical or identical results.
    """
    base_config = {
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
    
    results = []
    
    # 1. Baseline GPU
    cfg_gpu = yaml.safe_load(yaml.dump(base_config))
    cfg_gpu['system']['processor_type'] = 'GPU'
    cfg_gpu['system']['optimization']['parallel_execution'] = False
    result = run_simulation(cfg_gpu, "Baseline GPU (H100)")
    if result is not None:
        csv_gpu = result[0] if isinstance(result, tuple) else result
        m_gpu = extract_summary_metrics(csv_gpu)
        m_gpu['Config'] = 'Baseline GPU (4x H100)'
        m_gpu['csv_path'] = csv_gpu
        results.append(m_gpu)
    
    # 2. Duplex (GPU + Logic-PIM)
    cfg_duplex = yaml.safe_load(yaml.dump(base_config))
    cfg_duplex['system']['processor_type'] = 'GPU+LOGIC'
    cfg_duplex['system']['optimization']['parallel_execution'] = False
    result = run_simulation(cfg_duplex, "Duplex (GPU + Logic-PIM)")
    if result is not None:
        csv_duplex = result[0] if isinstance(result, tuple) else result
        m_duplex = extract_summary_metrics(csv_duplex)
        m_duplex['Config'] = 'Duplex (GPU + Logic-PIM)'
        m_duplex['csv_path'] = csv_duplex
        results.append(m_duplex)
    
    # 3. Duplex + PE (Co-processing)
    cfg_duplex_pe = yaml.safe_load(yaml.dump(base_config))
    cfg_duplex_pe['system']['processor_type'] = 'GPU+LOGIC'
    cfg_duplex_pe['system']['optimization']['parallel_execution'] = True
    result = run_simulation(cfg_duplex_pe, "Duplex + PE (Co-processing)")
    if result is not None:
        csv_pe = result[0] if isinstance(result, tuple) else result
        m_pe = extract_summary_metrics(csv_pe)
        m_pe['Config'] = 'Duplex + PE (Co-processing)'
        m_pe['csv_path'] = csv_pe
        results.append(m_pe)
    
    df = pd.DataFrame(results)
    
    # Compute speedup relative to Baseline GPU
    # Note: Duplex and Duplex+PE may produce identical results when expert parallelism
    # (not tensor parallelism) is used for MoE layers, because parallel_execution (co-processing)
    # only shows benefit with tensor parallelism for MoE experts (the paper calls this Duplex+PE+ET).
    # With the default expert parallelism in Mixtral on 4 devices (only 2 experts per device),
    # co-processing is minimal. Reference: Section V-B and VII-A of the Duplex paper.
    baseline_time = None
    if not df.empty and 'per_token_latency_us' in df.columns:
        baseline_rows = df[df['Config'].str.contains('Baseline', case=False, na=False)]
        if not baseline_rows.empty and pd.notna(baseline_rows['per_token_latency_us'].iloc[0]):
            baseline_time = baseline_rows['per_token_latency_us'].iloc[0]
            if baseline_time > 0:
                df['speedup'] = (baseline_time / df['per_token_latency_us']).round(3)
    
    # Reorder columns cleanly
    col_order = ['Config', 'per_token_latency_us', 'expert_ffn_latency_us', 'end_timestamp_us', 
                 'per_token_energy_mJ', 'total_energy_J', 'speedup', 'csv_path']
    existing_cols = [c for c in col_order if c in df.columns] + [c for c in df.columns if c not in col_order]
    df = df[existing_cols]

    out_summary = os.path.abspath("mid_evaluation/results/duplex_reproduction_summary.csv")
    os.makedirs(os.path.dirname(out_summary), exist_ok=True)
    df.to_csv(out_summary, index=False)
    print("\n================ Reproduction Summary ================")
    print(df.to_string())
    if baseline_time is not None and 'speedup' in df.columns:
        print("\n================ Speedup Analysis ================")
        for _, row in df.iterrows():
            config_name = row.get('Config', 'Unknown')
            speedup_val = row.get('speedup', 'N/A')
            token_lat = row.get('per_token_latency_us', 'N/A')
            exp_lat = row.get('expert_ffn_latency_us', 'N/A')
            print(f"  {config_name}: {token_lat} us/token (Expert FFN: {exp_lat} us) -> Speedup: {speedup_val}x")
    print(f"\nSaved summary to {out_summary}")

if __name__ == "__main__":
    run_duplex_reproduction()
