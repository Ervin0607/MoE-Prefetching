import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_offchip_moe_study():
    """
    Deliverable 2 & 3:
    Study a simplified single-GPU MoE architecture.
    Multi-GPU expert sharding and Logic-PIM disabled.
    All expert weights placed in off-chip DRAM (CPU memory).
    Fast local memory: A100-class 80 GB HBM (and H100 80GB HBM).
    Measure and report expert loading time vs expert execution time.
    """
    print("=" * 60)
    print("Running Single-GPU Off-Chip DRAM MoE Modeling Study")
    print("=" * 60)
    
    # Model Architecture: Mixtral 8x7B
    model_name = "Mixtral 8x7B"
    num_layers = 32
    hidden_dim = 4096
    intermediate_dim = 14336
    num_experts = 8
    top_k = 2
    precision_bytes = 2 # FP16
    
    # Expert parameter calculations
    # 3 linear projections: gate_proj (H x I), up_proj (H x I), down_proj (I x H)
    params_per_expert = 3 * hidden_dim * intermediate_dim
    expert_size_bytes = params_per_expert * precision_bytes
    expert_size_mb = expert_size_bytes / (1024 * 1024)
    total_moe_size_gb = (expert_size_bytes * num_experts * num_layers) / (1024**3)
    
    print(f"Model: {model_name}")
    print(f"Parameters per expert: {params_per_expert:,} ({params_per_expert/1e6:.2f} M)")
    print(f"Weight size per expert (FP16): {expert_size_mb:.2f} MB")
    print(f"Total MoE parameters across {num_layers} layers: {num_layers * num_experts * params_per_expert / 1e9:.2f} B")
    print(f"Total MoE weight footprint: {total_moe_size_gb:.2f} GB (Exceeds single 80GB GPU HBM!)")
    
    # Hardware Specifications
    hardware_configs = {
        'A100-80GB': {
            'peak_flops_fp16': 312e12,       # 312 TFLOPS
            'hbm_bandwidth': 2039e9,          # 2039 GB/s
            'hbm_capacity_gb': 80,
        },
        'H100-80GB': {
            'peak_flops_fp16': 989.4e12,     # 989.4 TFLOPS
            'hbm_bandwidth': 3352e9,          # 3352 GB/s
            'hbm_capacity_gb': 80,
        }
    }
    
    # Off-chip interconnects (Host CPU DRAM <-> GPU HBM)
    interconnects = {
        'PCIe Gen4 x16': 32e9,   # 32 GB/s
        'PCIe Gen5 x16': 64e9,   # 64 GB/s
        'CXL 2.0 / 3.0': 128e9   # 128 GB/s
    }
    
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    results = []
    
    for hw_name, hw in hardware_configs.items():
        peak_flops = hw['peak_flops_fp16']
        hbm_bw = hw['hbm_bandwidth']
        
        for ic_name, ic_bw in interconnects.items():
            for bs in batch_sizes:
                # Number of active experts in a layer for batch size bs with top-k=2
                # Exact expected value: Nex * (1 - (1 - k/Nex)^bs)
                prob_expert_not_picked = (1.0 - (top_k / num_experts)) ** bs
                expected_active_experts = num_experts * (1.0 - prob_expert_not_picked)
                
                # Average tokens per active expert
                tokens_per_active_expert = (bs * top_k) / expected_active_experts
                
                # 1. Computation Time on GPU for one expert (microseconds)
                # FLOPs = 2 * 3 * H * I * tokens_per_active_expert
                flops_per_expert = 2 * params_per_expert * tokens_per_active_expert
                compute_time_us = (flops_per_expert / peak_flops) * 1e6
                
                # 2. Local HBM Read Time if resident in HBM (microseconds)
                hbm_read_time_us = (expert_size_bytes / hbm_bw) * 1e6
                
                # 3. Off-Chip DRAM Loading Time per expert (microseconds)
                load_time_us = (expert_size_bytes / ic_bw) * 1e6
                
                # Total time per expert (on-demand un-overlapped execution)
                # In on-demand: GPU stalls while loading expert from off-chip DRAM, then computes
                execution_time_gpu_us = max(compute_time_us, hbm_read_time_us)
                total_on_demand_us = load_time_us + execution_time_gpu_us
                
                # Stall ratio
                exposed_stall_us = load_time_us
                stall_percentage = (exposed_stall_us / total_on_demand_us) * 100.0
                ratio_load_to_compute = load_time_us / compute_time_us
                
                # Arithmetic intensity Op/B for an active expert
                # FLOPs / total_bytes_accessed
                # In HBM: weights (expert_size_bytes) + activations
                op_b = flops_per_expert / expert_size_bytes
                
                results.append({
                    'Hardware': hw_name,
                    'Interconnect': ic_name,
                    'Interconnect_BW_GBs': ic_bw / 1e9,
                    'Batch_Size': bs,
                    'Active_Experts': round(expected_active_experts, 2),
                    'Tokens_Per_Active_Expert': round(tokens_per_active_expert, 2),
                    'Op_B': round(op_b, 2),
                    'Expert_Compute_Time_us': round(compute_time_us, 2),
                    'Expert_HBM_Read_Time_us': round(hbm_read_time_us, 2),
                    'Expert_Loading_Time_us': round(load_time_us, 2),
                    'Total_Expert_Time_us': round(total_on_demand_us, 2),
                    'Exposed_Stall_us': round(exposed_stall_us, 2),
                    'Stall_Percentage': round(stall_percentage, 2),
                    'Load_to_Compute_Ratio': round(ratio_load_to_compute, 1)
                })
                
    df = pd.DataFrame(results)
    out_csv = "mid_evaluation/results/offchip_dram_expert_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved metrics to {out_csv}")
    
    # Print sample comparison table for A100 with PCIe Gen5
    print("\n--- Summary for A100-80GB with PCIe Gen5 (64 GB/s) ---")
    sub_df = df[(df['Hardware'] == 'A100-80GB') & (df['Interconnect'] == 'PCIe Gen5 x16')]
    cols_to_show = ['Batch_Size', 'Active_Experts', 'Op_B', 'Expert_Compute_Time_us', 
                    'Expert_Loading_Time_us', 'Load_to_Compute_Ratio', 'Stall_Percentage']
    print(sub_df[cols_to_show].to_string(index=False))
    
    # Generate Visualizations
    generate_deliverable2_plots(df)
    return df

def generate_deliverable2_plots(df):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Plot 1: Loading Time vs Execution Time across batch sizes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    sub = df[(df['Hardware'] == 'A100-80GB') & (df['Interconnect'] == 'PCIe Gen5 x16')]
    
    ax1.plot(sub['Batch_Size'], sub['Expert_Loading_Time_us'], 'r-o', linewidth=2.5, markersize=8, label='Expert Loading Time (from Off-Chip DRAM)')
    ax1.plot(sub['Batch_Size'], sub['Expert_HBM_Read_Time_us'], 'g--s', linewidth=2, markersize=7, label='Local HBM Access Time')
    ax1.plot(sub['Batch_Size'], sub['Expert_Compute_Time_us'], 'b-^', linewidth=2.5, markersize=8, label='Expert GPU Compute Time')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Batch Size (Tokens)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Latency per Expert (μs, Log Scale)', fontsize=12, fontweight='bold')
    ax1.set_title('A100: Expert Loading vs Compute Time (PCIe Gen5)', fontsize=13, fontweight='bold')
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(fontsize=10, loc='center left')
    
    # Add annotation for gap
    ratio_b1 = sub[sub['Batch_Size'] == 1]['Load_to_Compute_Ratio'].values[0]
    ax1.annotate(f'{ratio_b1:.0f}x Gap\nat Batch=1', xy=(1, 5505), xytext=(1.5, 500),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))
    
    # Plot 2: Breakdown of Exposed Memory Stalls vs Interconnect Bandwidth
    for ic in ['PCIe Gen4 x16', 'PCIe Gen5 x16', 'CXL 2.0 / 3.0']:
        sub_ic = df[(df['Hardware'] == 'A100-80GB') & (df['Interconnect'] == ic)]
        ax2.plot(sub_ic['Batch_Size'], sub_ic['Stall_Percentage'], marker='o', linewidth=2, label=f'{ic} ({sub_ic["Interconnect_BW_GBs"].values[0]:.0f} GB/s)')
        
    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Batch Size (Tokens)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Exposed Memory Stall (% of Total Time)', fontsize=12, fontweight='bold')
    ax2.set_title('Exposed Memory Stall Ratio vs Batch Size', fontsize=13, fontweight='bold')
    ax2.set_ylim(85, 100.5)
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend(fontsize=10, loc='lower left')
    
    plt.tight_layout()
    p1 = "mid_evaluation/plots/moe_offchip_loading_vs_compute.png"
    plt.savefig(p1)
    plt.close()
    print(f"Generated plot: {p1}")

if __name__ == "__main__":
    run_offchip_moe_study()
