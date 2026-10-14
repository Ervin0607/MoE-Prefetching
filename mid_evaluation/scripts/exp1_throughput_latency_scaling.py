import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.abspath("mid_evaluation/plots")
RESULTS_DIR = os.path.abspath("mid_evaluation/results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

def build_comparison_tables():
    # Table 1: Summary Performance Data across Architecture Configurations
    perf_data = [
        {
            'Configuration': '8x B100 System (Batch 32)',
            'Variant': 'Baseline GPU',
            'TBT_us': 4137.8,
            'TBT_ms': 4.138,
            'Batch_Size': 32,
            'Throughput_tokens_s': 7733.6,
            'Throughput_Speedup': '1.00x',
            'Energy_per_token_mJ': 391.71
        },
        {
            'Configuration': '8x B100 System (Batch 32)',
            'Variant': 'Duplex (PE)',
            'TBT_us': 1623.5,
            'TBT_ms': 1.624,
            'Batch_Size': 32,
            'Throughput_tokens_s': 19710.5,
            'Throughput_Speedup': '2.55x',
            'Energy_per_token_mJ': 416.83
        },
        {
            'Configuration': '8x B100 System (Batch 32)',
            'Variant': 'Duplex (PE + ET)',
            'TBT_us': 1524.7,
            'TBT_ms': 1.525,
            'Batch_Size': 32,
            'Throughput_tokens_s': 20987.7,
            'Throughput_Speedup': '2.71x',
            'Energy_per_token_mJ': 413.22
        },
        {
            'Configuration': '8x B100 System (Batch 32)',
            'Variant': 'Duplex (Ideal PIM Ceiling)',
            'TBT_us': 460.89,
            'TBT_ms': 0.461,
            'Batch_Size': 32,
            'Throughput_tokens_s': 69430.9,
            'Throughput_Speedup': '8.98x',
            'Energy_per_token_mJ': 38.79
        },
        # Mixtral 8x7B (4x H100) Workload
        {
            'Configuration': '4x H100 Mixtral 8x7B (Analytic)',
            'Variant': 'Baseline GPU',
            'TBT_us': 7975.6,
            'TBT_ms': 7.976,
            'Batch_Size': 32,
            'Throughput_tokens_s': 4012.2,
            'Throughput_Speedup': '1.00x',
            'Energy_per_token_mJ': 3226.6
        },
        {
            'Configuration': '4x H100 Mixtral 8x7B (Analytic)',
            'Variant': 'Duplex (GPU + Logic-PIM)',
            'TBT_us': 3111.1,
            'TBT_ms': 3.111,
            'Batch_Size': 32,
            'Throughput_tokens_s': 10285.8,
            'Throughput_Speedup': '2.56x',
            'Energy_per_token_mJ': 3420.8
        },
        {
            'Configuration': '4x H100 Mixtral 8x7B (Ramulator)',
            'Variant': 'Baseline GPU',
            'TBT_us': 10040.0,
            'TBT_ms': 10.040,
            'Batch_Size': 32,
            'Throughput_tokens_s': 3187.3,
            'Throughput_Speedup': '1.00x',
            'Energy_per_token_mJ': 3256.0
        },
        {
            'Configuration': '4x H100 Mixtral 8x7B (Ramulator)',
            'Variant': 'Duplex (GPU + Logic-PIM)',
            'TBT_us': 3840.0,
            'TBT_ms': 3.840,
            'Batch_Size': 32,
            'Throughput_tokens_s': 8333.3,
            'Throughput_Speedup': '2.61x',
            'Energy_per_token_mJ': 1978.0
        }
    ]
    df_perf = pd.DataFrame(perf_data)
    out_perf = os.path.join(RESULTS_DIR, "system_performance_comparison.csv")
    df_perf.to_csv(out_perf, index=False)
    print(f"Saved performance comparison to {out_perf}")
    
    # Table 2: E2E Latency Estimation (Lin = 1024, Lout = 1024)
    # E2E = T2FT + (Lout - 1) * TBT
    e2e_data = [
        {'System_Config': '8x B100 System', 'Variant': 'Baseline GPU', 'TBT_ms': 4.138, 'T2FT_ms': 105.2, 'E2E_ms': 4338.4, 'E2E_s': 4.34, 'E2E_Speedup': '1.00x'},
        {'System_Config': '8x B100 System', 'Variant': 'Duplex (PE)', 'TBT_ms': 1.624, 'T2FT_ms': 105.2, 'E2E_ms': 1766.2, 'E2E_s': 1.77, 'E2E_Speedup': '2.46x'},
        {'System_Config': '8x B100 System', 'Variant': 'Duplex (PE + ET)', 'TBT_ms': 1.525, 'T2FT_ms': 105.2, 'E2E_ms': 1665.3, 'E2E_s': 1.67, 'E2E_Speedup': '2.60x'},
        # Mixtral 8x7B Primary Numbers (T2FT ~ 185.4 ms)
        {'System_Config': '4x H100 Mixtral 8x7B (Analytic)', 'Variant': 'Baseline GPU', 'TBT_ms': 7.976, 'T2FT_ms': 185.4, 'E2E_ms': 8344.9, 'E2E_s': 8.34, 'E2E_Speedup': '1.00x'},
        {'System_Config': '4x H100 Mixtral 8x7B (Analytic)', 'Variant': 'Duplex (Logic-PIM)', 'TBT_ms': 3.111, 'T2FT_ms': 185.4, 'E2E_ms': 3368.0, 'E2E_s': 3.37, 'E2E_Speedup': '2.48x'},
        {'System_Config': '4x H100 Mixtral 8x7B (Ramulator)', 'Variant': 'Baseline GPU', 'TBT_ms': 10.040, 'T2FT_ms': 225.0, 'E2E_ms': 10496.0, 'E2E_s': 10.50, 'E2E_Speedup': '1.00x'},
        {'System_Config': '4x H100 Mixtral 8x7B (Ramulator)', 'Variant': 'Duplex (Logic-PIM)', 'TBT_ms': 3.840, 'T2FT_ms': 225.0, 'E2E_ms': 4153.3, 'E2E_s': 4.15, 'E2E_Speedup': '2.53x'}
    ]
    df_e2e = pd.DataFrame(e2e_data)
    out_e2e = os.path.join(RESULTS_DIR, "e2e_latency_comparison.csv")
    df_e2e.to_csv(out_e2e, index=False)
    print(f"Saved E2E comparison to {out_e2e}")
    
    return df_perf, df_e2e

def plot_fig11_throughput():
    """Recreate Fig. 11 Slice: Throughput Comparison (tokens/s)"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)
    
    # Subplot 1: 8x B100 System Configuration
    variants_p = ['Baseline GPU', 'Duplex (PE)', 'Duplex (PE + ET)', 'Duplex (Ideal PIM)']
    throughputs_p = [7733.6, 19710.5, 20987.7, 69430.9]
    speedups_p = ['1.00x', '2.55x', '2.71x', '8.98x']
    colors_p = ['#777777', '#4a77d4', '#2c4ba1', '#e67e22']
    
    y_pos = np.arange(len(variants_p))
    bars1 = axes[0].barh(y_pos, throughputs_p, color=colors_p, height=0.55, edgecolor='black', linewidth=0.8)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(variants_p, fontsize=10, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Throughput (tokens/s)', fontsize=11, fontweight='bold')
    axes[0].set_title('8x B100 System: Throughput Comparison (Batch 32)', fontsize=12, fontweight='bold', pad=12)
    axes[0].set_xlim(0, 85000)
    axes[0].grid(axis='x', linestyle='--', alpha=0.4)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    for i, bar in enumerate(bars1):
        w = bar.get_width()
        axes[0].text(w + 1200, i, f"{w:,.1f} tokens/s  ({speedups_p[i]})", va='center', fontsize=9.5, fontweight='bold')
        
    # Subplot 2: 4x H100 Mixtral 8x7B Workload
    variants_m = ['Baseline GPU (Analytic)', 'Duplex (Analytic)', 'Baseline GPU (Ramulator)', 'Duplex (Ramulator)']
    throughputs_m = [4012.2, 10285.8, 3187.3, 8333.3]
    speedups_m = ['1.00x', '2.56x', '1.00x', '2.61x']
    colors_m = ['#777777', '#4a77d4', '#999999', '#2c4ba1']
    
    y_pos_m = np.arange(len(variants_m))
    bars2 = axes[1].barh(y_pos_m, throughputs_m, color=colors_m, height=0.55, edgecolor='black', linewidth=0.8)
    axes[1].set_yticks(y_pos_m)
    axes[1].set_yticklabels(variants_m, fontsize=10, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Throughput (tokens/s)', fontsize=11, fontweight='bold')
    axes[1].set_title('4x H100 Mixtral 8x7B: Throughput Comparison (Batch 32)', fontsize=12, fontweight='bold', pad=12)
    axes[1].set_xlim(0, 14000)
    axes[1].grid(axis='x', linestyle='--', alpha=0.4)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    for i, bar in enumerate(bars2):
        w = bar.get_width()
        axes[1].text(w + 300, i, f"{w:,.1f} tokens/s  ({speedups_m[i]})", va='center', fontsize=9.5, fontweight='bold')
        
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig11_throughput_comparison.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated Figure 11: {out_path}")

def plot_fig12_latency():
    """Recreate Fig. 12 Slice: Time Between Tokens (TBT) & E2E Latency"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)
    
    # 1. TBT Comparison
    variants = ['Baseline GPU', 'Duplex (PE)', 'Duplex (PE + ET)', 'Duplex (Ideal PIM)']
    tbt_ms = [4.138, 1.624, 1.525, 0.461]
    colors = ['#d9534f', '#337ab7', '#2c4ba1', '#5cb85c']
    
    y_pos = np.arange(len(variants))
    bars1 = ax1.barh(y_pos, tbt_ms, color=colors, height=0.55, edgecolor='black', linewidth=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(variants, fontsize=10, fontweight='bold')
    ax1.invert_yaxis()
    ax1.set_xlabel('TBT (Decode Step Execution Time, ms)', fontsize=11, fontweight='bold')
    ax1.set_title('1. Time Between Tokens (TBT) Comparison', fontsize=12, fontweight='bold', pad=12)
    ax1.set_xlim(0, 5.2)
    ax1.grid(axis='x', linestyle='--', alpha=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    for i, bar in enumerate(bars1):
        w = bar.get_width()
        ax1.text(w + 0.1, i, f"{w:.3f} ms", va='center', fontsize=10, fontweight='bold')
        
    # 2. E2E Latency Comparison (Lin=1024, Lout=1024)
    e2e_variants = ['Baseline GPU', 'Duplex (PE)', 'Duplex (PE + ET)']
    t2ft = [105.2, 105.2, 105.2]
    decode_tot = [1023 * 4.138, 1023 * 1.624, 1023 * 1.525]
    speedups = ['1.00x', '2.46x', '2.60x']
    
    y_pos_e2e = np.arange(len(e2e_variants))
    ax2.barh(y_pos_e2e, [t/1000.0 for t in t2ft], height=0.55, label='Prefill (T2FT ~105 ms)', color='#f0ad4e', edgecolor='black', linewidth=0.8)
    ax2.barh(y_pos_e2e, [d/1000.0 for d in decode_tot], height=0.55, left=[t/1000.0 for t in t2ft], label='Decode (1023 x TBT)', color='#337ab7', edgecolor='black', linewidth=0.8)
    
    ax2.set_yticks(y_pos_e2e)
    ax2.set_yticklabels(e2e_variants, fontsize=10, fontweight='bold')
    ax2.invert_yaxis()
    ax2.set_xlabel('End-to-End Latency (Seconds, Lout=1024)', fontsize=11, fontweight='bold')
    ax2.set_title('2. Estimated E2E Latency (Lin=1024, Lout=1024)', fontsize=12, fontweight='bold', pad=12)
    ax2.set_xlim(0, 5.5)
    ax2.grid(axis='x', linestyle='--', alpha=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(loc='lower right', frameon=False, fontsize=9.5)
    
    for i in range(len(e2e_variants)):
        tot_s = (t2ft[i] + decode_tot[i]) / 1000.0
        ax2.text(tot_s + 0.12, i, f"{tot_s:.2f} s  ({speedups[i]})", va='center', fontsize=9.5, fontweight='bold')
        
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig12_latency_comparison.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated Figure 12: {out_path}")

if __name__ == "__main__":
    build_comparison_tables()
    plot_fig11_throughput()
    plot_fig12_latency()
