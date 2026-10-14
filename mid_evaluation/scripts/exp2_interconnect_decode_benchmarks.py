"""
exp2_interconnect_decode_benchmarks.py
Architectural Evaluation & Interconnect Scaling for Off-Chip MoE Inference.

Replicates and plots:
1. Decode-step latency breakdown (expert loading dominance across batch sizes 1-64)
2. Per-expert copy vs. compute time (log-scale comparison)
3. Interconnect bandwidth scaling (PCIe 4.0 to 2x NVLink-C2C) & prefetching upper bounds

Evaluates both:
- Mixtral 8x7B (FP16, 32 layers, 8 experts, k=2, 352 MB/expert)
- DeepSeek-V3 (FP8, 61 layers, 256 experts + 1 shared, 44 MB/expert, 8 routed + 1 shared)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'legend.fontsize': 10.5,
    'figure.titlesize': 14
})

def generate_figure2_expert_loading_dominance():
    """
    Plots stacked bar chart showing expert loading accounts for 96%-98%
    of decode step time across batch sizes 1 to 64.
    """
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    
    # PCIe 4.0 x16 (32 GB/s)
    # 1. DeepSeek-V3 (FP8)
    ds_hbm_base = [17.6, 17.6, 17.7, 18.2, 19.5, 23.0, 31.0]
    ds_active_exp = [9, 17, 31, 52, 85, 130, 180]
    ds_weight_per_exp_mb = 44.0
    ds_loads = [(act * 61 * ds_weight_per_exp_mb / 1000.0) / 32.0 * 1000.0 for act in ds_active_exp]
    
    # 2. Mixtral 8x7B (FP16)
    mix_hbm_base = [12.2, 12.3, 12.5, 13.0, 14.5, 18.0, 26.0]
    mix_active_exp = [2.0, 3.5, 5.47, 7.2, 7.92, 8.0, 8.0]
    mix_weight_per_exp_mb = 352.32
    mix_loads = [(act * 32 * mix_weight_per_exp_mb / 1000.0) / 32.0 * 1000.0 for act in mix_active_exp]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    width = 0.55
    x = np.arange(len(batch_sizes))

    # Subplot A: DeepSeek-V3
    axes[0].bar(x, ds_loads, width, label='expert load from off-chip DRAM (PCIe 4.0)', color='#4a77d4')
    axes[0].bar(x, ds_hbm_base, width, bottom=ds_loads, label='everything else (attention + shared + compute)', color='#53b378')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(batch_sizes)
    axes[0].set_xlabel('batch size (sequences decoded together)', fontweight='bold')
    axes[0].set_ylabel('time per decode step (ms)', fontweight='bold')
    axes[0].set_title('DeepSeek-V3 (FP8): Expert Loading Dominates Decode Step', fontweight='bold', pad=12)
    axes[0].grid(axis='y', linestyle='-', alpha=0.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].legend(loc='upper left', frameon=False)

    for i in range(len(batch_sizes)):
        tot = ds_loads[i] + ds_hbm_base[i]
        pct = (ds_loads[i] / tot) * 100.0
        axes[0].text(x[i], tot + 150, f"{pct:.1f}%", ha='center', fontsize=9.5, fontweight='bold', color='#2a4a8b')

    # Subplot B: Mixtral 8x7B
    axes[1].bar(x, mix_loads, width, label='expert load from off-chip DRAM (PCIe 4.0)', color='#4a77d4')
    axes[1].bar(x, mix_hbm_base, width, bottom=mix_loads, label='everything else (attention + compute)', color='#53b378')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(batch_sizes)
    axes[1].set_xlabel('batch size (sequences decoded together)', fontweight='bold')
    axes[1].set_ylabel('time per decode step (ms)', fontweight='bold')
    axes[1].set_title('Mixtral 8x7B (FP16): Expert Loading Dominates Decode Step', fontweight='bold', pad=12)
    axes[1].grid(axis='y', linestyle='-', alpha=0.3)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].legend(loc='upper left', frameon=False)

    for i in range(len(batch_sizes)):
        tot = mix_loads[i] + mix_hbm_base[i]
        pct = (mix_loads[i] / tot) * 100.0
        axes[1].text(x[i], tot + 60, f"{pct:.1f}%", ha='center', fontsize=9.5, fontweight='bold', color='#2a4a8b')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "expert_loading_dominates_decode.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")


def generate_figure3_expert_copy_vs_compute():
    """
    Plots per-expert copy vs. compute time in log-scale across batch sizes 1 to 64.
    """
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    
    # 1. DeepSeek-V3
    ds_copy_pcie4 = [1375.0] * len(batch_sizes)
    ds_copy_pcie5 = [687.5] * len(batch_sizes)
    ds_exec_hbm = [21.6] * len(batch_sizes)
    ds_compute = [0.141 * b for b in batch_sizes]
    
    # 2. Mixtral 8x7B
    mix_copy_pcie4 = [11010.0] * len(batch_sizes)
    mix_copy_pcie5 = [5505.0] * len(batch_sizes)
    mix_exec_hbm = [172.8] * len(batch_sizes)
    mix_compute = [1.13 * b for b in batch_sizes]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)

    # Subplot A: DeepSeek-V3
    axes[0].plot(batch_sizes, ds_copy_pcie4, 'o-', color='#3b629b', linewidth=2.2, markersize=7, label='load over PCIe 4.0 (32 GB/s)')
    axes[0].plot(batch_sizes, ds_copy_pcie5, 'o--', color='#6589bf', linewidth=2.2, markersize=7, label='load over PCIe 5.0 (64 GB/s)')
    axes[0].plot(batch_sizes, ds_exec_hbm, 's-', color='#d96b3f', linewidth=2.2, markersize=7, label='local HBM access on A100')
    axes[0].plot(batch_sizes, ds_compute, '^:', color='#e2936e', linewidth=2.2, markersize=7, label='A100 GPU compute (FP8/FP16)')
    axes[0].set_xscale('log', base=2)
    axes[0].set_yscale('log')
    axes[0].set_xticks(batch_sizes)
    axes[0].set_xticklabels(batch_sizes)
    axes[0].set_ylim(0.05, 5000)
    axes[0].set_xlabel('batch size (sequences decoded together)', fontweight='bold')
    axes[0].set_ylabel('time per expert (us, log scale)', fontweight='bold')
    axes[0].set_title('One DeepSeek-V3 expert (44 MB): copy vs compute', fontweight='bold', pad=12)
    axes[0].grid(True, which='both', linestyle='-', alpha=0.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].annotate('', xy=(1, 0.16), xytext=(1, 650), arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.5))
    axes[0].text(1.2, 10.0, '4875x longer to copy (PCIe 5)\nthan compute at B=1\n(32x vs HBM access)', va='center')
    axes[0].legend(loc='lower center', bbox_to_anchor=(0.5, -0.32), frameon=False, ncol=2)

    # Subplot B: Mixtral 8x7B
    axes[1].plot(batch_sizes, mix_copy_pcie4, 'o-', color='#3b629b', linewidth=2.2, markersize=7, label='load over PCIe 4.0 (32 GB/s)')
    axes[1].plot(batch_sizes, mix_copy_pcie5, 'o--', color='#6589bf', linewidth=2.2, markersize=7, label='load over PCIe 5.0 (64 GB/s)')
    axes[1].plot(batch_sizes, mix_exec_hbm, 's-', color='#d96b3f', linewidth=2.2, markersize=7, label='local HBM access on A100')
    axes[1].plot(batch_sizes, mix_compute, '^:', color='#e2936e', linewidth=2.2, markersize=7, label='A100 GPU compute (312 TFLOPS)')
    axes[1].set_xscale('log', base=2)
    axes[1].set_yscale('log')
    axes[1].set_xticks(batch_sizes)
    axes[1].set_xticklabels(batch_sizes)
    axes[1].set_ylim(0.5, 25000)
    axes[1].set_xlabel('batch size (sequences decoded together)', fontweight='bold')
    axes[1].set_ylabel('time per expert (us, log scale)', fontweight='bold')
    axes[1].set_title('One Mixtral 8x7B expert (352 MB): copy vs compute', fontweight='bold', pad=12)
    axes[1].grid(True, which='both', linestyle='-', alpha=0.3)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].annotate('', xy=(1, 1.3), xytext=(1, 5000), arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.5))
    axes[1].text(1.2, 80.0, '4875x longer to copy (PCIe 5)\nthan compute at B=1\n(32x vs HBM access)', va='center')
    axes[1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.32), frameon=False, ncol=2)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "expert_copy_vs_compute_logscale.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")


def generate_figure4_interconnect_scaling():
    """
    Plots horizontal bar chart across interconnects at batch size 1 with prefetch speedup ceiling.
    """
    links = [
        'PCIe 4.0 x16\n32 GB/s',
        'PCIe 5.0 x16\n64 GB/s',
        'PCIe 6.0 x16 / CXL\n128 GB/s',
        'NVLink-C2C\n450 GB/s',
        '2x NVLink-C2C\n900 GB/s'
    ]
    bws = [32.0, 64.0, 128.0, 450.0, 900.0]

    # 1. DeepSeek-V3
    ds_hbm_base = 17.6
    ds_total_weight_gb = 61 * (8 + 1) * 0.044  # 24.156 GB
    ds_loads = [(ds_total_weight_gb / bw) * 1000.0 for bw in bws]
    ds_totals = [load + ds_hbm_base for load in ds_loads]
    ds_speedups = [tot / max(load, ds_hbm_base) for tot, load in zip(ds_totals, ds_loads)]

    # 2. Mixtral 8x7B
    mix_hbm_base = 12.2
    mix_total_weight_gb = 32 * 2 * 0.35232  # 22.548 GB
    mix_loads = [(mix_total_weight_gb / bw) * 1000.0 for bw in bws]
    mix_totals = [load + mix_hbm_base for load in mix_loads]
    mix_speedups = [tot / max(load, mix_hbm_base) for tot, load in zip(mix_totals, mix_loads)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)
    y_pos = np.arange(len(links))
    height = 0.52

    # Subplot A: DeepSeek-V3
    axes[0].barh(y_pos, ds_loads, height, label='expert load (routed + shared)', color='#4a77d4')
    axes[0].barh(y_pos, [ds_hbm_base]*len(links), height, left=ds_loads, label='everything else', color='#53b378')
    axes[0].axvline(x=ds_hbm_base, color='black', linestyle='--', linewidth=1.2)
    axes[0].text(ds_hbm_base + 8, 4.3, f'all experts in HBM: {ds_hbm_base:.1f} ms', va='bottom')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(links)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('time per decode step (ms), batch 1', fontweight='bold')
    axes[0].set_title('DeepSeek-V3: Faster links shrink stall; prefetching helps when link is fast', fontweight='bold', pad=12)
    axes[0].set_xlim(0, 950)
    axes[0].grid(axis='x', linestyle='-', alpha=0.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    for i in range(len(links)):
        tot = ds_totals[i]
        sp = ds_speedups[i]
        axes[0].text(tot + 15, i, f'{tot:.0f} ms  (prefetch <= {sp:.2f}x)', va='center')
    axes[0].legend(loc='lower right', frameon=False)

    # Subplot B: Mixtral 8x7B
    axes[1].barh(y_pos, mix_loads, height, label='expert load (routed experts)', color='#4a77d4')
    axes[1].barh(y_pos, [mix_hbm_base]*len(links), height, left=mix_loads, label='everything else', color='#53b378')
    axes[1].axvline(x=mix_hbm_base, color='black', linestyle='--', linewidth=1.2)
    axes[1].text(mix_hbm_base + 8, 4.3, f'all experts in HBM: {mix_hbm_base:.1f} ms', va='bottom')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(links)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('time per decode step (ms), batch 1', fontweight='bold')
    axes[1].set_title('Mixtral 8x7B: Faster links shrink stall; prefetching helps when link is fast', fontweight='bold', pad=12)
    axes[1].set_xlim(0, 950)
    axes[1].grid(axis='x', linestyle='-', alpha=0.3)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    for i in range(len(links)):
        tot = mix_totals[i]
        sp = mix_speedups[i]
        axes[1].text(tot + 15, i, f'{tot:.0f} ms  (prefetch <= {sp:.2f}x)', va='center')
    axes[1].legend(loc='lower right', frameon=False)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "interconnect_scaling_prefetch_bound.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")


if __name__ == "__main__":
    generate_figure2_expert_loading_dominance()
    generate_figure3_expert_copy_vs_compute()
    generate_figure4_interconnect_scaling()
