"""
correlation_analysis.py
Statistical Characterization of Spatio-Temporal Expert Routing Patterns
using REAL inference traces from Hugging Face:
Model: mistralai/Mixtral-8x7B
Dataset: allenai/analysis_mixtral (Real inference on C4 corpus)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PROJECT_ROOT, "mid_evaluation", "data", "hf_mixtral_c4_routing.jsonl")
OUTPUT_PLOT_DIR = os.path.join(PROJECT_ROOT, "mid_evaluation", "plots")
OUTPUT_CSV_DIR = os.path.join(PROJECT_ROOT, "mid_evaluation", "results")

os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})

def load_real_hf_traces(data_path, max_tokens=10000):
    """
    Loads real Mixtral 8x7B routing decisions from Hugging Face C4 dataset.
    Returns:
      traces: list of dicts with:
        - 'seq_id': sequence index
        - 'token_id': token index
        - 'layer_id': layer index (0 to 31)
        - 'experts': [top_1, top_2]
    """
    print(f"Loading real Hugging Face Mixtral-8x7B routing traces from: {data_path}")
    traces = []
    total_tokens_loaded = 0
    
    with open(data_path, "r", encoding="utf-8") as f:
        for seq_id, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            exp_ids = record["exp_ids"] # shape: [num_tokens, 2, 32]
            
            for t_idx, token_data in enumerate(exp_ids):
                top1_per_layer = token_data[0] # length 32
                top2_per_layer = token_data[1] # length 32
                
                for layer_id in range(len(top1_per_layer)):
                    traces.append({
                        'seq_id': seq_id,
                        'token_id': t_idx,
                        'layer_id': layer_id,
                        'top1': top1_per_layer[layer_id],
                        'top2': top2_per_layer[layer_id]
                    })
                total_tokens_loaded += 1
                if total_tokens_loaded >= max_tokens:
                    break
            if total_tokens_loaded >= max_tokens:
                break
                
    print(f"Loaded {total_tokens_loaded:,} tokens across 32 layers ({len(traces):,} layer-routing points).")
    return traces

def calculate_gini(frequencies):
    """Calculate Gini coefficient of expert frequencies."""
    sorted_freqs = np.sort(frequencies)
    n = len(frequencies)
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * sorted_freqs)) / (n * np.sum(sorted_freqs)))

def calculate_shannon_entropy(frequencies):
    """Calculate Shannon entropy in bits."""
    probs = frequencies / np.sum(frequencies)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

def analyze_real_traces(traces, num_experts=8, num_layers=32):
    df = pd.DataFrame(traces)
    
    # 1. Expert Activation Frequency (Top-1 and Top-2)
    all_activations = np.concatenate([df['top1'].values, df['top2'].values])
    expert_counts = np.bincount(all_activations, minlength=num_experts)
    expert_freqs = expert_counts / len(all_activations)
    
    gini = calculate_gini(expert_freqs)
    entropy = calculate_shannon_entropy(expert_freqs)
    
    # Top 2 experts cumulative share
    sorted_indices = np.argsort(expert_freqs)[::-1]
    top2_share = float(expert_freqs[sorted_indices[:2]].sum())
    
    # 2. Spatial Correlation Across Adjacent Layers (L -> L+1)
    # Track transition for the same token and sequence across layers
    spatial_transitions = np.zeros((num_experts, num_experts))
    
    # Pivot df by (seq_id, token_id, layer_id)
    # Since tokens are processed sequentially, sort by seq_id, token_id, layer_id
    df_sorted = df.sort_values(by=['seq_id', 'token_id', 'layer_id'])
    
    # Filter for layer transitions L -> L+1
    df_current = df_sorted[df_sorted['layer_id'] < num_layers - 1].reset_index(drop=True)
    df_next = df_sorted[df_sorted['layer_id'] > 0].reset_index(drop=True)
    
    # Check that seq_id and token_id match
    mask = (df_current['seq_id'] == df_next['seq_id']) & (df_current['token_id'] == df_next['token_id'])
    curr_top1 = df_current.loc[mask, 'top1'].values
    next_top1 = df_next.loc[mask, 'top1'].values
    next_top2 = df_next.loc[mask, 'top2'].values
    
    for c, n in zip(curr_top1, next_top1):
        spatial_transitions[c, n] += 1
        
    spatial_matrix = spatial_transitions / spatial_transitions.sum(axis=1, keepdims=True)
    
    # Top-1 accuracy: if we predict argmax(spatial_matrix[curr]), does it match next_top1?
    best_top1_pred = np.argmax(spatial_matrix, axis=1)
    pred_top1_correct = (best_top1_pred[curr_top1] == next_top1).mean()
    
    # Top-2 accuracy: if we predict top-2 from spatial_matrix, does it cover next_top1 or next_top2?
    top2_preds = np.argsort(spatial_matrix, axis=1)[:, -2:] # shape [8, 2]
    top2_matches = 0
    for c, n1, n2 in zip(curr_top1, next_top1, next_top2):
        preds = top2_preds[c]
        if n1 in preds or n2 in preds:
            top2_matches += 1
    pred_top2_accuracy = top2_matches / len(curr_top1)
    
    # 3. Temporal Correlation Across Consecutive Decoding Steps (t -> t+1 at same layer)
    temporal_transitions = np.zeros((num_experts, num_experts))
    
    # For temporal, sort by seq_id, layer_id, token_id
    df_temporal = df.sort_values(by=['seq_id', 'layer_id', 'token_id']).reset_index(drop=True)
    df_t_curr = df_temporal.iloc[:-1].reset_index(drop=True)
    df_t_next = df_temporal.iloc[1:].reset_index(drop=True)
    
    t_mask = (df_t_curr['seq_id'] == df_t_next['seq_id']) & \
             (df_t_curr['layer_id'] == df_t_next['layer_id']) & \
             (df_t_next['token_id'] == df_t_curr['token_id'] + 1)
             
    t_curr_e = df_t_curr.loc[t_mask, 'top1'].values
    t_next_e = df_t_next.loc[t_mask, 'top1'].values
    
    for c, n in zip(t_curr_e, t_next_e):
        temporal_transitions[c, n] += 1
        
    temporal_matrix = temporal_transitions / temporal_transitions.sum(axis=1, keepdims=True)
    temporal_self_persistence = float(np.diag(temporal_matrix).mean())
    
    # 4. Pairwise Co-Activation Strength for (top1, top2)
    coact_matrix = np.zeros((num_experts, num_experts))
    for t1, t2 in zip(df['top1'].values, df['top2'].values):
        if t1 != t2:
            coact_matrix[t1, t2] += 1
            coact_matrix[t2, t1] += 1
            
    # Normalize by max
    coact_norm = coact_matrix / coact_matrix.max()
    
    return {
        'expert_freqs': expert_freqs,
        'gini': float(gini),
        'entropy': float(entropy),
        'top2_share': float(top2_share),
        'spatial_matrix': spatial_matrix,
        'top1_spatial_acc': float(pred_top1_correct),
        'top2_spatial_acc': float(pred_top2_accuracy),
        'temporal_matrix': temporal_matrix,
        'temporal_persistence': float(temporal_self_persistence),
        'coact_matrix': coact_norm,
        'num_samples': len(df)
    }

def plot_real_dashboard(stats, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)
    num_experts = 8
    experts = np.arange(num_experts)
    
    # (a) Expert Popularity
    freqs = stats['expert_freqs']
    bars = axes[0, 0].bar(experts, freqs * 100, color='#3b629b', edgecolor='black', alpha=0.85, width=0.6)
    axes[0, 0].axhline(y=(100.0/num_experts), color='red', linestyle='--', linewidth=1.5, label='Uniform Baseline (12.5%)')
    axes[0, 0].set_xlabel('Expert ID', fontweight='bold')
    axes[0, 0].set_ylabel('Activation Frequency (%)', fontweight='bold')
    axes[0, 0].set_title(f'(a) Real Mixtral-8x7B Expert Popularity (C4 Corpus)\nGini: {stats["gini"]:.4f} | Top-2 Share: {stats["top2_share"]*100:.1f}%', fontweight='bold', pad=10)
    axes[0, 0].set_xticks(experts)
    axes[0, 0].set_xticklabels([f'E{i}' for i in experts])
    axes[0, 0].set_ylim(0, max(freqs * 100) * 1.25)
    axes[0, 0].grid(axis='y', linestyle='-', alpha=0.3)
    axes[0, 0].spines['top'].set_visible(False)
    axes[0, 0].spines['right'].set_visible(False)
    axes[0, 0].legend(loc='upper right', frameon=False)
    
    for bar in bars:
        h = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{h:.1f}%', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
        
    # (b) Spatial Correlation Matrix (L -> L+1)
    im1 = axes[0, 1].imshow(stats['spatial_matrix'], cmap='Blues', aspect='auto', vmin=0, vmax=stats['spatial_matrix'].max())
    axes[0, 1].set_xlabel('Expert in Layer L+1', fontweight='bold')
    axes[0, 1].set_ylabel('Expert in Layer L', fontweight='bold')
    axes[0, 1].set_title(f'(b) Spatial Transition Probability P(E_L+1 | E_L)\nTop-1 Acc: {stats["top1_spatial_acc"]*100:.2f}% | Top-2 Acc: {stats["top2_spatial_acc"]*100:.2f}%', fontweight='bold', pad=10)
    axes[0, 1].set_xticks(experts)
    axes[0, 1].set_yticks(experts)
    axes[0, 1].set_xticklabels([f'E{i}' for i in experts])
    axes[0, 1].set_yticklabels([f'E{i}' for i in experts])
    cbar1 = fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cbar1.set_label('Transition Probability', rotation=270, labelpad=15)
    
    # (c) Temporal Correlation Matrix (t -> t+1)
    im2 = axes[1, 0].imshow(stats['temporal_matrix'], cmap='Greens', aspect='auto', vmin=0, vmax=stats['temporal_matrix'].max())
    axes[1, 0].set_xlabel('Expert at Token t+1', fontweight='bold')
    axes[1, 0].set_ylabel('Expert at Token t', fontweight='bold')
    axes[1, 0].set_title(f'(c) Temporal Transition Probability P(E_t+1 | E_t)\nMean Self-Persistence (Diagonal): {stats["temporal_persistence"]*100:.2f}%', fontweight='bold', pad=10)
    axes[1, 0].set_xticks(experts)
    axes[1, 0].set_yticks(experts)
    axes[1, 0].set_xticklabels([f'E{i}' for i in experts])
    axes[1, 0].set_yticklabels([f'E{i}' for i in experts])
    cbar2 = fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar2.set_label('Transition Probability', rotation=270, labelpad=15)
    
    # (d) Pairwise Co-Activation Strength
    im3 = axes[1, 1].imshow(stats['coact_matrix'], cmap='Oranges', aspect='auto', vmin=0, vmax=1.0)
    axes[1, 1].set_xlabel('Expert j', fontweight='bold')
    axes[1, 1].set_ylabel('Expert i', fontweight='bold')
    axes[1, 1].set_title('(d) Pairwise Co-Activation Strength (Top-2 Experts)\nNormalized Co-occurrence in Real Inference', fontweight='bold', pad=10)
    axes[1, 1].set_xticks(experts)
    axes[1, 1].set_yticks(experts)
    axes[1, 1].set_xticklabels([f'E{i}' for i in experts])
    axes[1, 1].set_yticklabels([f'E{i}' for i in experts])
    cbar3 = fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar3.set_label('Co-activation Strength', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated Real Dashboard: {out_path}")

if __name__ == "__main__":
    traces = load_real_hf_traces(DATA_PATH, max_tokens=10000)
    stats = analyze_real_traces(traces)
    
    print("\n" + "=" * 60)
    print("REAL HUGGING FACE MIXTRAL-8x7B ROUTING STATISTICS (C4 Corpus)")
    print("=" * 60)
    print(f"Total layer-routing points analyzed: {stats['num_samples']:,}")
    print(f"Expert Popularity Skew (Gini):       {stats['gini']:.4f}")
    print(f"Shannon Entropy:                     {stats['entropy']:.4f} bits (Uniform max: 3.0000 bits)")
    print(f"Top-2 Experts Cumulative Share:      {stats['top2_share']*100:.2f}%")
    print(f"Top-1 Spatial Prediction Accuracy:   {stats['top1_spatial_acc']*100:.2f}% (Baseline random: 12.5%)")
    print(f"Top-2 Spatial Prediction Accuracy:   {stats['top2_spatial_acc']*100:.2f}% (Baseline random: 25.0%)")
    print(f"Temporal Self-Persistence (Diag):    {stats['temporal_persistence']*100:.2f}% (Baseline random: 12.5%)")
    
    out_csv = os.path.join(OUTPUT_CSV_DIR, "correlation_statistics.csv")
    csv_df = pd.DataFrame([{
        'metric': 'gini_coefficient', 'value': round(stats['gini'], 4), 'source': 'HuggingFace_Mixtral_8x7B_C4'
    }, {
        'metric': 'shannon_entropy_bits', 'value': round(stats['entropy'], 4), 'source': 'HuggingFace_Mixtral_8x7B_C4'
    }, {
        'metric': 'top2_expert_share_pct', 'value': round(stats['top2_share'] * 100, 2), 'source': 'HuggingFace_Mixtral_8x7B_C4'
    }, {
        'metric': 'top1_spatial_accuracy_pct', 'value': round(stats['top1_spatial_acc'] * 100, 2), 'source': 'HuggingFace_Mixtral_8x7B_C4'
    }, {
        'metric': 'top2_spatial_accuracy_pct', 'value': round(stats['top2_spatial_acc'] * 100, 2), 'source': 'HuggingFace_Mixtral_8x7B_C4'
    }, {
        'metric': 'temporal_self_persistence_pct', 'value': round(stats['temporal_persistence'] * 100, 2), 'source': 'HuggingFace_Mixtral_8x7B_C4'
    }, {
        'metric': 'total_routing_samples', 'value': stats['num_samples'], 'source': 'HuggingFace_Mixtral_8x7B_C4'
    }])
    csv_df.to_csv(out_csv, index=False)
    print(f"Saved real correlation statistics to: {out_csv}")
    
    out_plot = os.path.join(OUTPUT_PLOT_DIR, "spatio_temporal_correlation_dashboard.png")
    plot_real_dashboard(stats, out_plot)
