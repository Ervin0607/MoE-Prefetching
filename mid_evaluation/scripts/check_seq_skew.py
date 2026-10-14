import json
import numpy as np
import pandas as pd

with open(r"c:\Users\ashpr\OneDrive\Desktop\Ashu\Programs\AntiGravProjects\mid_evaluation\data\hf_mixtral_c4_routing.jsonl", "r") as f:
    lines = [json.loads(line) for line in f if line.strip()]

seq_ginis = []
seq_top2_shares = []

def calc_gini(freqs):
    sorted_freqs = np.sort(freqs)
    n = len(freqs)
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * sorted_freqs)) / (n * np.sum(sorted_freqs)))

for rec in lines:
    exp_ids = rec["exp_ids"] # [2048, 2, 32]
    acts = []
    for t_data in exp_ids:
        acts.extend(t_data[0])
        acts.extend(t_data[1])
    counts = np.bincount(acts, minlength=8)
    freqs = counts / len(acts)
    g = calc_gini(freqs)
    sorted_f = np.sort(freqs)[::-1]
    top2 = sorted_f[:2].sum()
    seq_ginis.append(g)
    seq_top2_shares.append(top2)

print(f"Number of real sequences: {len(lines)}")
print(f"Per-sequence Mean Gini: {np.mean(seq_ginis):.4f} (range: {np.min(seq_ginis):.4f} - {np.max(seq_ginis):.4f})")
print(f"Per-sequence Mean Top-2 Share: {np.mean(seq_top2_shares)*100:.2f}% (range: {np.min(seq_top2_shares)*100:.2f}% - {np.max(seq_top2_shares)*100:.2f}%)")
