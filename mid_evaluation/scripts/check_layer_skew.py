import json
import numpy as np

with open(r"c:\Users\ashpr\OneDrive\Desktop\Ashu\Programs\AntiGravProjects\mid_evaluation\data\hf_mixtral_c4_routing.jsonl", "r") as f:
    lines = [json.loads(line) for line in f if line.strip()]

layer_counts = np.zeros((32, 8))

for rec in lines:
    exp_ids = rec["exp_ids"] # [2048, 2, 32]
    for t_data in exp_ids:
        top1 = t_data[0] # 32 layers
        top2 = t_data[1] # 32 layers
        for l in range(32):
            layer_counts[l, top1[l]] += 1
            layer_counts[l, top2[l]] += 1

print("Layer-by-layer expert distributions:")
layer_ginis = []
for l in range(32):
    freqs = layer_counts[l] / layer_counts[l].sum()
    sorted_f = np.sort(freqs)
    n = len(freqs)
    idx = np.arange(1, n + 1)
    g = float((np.sum((2 * idx - n - 1) * sorted_f)) / (n * np.sum(sorted_f)))
    layer_ginis.append(g)
    top1_exp = np.argmax(freqs)
    top2_share = np.sort(freqs)[-2:].sum()
    if l in [0, 1, 2, 8, 16, 24, 31]:
        print(f"Layer {l:2d}: Most popular=E{top1_exp} ({freqs[top1_exp]*100:.1f}%), Top-2 Share={top2_share*100:.1f}%, Gini={g:.4f}")

print(f"\nMean Layer Gini: {np.mean(layer_ginis):.4f} (Max={np.max(layer_ginis):.4f} at Layer {np.argmax(layer_ginis)})")
