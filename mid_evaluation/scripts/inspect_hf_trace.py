import json

with open("mid_evaluation/data/hf_mixtral_c4_routing.jsonl", "r") as f:
    rec = json.loads(f.readline())

print("exp_ids type:", type(rec["exp_ids"]))
print("exp_ids length (tokens):", len(rec["exp_ids"]))
print("Token 0 exp_ids type:", type(rec["exp_ids"][0]))
print("Token 0 exp_ids len:", len(rec["exp_ids"][0]))
print("Token 0 exp_ids[0] (top-1 across 32 layers):", rec["exp_ids"][0][0])
print("Token 0 exp_ids[1] (top-2 across 32 layers):", rec["exp_ids"][0][1])
print("Are there 32 layers in top-1?", len(rec["exp_ids"][0][0]))
print("Are there 32 layers in top-2?", len(rec["exp_ids"][0][1]))
