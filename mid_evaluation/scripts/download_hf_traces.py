"""
download_hf_traces.py
Downloads real Mixtral 8x7B routing traces from Hugging Face:
Dataset: allenai/analysis_mixtral (real inference on C4 corpus)
"""

import os
import urllib.request
import json

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)
OUT_PATH = os.path.join(DATA_DIR, "hf_mixtral_c4_routing.jsonl")

URL = "https://huggingface.co/datasets/allenai/analysis_mixtral/resolve/main/c4_results.jsonl"

print(f"Downloading real Mixtral-8x7B routing traces from Hugging Face: {URL}")
req = urllib.request.Request(URL, headers={
    "User-Agent": "Mozilla/5.0",
    "Range": "bytes=0-5242880"  # Download 5 MB (approx 20 sequences of 2048 tokens each)
})

with urllib.request.urlopen(req) as res:
    raw_data = res.read().decode("utf-8", errors="ignore")

lines = raw_data.strip().split("\n")
# If the last line is truncated by range boundary, drop it
valid_records = []
for i, line in enumerate(lines):
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
        valid_records.append(obj)
    except Exception:
        # Last line may be truncated
        pass

print(f"Successfully loaded {len(valid_records)} complete sequences from Hugging Face.")
total_tokens = sum(len(rec["exp_ids"]) for rec in valid_records)
print(f"Total tokens across sequences: {total_tokens:,}")
print(f"Total routing decisions: {total_tokens * 32 * 2:,} (across 32 layers, top-2)")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for rec in valid_records:
        f.write(json.dumps(rec) + "\n")

print(f"Saved real Hugging Face traces to: {OUT_PATH} ({os.path.getsize(OUT_PATH):,} bytes)")
