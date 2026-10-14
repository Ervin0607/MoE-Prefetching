import urllib.request
import json

url = "https://huggingface.co/datasets/allenai/analysis_mixtral/resolve/main/c4_results.jsonl"
try:
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Range": "bytes=0-524288"  # Download first 512 KB
    })
    with urllib.request.urlopen(req) as res:
        chunk = res.read().decode("utf-8", errors="ignore")
        lines = chunk.split("\n")
        print(f"Downloaded {len(chunk)} bytes, {len(lines)} lines")
        if lines:
            first_obj = json.loads(lines[0])
            print("Keys in first record:", list(first_obj.keys()))
            for k in first_obj:
                val = first_obj[k]
                if isinstance(val, list):
                    print(f" - {k}: list of length {len(val)}, first elem type={type(val[0])}")
                    if len(val) > 0 and isinstance(val[0], list):
                        print(f"   inner list length={len(val[0])}, sample={val[0][:5]}")
                else:
                    print(f" - {k}: {val}")
except Exception as e:
    print(f"Error fetching chunk: {e}")
