import pandas as pd
import json
from pathlib import Path

def jsonl_to_csv(in_file: Path, out_file: Path):
    rows = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df.to_csv(out_file, index=False)
    print(f"✅ Converted {in_file} → {out_file}")

if __name__ == "__main__":
    raw_dir = Path("raw")
    out_dir = Path("csv")
    out_dir.mkdir(exist_ok=True)

    for jsonl_file in raw_dir.rglob("*.jsonl"):  # search recursively
        out_file = out_dir / (jsonl_file.stem + ".csv")
        jsonl_to_csv(jsonl_file, out_file)
