import json
import re
from pathlib import Path
from typing import Iterable


URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    s = URL_RE.sub("", s)
    s = s.replace("\n", " ")
    s = WS_RE.sub(" ", s)
    return s.strip()


def stream_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def clean_folder(in_dir: str, out_file: str):
    in_path = Path(in_dir)
    rows = []
    for p in in_path.glob("*.jsonl"):
        for item in stream_jsonl(p):
            text = (item.get("text") or item.get("title") or "").strip()
            if text:
                item["text_norm"] = normalize_text(text)
                rows.append(item)
    with Path(out_file).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {out_file}")


if __name__ == "__main__":
    import typer
    typer.run(clean_folder)