import os
import time
import json
import glob
from pathlib import Path
from googleapiclient.discovery import build

from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")

OUTPUT = Path(os.getenv("OUTPUT_DIR", "./raw/youtube")).resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)
API_KEY = os.getenv("YOUTUBE_API_KEY")
QUERY = os.getenv("YOUTUBE_QUERY", "breaking news")
MAX_RESULTS = int(os.getenv("YOUTUBE_MAX_RESULTS", 100))




def search():
    rows = []
    # Delete all files in OUTPUT folder
    for f in OUTPUT.glob("*"):
        if f.is_file():
            f.unlink()
            print(f"🗑️ Deleted existing file: {f}")

    yt = build("youtube", "v3", developerKey=API_KEY)
    req = yt.search().list(q=QUERY, part="snippet", maxResults=MAX_RESULTS, type="video")
    res = req.execute()
    for item in res.get("items", []):
        s = item["snippet"]
        rows.append({
            "source": "youtube",
            "videoId": item["id"]["videoId"],
            "title": s.get("title", ""),
            "text": s.get("description", ""),
            "channelTitle": s.get("channelTitle", ""),
            "publishedAt": s.get("publishedAt", ""),
            "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
        })
    return rows


if __name__ == "__main__":
    rows = search()
    out = OUTPUT / f"youtube_{int(time.time())}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} to {out}")