import os
import time
import json
from typing import List, Dict
from pathlib import Path

import praw
import glob
from dotenv import load_dotenv


project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")

OUTPUT = Path(os.getenv("OUTPUT_DIR", "./raw/reddit")).resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)
SUBREDDITS = os.getenv("SUBREDDITS", "worldnews,politics,fakenews,news,Economics,sports,Culture,science").split(",")
LIMIT = int(os.getenv("POST_LIMIT", 300))

client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
user_agent = os.getenv("REDDIT_USER_AGENT", "fake-news-detector/1.0")


# Validate
if not client_id or not client_secret:
    raise ValueError("Missing REDDIT_CLIENT_ID or REDDIT_CLIENT_SECRET in .env")

# Initialize Reddit client
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)


def fetch_submissions() -> List[Dict]:
    all_items = []
    # Delete all files in OUTPUT folder
    for f in OUTPUT.glob("*"):
        if f.is_file():
            f.unlink()
            print(f"🗑️ Deleted existing file: {f}")
    for sub in SUBREDDITS:
        sr = reddit.subreddit(sub)
        for post in sr.new(limit=LIMIT):
            all_items.append({
                "source": "reddit",
                "subreddit": sub,
                "id": post.id,
                "title": post.title,
                "text": post.selftext or "",
                "url": f"https://www.reddit.com{post.permalink}",
                "created_utc": int(post.created_utc)
            })
            time.sleep(0.2)
    return all_items


if __name__ == "__main__":
    data = fetch_submissions()
    out = OUTPUT / f"reddit_{int(time.time())}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(data)} items to {out}")