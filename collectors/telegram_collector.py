# collectors/telegram_collector.py
import os
import json
import time
from pathlib import Path
from telethon import TelegramClient

import glob
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")


API_ID = int(os.getenv("TELEGRAM_API_ID"))
API_HASH = os.getenv("TELEGRAM_API_HASH")
SESSION = os.getenv("TELEGRAM_SESSION", "session")
CHANNELS = os.getenv("TELEGRAM_CHANNELS", "breakingmash,bbcbreaking,disclosetv,uncut_news,surf_noise_eng").split(",")
OUTPUT = Path(os.getenv("OUTPUT_DIR", "./raw/telegram")).resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)


async def dump():
    async with TelegramClient(SESSION, API_ID, API_HASH) as client:
        rows = []
        for ch in CHANNELS:
            async for m in client.iter_messages(ch, limit=500):
                rows.append({
                    "source": "telegram",
                    "channel": ch,
                    "id": m.id,
                    "text": m.text or "",
                    "date": m.date.isoformat(),
                    "url": f"https://t.me/{ch}/{m.id}"
                })
        out = OUTPUT / f"telegram_{int(time.time())}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} to {out}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(dump())