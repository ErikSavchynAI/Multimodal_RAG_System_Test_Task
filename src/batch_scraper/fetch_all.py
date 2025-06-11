"""
Batch Scraper for deeplearning.ai *The Batch* newsletters.

Usage:
    python -m src.batch_scraper.fetch_all --since 2024-01-01
    make scrape-now

Features:
* Downloads raw HTML for each issue and referenced images to `data/raw/`.
* Idempotent: skips already‑fetched files.
* Automatically detects latest issue unless --end supplied.
* Respects a polite delay between requests (default 1.5s).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://www.deeplearning.ai/the-batch/issue-{num}/"
RAW_DIR = Path("data/raw")
ISSUES_DIR = RAW_DIR / "issues"
IMAGES_DIR = RAW_DIR / "images"

HEADERS = {"User-Agent": "Multimodal_RAG_System_Test_Task/0.1 (+https://github.com/ErikSavchynAI)"}
POLITE_DELAY_SEC = 1.5


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def hash_url(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:8]


def request_url(url: str) -> Optional[str]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.text
    except requests.RequestException:
        print(f"[warn] Request failed for {url}")
        return None


def download_binary(url: str, dest: Path) -> None:
    if dest.exists():
        return
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    except requests.RequestException:
        print(f"[warn] Image download failed for {url}")


def discover_latest_issue(max_misses: int = 5) -> int:
    misses = latest = 0
    n = 1
    while misses < max_misses:
        if request_url(BASE_URL.format(num=n)) is None:
            misses += 1
        else:
            latest, misses = n, 0
        n += 1
        time.sleep(0.2)
    return latest


def enumerate_issues(start: int, end: Optional[int]) -> Iterable[int]:
    yield from range(start, (discover_latest_issue() if end is None else end) + 1)


def fetch_issue_html(issue_num: int, delay: float = POLITE_DELAY_SEC) -> Optional[Path]:
    safe_mkdir(ISSUES_DIR)
    dest = ISSUES_DIR / f"issue-{issue_num:03d}.html"
    if dest.exists():
        return dest

    html = request_url(BASE_URL.format(num=issue_num))
    if html is None:
        return None

    dest.write_text(html, encoding="utf-8")
    time.sleep(delay)
    return dest


def parse_and_download_images(issue_html_path: Path) -> None:
    safe_mkdir(IMAGES_DIR)
    soup = BeautifulSoup(issue_html_path.read_text(encoding="utf-8"), "html.parser")
    issue_slug = issue_html_path.stem

    for img in soup.find_all("img"):
        src = img.get("src") or ""
        if not src or src.startswith("data:"):
            continue
        if src.startswith("//"):
            src = "https:" + src
        elif src.startswith("/"):
            src = "https://www.deeplearning.ai" + src

        ext = os.path.splitext(src.split("?")[0])[1] or ".jpg"
        fname = f"{issue_slug}_{hash_url(src)}{ext}"
        download_binary(src, IMAGES_DIR / fname)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape The Batch issues and images")
    parser.add_argument("--start", type=int, default=1, help="first issue number (default 1)")
    parser.add_argument("--end", type=int, help="last issue number (auto detect if omitted)")
    parser.add_argument("--since", type=str, help="scrape only issues dated >= YYYY-MM-DD")
    parser.add_argument("--delay", type=float, default=POLITE_DELAY_SEC, help="delay between requests")
    args = parser.parse_args()

    since_date = datetime.fromisoformat(args.since).date() if args.since else None
    fetched = 0

    for n in tqdm(enumerate_issues(args.start, args.end), desc="Issues"):
        html_path = fetch_issue_html(n, delay=args.delay)
        if html_path is None:
            continue

        if since_date:
            soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
            time_tag = soup.find("time")
            pub_date = (
                datetime.fromisoformat(time_tag["datetime"]).date()
                if time_tag and time_tag.has_attr("datetime")
                else None
            )
            if pub_date and pub_date < since_date:
                continue

        parse_and_download_images(html_path)
        fetched += 1

    print(f"[done] fetched {fetched} issue(s)")


if __name__ == "__main__":
    main()
