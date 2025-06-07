"""Batch Scraper – Stage1

Download raw HTML pages for every *TheBatch* newsletter issue plus all
image assets referenced in each page.  Files are written under
`data/raw/` so later stages (parsing, embedding) can run offline.

Usage (run from project root):

    python -m src.batch_scraper.fetch_all --since 2024-01-01

or simply

    make scrape-now

The script is **idempotent** – if an issue’s HTML file already exists on
disk it will be skipped, likewise for previously‑downloaded images.

Structure on disk
-----------------

```
data/
└── raw/
    ├── issues/
    │   ├── issue-001.html
    │   ├── issue-002.html
    │   └── …
    └── images/
        ├── issue-001_f3e6c49d.jpg
        ├── issue-001_b44a91b2.png
        └── …
```

Notes
~~~~~
* The Batch URLs are sequential: `/the-batch/issue-<n>/`.
* A polite delay (1.5s) is inserted between requests.
* Requests are made with a custom User‑Agent identifying this scraper.
* If the site ever changes to block rapid sequential access, add an
  exponential back‑off or run with `--delay` >1.5.
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://www.deeplearning.ai/the-batch/issue-{num}/"
RAW_DIR = Path("data/raw")
ISSUES_DIR = RAW_DIR / "issues"
IMAGES_DIR = RAW_DIR / "images"

HEADERS = {
    "User-Agent": "the-batch-multimodal-rag-scraper/0.1 (+https://github.com/YOUR-USERNAME)"
}

POLITE_DELAY_SEC = 1.5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_mkdir(path: Path) -> None:
    """Create *path* and parents if needed (no error if exists)."""
    path.mkdir(parents=True, exist_ok=True)


def hash_url(url: str) -> str:
    """Return an 8‑character hex digest of *url*."""
    return hashlib.md5(url.encode()).hexdigest()[:8]


def request_url(url: str) -> Optional[str]:
    """HTTP‑GET *url* and return the text body or *None* on 404/other error."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as exc:
        print(f"[warn]Request failed for {url}: {exc}")
        return None


def download_binary(url: str, dest: Path) -> None:
    """Fetch *url* and write content to *dest* (skip if file exists)."""
    if dest.exists():
        return
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    except requests.RequestException as exc:
        print(f"[warn]Image download failed for {url}: {exc}")

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def discover_latest_issue(max_consecutive_misses: int = 5) -> int:
    """Return the highest issue number available by probing until *N* misses."""
    misses = 0
    n = 1
    latest = 0
    while misses < max_consecutive_misses:
        url = BASE_URL.format(num=n)
        if request_url(url) is None:
            misses += 1
        else:
            latest = n
            misses = 0
        n += 1
        time.sleep(0.2)
    return latest


def enumerate_issues(start: int, end: Optional[int]) -> Iterable[int]:
    """Generate issue numbers from *start* to *end* (inclusive). If *end* is
    None, discover the latest available automatically."""
    latest = discover_latest_issue() if end is None else end
    yield from range(start, latest + 1)


def fetch_issue_html(issue_num: int, delay: float = POLITE_DELAY_SEC) -> Optional[Path]:
    """Fetch one issue page and save raw HTML; return path or *None* if 404."""
    safe_mkdir(ISSUES_DIR)
    dest = ISSUES_DIR / f"issue-{issue_num:03d}.html"
    if dest.exists():
        return dest  # already scraped

    url = BASE_URL.format(num=issue_num)
    html = request_url(url)
    if html is None:
        return None

    dest.write_text(html, encoding="utf-8")
    time.sleep(delay)
    return dest


def parse_and_download_images(issue_html_path: Path) -> None:
    """Parse *issue_html_path*, find all <img> tags, download each image.
    Files are stored in `data/raw/images/` with a deterministic name."""
    safe_mkdir(IMAGES_DIR)
    soup = BeautifulSoup(issue_html_path.read_text(encoding="utf-8"), "html.parser")
    issue_slug = issue_html_path.stem  # e.g. "issue-001"

    for img in soup.find_all("img"):
        src = img.get("src") or ""
        if not src or src.startswith("data:"):
            continue  # skip inline images
        # Ensure absolute URL
        if src.startswith("//"):
            src = "https:" + src
        elif src.startswith("/"):
            src = "https://www.deeplearning.ai" + src

        # Derive deterministic filename
        ext = os.path.splitext(src.split("?")[0])[1] or ".jpg"
        fname = f"{issue_slug}_{hash_url(src)}{ext}"
        dest = IMAGES_DIR / fname
        download_binary(src, dest)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401  –pylint/docstring style
    parser = argparse.ArgumentParser(description="Scrape *The Batch* issues + images")
    parser.add_argument("--start", type=int, default=1, help="first issue number to fetch (default 1)")
    parser.add_argument("--end", type=int, help="last issue number; if omitted, auto‑detect latest")
    parser.add_argument(
        "--since",
        type=str,
        help="scrape only issues dated >= YYYY-MM-DD (overrides --start)",
    )
    parser.add_argument("--delay", type=float, default=POLITE_DELAY_SEC, help="seconds to sleep between requests")

    args = parser.parse_args()

    if args.since:
        try:
            since_date = datetime.fromisoformat(args.since).date()
        except ValueError as exc:
            raise SystemExit(f"Invalid date for --since: {args.since}") from exc
    else:
        since_date = None

    fetched = 0
    for n in tqdm(enumerate_issues(args.start, args.end), desc="Issues"):
        html_path = fetch_issue_html(n, delay=args.delay)
        if html_path is None:
            continue
        if since_date:
            # Quick filter: use <time> tag in page header to check publication date
            soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
            time_tag = soup.find("time")
            pub_date = (
                datetime.fromisoformat(time_tag["datetime"]).date() if time_tag and time_tag.has_attr("datetime") else None
            )
            if pub_date and pub_date < since_date:
                continue  # older than cutoff
        parse_and_download_images(html_path)
        fetched += 1

    print(f"[done] fetched {fetched} issue(s)")


if __name__ == "__main__":
    main()
