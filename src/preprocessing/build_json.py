"""HTML → JSONL normalizer for *The Batch* newsletters.

Reads raw HTML issues (Stage‑1 output) and writes
`data/processed/batch_articles.jsonl`, one record per intro letter or
news article.

Usage:
    python -m src.preprocessing.build_json
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from bs4 import BeautifulSoup, NavigableString, Tag
from tqdm import tqdm

RAW_ISSUES_DIR = Path("data/raw/issues")
RAW_IMAGES_DIR = Path("data/raw/images")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "batch_articles.jsonl"

# ── helpers ──────────────────────────────────────────────────────────────

def hash_url(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:8]


def paragraph_text(tag: Tag) -> str:
    return " ".join(tag.get_text(" ").split())


_MONTH_REGEX = (
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}"
)


def extract_pub_date(soup: BeautifulSoup) -> Optional[str]:
    meta = soup.find("meta", property="article:published_time")
    if meta and meta.has_attr("content"):
        try:
            return datetime.fromisoformat(meta["content"].split("T")[0]).date().isoformat()
        except ValueError:
            pass
    time_tag = soup.find("time", datetime=True)
    if time_tag:
        try:
            return datetime.fromisoformat(time_tag["datetime"].split("T")[0]).date().isoformat()
        except ValueError:
            pass
    head_text = soup.get_text(" \n")[:3000]
    m = re.search(_MONTH_REGEX, head_text)
    if m:
        try:
            return datetime.strptime(m.group(0), "%B %d, %Y").date().isoformat()
        except ValueError:
            pass
    return None


# ── article container ───────────────────────────────────────────────────


class ArticleRecord:
    def __init__(self, issue: int, date: str, section: str, title: str):
        self.issue = issue
        self.date = date
        self.section = section
        self.title = title
        self._paras: List[str] = []
        self._images: List[dict] = []

    def add_text(self, text: str) -> None:
        text = " ".join(text.split())
        if text:
            self._paras.append(text)

    def add_image(self, src: str, alt: str) -> None:
        if not src:
            return
        ext = Path(src.split("?")[0]).suffix or ".jpg"
        fname = f"issue-{self.issue:03d}_{hash_url(src)}{ext}"
        self._images.append({"path": str(RAW_IMAGES_DIR / fname), "alt": alt})

    def is_empty(self) -> bool:
        return not self._paras and not self._images

    def as_dict(self, idx: int) -> dict:
        rec_id = f"issue-{self.issue:03d}_{self.section.lower()}_{idx}"
        return {
            "id": rec_id,
            "issue": self.issue,
            "date": self.date,
            "section": self.section,
            "title": self.title,
            "text": "\n\n".join(self._paras),
            "images": self._images,
        }


# ── parsing logic ────────────────────────────────────────────────────────

_SKIP_HEAD_RE = re.compile(r"^(news|sponsors?|a message from)\b", re.I)
_SIGNOFF_RE = re.compile(r"Keep (learning|building|pushing|coding|exploring)", re.I)


def process_issue(issue_path: Path) -> List[dict]:
    issue_num = int(issue_path.stem.split("-")[-1])
    soup = BeautifulSoup(issue_path.read_text(encoding="utf-8"), "html.parser")
    date_iso = extract_pub_date(soup) or ""
    main = soup.find("article") or soup

    output: List[dict] = []

    # intro letter
    intro_start = main.find("p", string=re.compile(r"^Dear ", re.I))
    if intro_start:
        intro = ArticleRecord(issue_num, date_iso, "Intro", "Intro Letter")
        for elem in intro_start.next_siblings:
            if isinstance(elem, NavigableString):
                continue
            if elem.name in {"h1", "h2", "h3"}:
                break
            txt = paragraph_text(elem)
            if _SIGNOFF_RE.search(txt):
                intro.add_text(txt)
                break
            for img in elem.find_all("img"):
                intro.add_image(img.get("src", ""), img.get("alt", ""))
            intro.add_text(txt)
        if not intro.is_empty():
            output.append(intro.as_dict(0))

    # news articles
    idx = 1
    for heading in main.find_all(["h2", "h3"]):
        title = paragraph_text(heading)
        if _SKIP_HEAD_RE.match(title):
            continue
        if not title:
            img_tag = heading.find("img")
            title = img_tag["alt"].strip() if img_tag and img_tag.get("alt") else "Untitled"
        article = ArticleRecord(issue_num, date_iso, "News", title)
        for elem in heading.next_siblings:
            if isinstance(elem, NavigableString):
                continue
            if elem.name in {"h2", "h3"}:
                break
            for img in elem.find_all("img"):
                article.add_image(img.get("src", ""), img.get("alt", ""))
            article.add_text(paragraph_text(elem))
        if not article.is_empty():
            output.append(article.as_dict(idx))
            idx += 1

    return output


# ── entry point ──────────────────────────────────────────────────────────

def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as out_f:
        total = 0
        for issue_file in tqdm(sorted(RAW_ISSUES_DIR.glob("issue-*.html")), desc="Parse issues"):
            for rec in process_issue(issue_file):
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1
    print(f"[done] wrote {total} records → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
