"""
AI 领袖动态 — 每日自动抓取、翻译、发布到 GitHub Pages

数据流：
  people.yml
      │
      ▼
  fetch_tweets()      ← TweeterPy (无需 Bearer Token，使用 Guest Session)
      │ new tweets only (去重: processed_ids.json + LOOKBACK_DAYS 窗口)
      ▼
  translate()         ← Gemini Flash (TITLE: / SUMMARY: 格式)
      │
      ▼
  render_html()       ← Jinja2 模板
      │
      ▼
  docs/index.html + docs/archive/YYYY-MM-DD.html
  processed_ids.json  ← 更新去重状态（由 Actions 提交到 main）
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from google import genai
import yaml
from jinja2 import Environment, FileSystemLoader
from tweeterpy import TweeterPy
from tweeterpy.util import Tweet

# Configure logging AFTER TweeterPy imports.
# TweeterPy calls logging.config.dictConfig() at import time, which installs
# handlers on both the root logger and the "__main__" logger.  Without the fix
# below, every pipeline log line appears twice (once from the "__main__" handler
# and once via propagation to the root handler).
logging.basicConfig(
    force=True,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] :: %(message)s",
)
# Remove the extra handlers TweeterPy attached to the "__main__" named logger so
# that messages are only emitted once (through the root handler configured above).
logging.getLogger("__main__").handlers.clear()
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
PEOPLE_FILE = ROOT / "people.yml"
PROCESSED_IDS_FILE = ROOT / "processed_ids.json"
DOCS_DIR = ROOT / "docs"
ARCHIVE_DIR = DOCS_DIR / "archive"
TEMPLATES_DIR = ROOT / "templates"

# Only consider tweets published within this many days of today.
# This keeps each daily run focused on recent content and prevents the
# ever-growing processed_ids.json from masking genuinely new tweets.
LOOKBACK_DAYS = 7

# Gemini 免费 tier 限制：每分钟 5 次请求，间隔至少 12 秒
_GEMINI_MIN_INTERVAL = 12.0  # seconds
_last_gemini_request: float = 0.0

# LLM prompt — 固定模板，稳定输出格式
PROMPT_TEMPLATE = """\
将以下推文翻译成中文。

格式要求（必须严格遵守，不要添加任何前缀或额外文字）：
TITLE: [一句话核心观点，不超过 20 字]
SUMMARY: [中文翻译，保留原意，不超过 100 字]

推文原文：
{tweet_text}"""


# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------

def _is_within_lookback(tweet_date: str) -> bool:
    """
    Return True if *tweet_date* (``'YYYY-MM-DD HH:MM'`` or ``'YYYY-MM-DD'``)
    falls within the rolling lookback window defined by :data:`LOOKBACK_DAYS`.

    An empty / None date is treated as *recent* so the tweet is not silently
    dropped when date parsing fails.
    """
    if not tweet_date:
        return True
    cutoff = (date.today() - timedelta(days=LOOKBACK_DAYS)).isoformat()
    return tweet_date[:10] >= cutoff


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class Person:
    id: str
    name: str
    twitter_handle: str
    twitter_enabled: bool


@dataclass
class TweetEntry:
    tweet_id: str
    person_id: str
    person_name: str
    original_text: str
    tweet_url: str
    created_at: str
    title: str = ""
    summary: str = ""


# ---------------------------------------------------------------------------
# 配置与状态
# ---------------------------------------------------------------------------

def load_config(path: Path = PEOPLE_FILE) -> list[Person]:
    """解析 people.yml，返回 Person 列表。"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    people = []
    for item in data.get("people", []):
        twitter_enabled = any(
            s.get("type") == "twitter" and s.get("enabled", False)
            for s in item.get("sources", [])
        )
        people.append(Person(
            id=item["id"],
            name=item["name"],
            twitter_handle=item["twitter_handle"],
            twitter_enabled=twitter_enabled,
        ))
    return people


def load_processed_ids(path: Path = PROCESSED_IDS_FILE) -> set[str]:
    """
    加载已处理的推文 ID 集合。
    文件不存在或格式损坏时返回空集合（不崩溃）。
    """
    if not path.exists():
        return set()
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return set(data.get("ids", []))
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning("processed_ids.json 格式损坏，重置为空集合")
        return set()


def save_processed_ids(ids: set[str], path: Path = PROCESSED_IDS_FILE) -> None:
    """持久化已处理的推文 ID（写入 JSON，由 GitHub Actions 提交到仓库）。"""
    with open(path, "w") as f:
        json.dump({"ids": sorted(ids)}, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 抓取
# ---------------------------------------------------------------------------

def _parse_twitter_date(date_str: str) -> str:
    """将 Twitter 时间格式（'Mon Mar 22 09:00:00 +0000 2026'）转换为 'YYYY-MM-DD HH:MM'。"""
    try:
        dt = datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return ""


def fetch_tweets(handle: str, client: TweeterPy) -> list[dict]:
    """
    抓取指定 Twitter 用户的最近推文（排除转推和回复）。
    任何错误都返回空列表，不抛出异常（单人失败不影响整体流程）。

    返回格式：[{"id": str, "text": str, "created_at": str, "url": str}]
    """
    try:
        response = client.get_user_tweets(handle, total=10)
        if not response or not response.get("data"):
            logger.warning(f"Twitter 用户未找到或无推文：@{handle}")
            return []

        raw_count = len(response["data"])
        results = []
        skipped_no_text = skipped_rt = skipped_reply = 0
        for item in response["data"]:
            tweet = Tweet(item)
            if not tweet.full_text:
                skipped_no_text += 1
                continue
            # 排除转推
            if tweet.full_text.startswith("RT @"):
                skipped_rt += 1
                continue
            # 排除回复
            if tweet.in_reply_to_status_id_str:
                skipped_reply += 1
                continue

            tweet_id = tweet.rest_id or tweet.id_str
            results.append({
                "id": tweet_id,
                "text": tweet.full_text,
                "created_at": _parse_twitter_date(tweet.created_at),
                "url": tweet.tweet_url or f"https://x.com/{handle}/status/{tweet_id}",
            })

        logger.info(
            f"  @{handle}: 原始={raw_count}, 无文本={skipped_no_text}, "
            f"转推={skipped_rt}, 回复={skipped_reply}, 有效={len(results)}"
        )
        return results
    except Exception as e:
        logger.warning(f"@{handle} 抓取失败（已跳过）：{e}")
        return []


# ---------------------------------------------------------------------------
# 翻译
# ---------------------------------------------------------------------------

def parse_llm_output(text: str, original: str) -> dict[str, str]:
    """
    解析 LLM 返回的纯文本，提取 TITLE: 和 SUMMARY: 字段。

    格式正确时返回 {"title": ..., "summary": ...}。
    格式不符时返回原文 fallback（不跳过条目，避免页面出现空白项）。
    """
    title = ""
    summary = ""

    for line in text.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("TITLE:"):
            title = stripped[len("TITLE:"):].strip()
        elif stripped.startswith("SUMMARY:"):
            summary = stripped[len("SUMMARY:"):].strip()

    if not title and not summary:
        logger.warning("LLM 输出格式不符，使用原文 fallback")
        return {"title": "（原文）", "summary": original}

    return {
        "title": title or "（无标题）",
        "summary": summary or original,
    }


def _extract_retry_delay(error_str: str) -> float:
    """从 429 错误信息中解析建议的重试等待秒数，解析失败时默认 60 秒。"""
    match = re.search(r"retryDelay['\"]:\s*['\"](\d+)s", error_str)
    if match:
        return float(match.group(1)) + 2.0  # 额外留 2 秒缓冲
    return 60.0


def translate(tweet: dict, client: genai.Client) -> dict[str, str]:
    """
    用 Gemini Flash 翻译一条推文。
    - 每次请求前强制间隔 _GEMINI_MIN_INTERVAL 秒，避免超出免费 tier 速率限制。
    - 遇到 429 时读取 retryDelay 并自动重试，最多 3 次。
    - API 最终失败时返回原文 fallback（不抛出异常）。

    返回格式：{"title": str, "summary": str, "original": str}
    """
    global _last_gemini_request

    fallback = {
        "title": "（翻译失败）",
        "summary": tweet["text"],
        "original": tweet["text"],
    }

    for attempt in range(3):
        # 速率限制：距上次请求不足间隔时主动等待
        elapsed = time.time() - _last_gemini_request
        if elapsed < _GEMINI_MIN_INTERVAL:
            time.sleep(_GEMINI_MIN_INTERVAL - elapsed)

        try:
            _last_gemini_request = time.time()
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=PROMPT_TEMPLATE.format(tweet_text=tweet["text"]),
            )
            raw = response.text
            parsed = parse_llm_output(raw, tweet["text"])
            return {**parsed, "original": tweet["text"]}
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and attempt < 2:
                delay = _extract_retry_delay(error_str)
                logger.warning(f"  触发速率限制，等待 {delay:.0f}s 后重试（第 {attempt + 1} 次）…")
                time.sleep(delay)
                _last_gemini_request = 0.0  # 重置，让重试后立即发出请求
            else:
                logger.warning(f"LLM 翻译失败（使用原文 fallback）：{e}")
                return fallback

    return fallback


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def render_html(
    entries: list[TweetEntry],
    today: str,
    archive_url: str = "archive/",
    templates_dir: Path = TEMPLATES_DIR,
) -> str:
    """用 Jinja2 模板渲染 HTML。entries 为空时渲染空状态页面。"""
    env = Environment(loader=FileSystemLoader(str(templates_dir)), autoescape=True)
    template = env.get_template("day.html.j2")
    return template.render(entries=entries, date=today, archive_url=archive_url)


def render_archive_index(
    archive_dir: Path,
    templates_dir: Path = TEMPLATES_DIR,
) -> str:
    """扫描 archive_dir 下所有 YYYY-MM-DD.html，生成存档索引页。"""
    dates = sorted(
        [p.stem for p in archive_dir.glob("????-??-??.html")],
        reverse=True,
    )
    env = Environment(loader=FileSystemLoader(str(templates_dir)), autoescape=True)
    template = env.get_template("archive.html.j2")
    return template.render(dates=dates)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY 环境变量未设置")

    # log_level="ERROR" suppresses TweeterPy's INFO/WARNING noise (including the
    # harmless "[Errno 2] No such file or directory: '/tmp/tweeterpy_api.json'"
    # warning that appears on every clean CI run).  TweeterPy's constructor also
    # calls set_log_level() which resets ALL named loggers; restore our level
    # immediately afterwards so pipeline INFO messages remain visible.
    twitter_client = TweeterPy(log_level="ERROR")
    logger.setLevel(logging.INFO)

    gemini_client = genai.Client(api_key=api_key)

    people = load_config(PEOPLE_FILE)
    processed_ids = load_processed_ids(PROCESSED_IDS_FILE)
    today = date.today().isoformat()

    # On the very first run processed_ids is empty; skip the lookback window
    # so that we always bootstrap with whatever recent tweets are available,
    # regardless of how old they are relative to LOOKBACK_DAYS.
    is_first_run = not processed_ids
    if is_first_run:
        logger.info("processed_ids 为空，首次运行：跳过 lookback 窗口过滤")

    entries: list[TweetEntry] = []
    new_ids: set[str] = set()

    for person in people:
        if not person.twitter_enabled:
            continue

        logger.info(f"抓取 @{person.twitter_handle} 的推文…")
        tweets = fetch_tweets(person.twitter_handle, twitter_client)

        for tweet in tweets:
            if tweet["id"] in processed_ids:
                continue
            # Skip tweets that fall outside the rolling lookback window so that
            # an ever-growing processed_ids.json does not cause all fetched tweets
            # to look "already processed" after a few pipeline runs.
            # Always process all tweets on the first run (processed_ids empty).
            if not is_first_run and not _is_within_lookback(tweet["created_at"]):
                logger.debug(
                    f"  跳过推文 {tweet['id']}（日期 {tweet['created_at']} 超出 "
                    f"{LOOKBACK_DAYS} 天窗口）"
                )
                continue

            logger.info(f"  翻译推文 {tweet['id']}")
            translated = translate(tweet, gemini_client)

            entries.append(TweetEntry(
                tweet_id=tweet["id"],
                person_id=person.id,
                person_name=person.name,
                original_text=tweet["text"],
                tweet_url=tweet["url"],
                created_at=tweet["created_at"],
                title=translated["title"],
                summary=translated["summary"],
            ))
            new_ids.add(tweet["id"])

    # 写出 HTML
    DOCS_DIR.mkdir(exist_ok=True)
    ARCHIVE_DIR.mkdir(exist_ok=True)

    # Main page: archive link is "archive/" (relative to root)
    # Archive date page: archive link is "./" (current directory = archive/)
    (DOCS_DIR / "index.html").write_text(
        render_html(entries, today, archive_url="archive/"), encoding="utf-8"
    )
    (ARCHIVE_DIR / f"{today}.html").write_text(
        render_html(entries, today, archive_url="./"), encoding="utf-8"
    )

    archive_index = render_archive_index(ARCHIVE_DIR)
    (ARCHIVE_DIR / "index.html").write_text(archive_index, encoding="utf-8")

    # 更新去重状态
    save_processed_ids(processed_ids | new_ids, PROCESSED_IDS_FILE)

    logger.info(f"完成：{len(entries)} 条新内容，日期 {today}")


if __name__ == "__main__":
    main()
