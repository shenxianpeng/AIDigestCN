"""
AI 领袖动态 — 每日自动抓取、翻译、发布到 GitHub Pages

数据流：
  people.yml
      │
      ▼
  fetch_tweets()      ← TweeterPy (无需 Bearer Token，使用 Guest Session)
      │ new tweets only (去重: processed_ids.json + LOOKBACK_DAYS 窗口)
      ▼
  translate()         ← GitHub Copilot 模型 (TITLE: / SUMMARY: 格式)
      │
      ▼
  render_html()       ← Jinja2 模板
      │
      ▼
  docs/index.html + docs/archive/YYYY-MM-DD.html
  processed_ids.json  ← 更新去重状态（由 Actions 提交到 main）
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from copilot import CopilotClient, SubprocessConfig, PermissionHandler
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

# Pull a larger tweet window per account so that we can still find unseen
# original tweets when a user's latest posts are mostly replies/retweets.
FETCH_TWEETS_PER_USER = int(os.environ.get("TWEETS_PER_USER", "40"))

# GitHub Copilot SDK 配置
# 通过环境变量可覆盖模型名称，默认使用 gpt-4o-mini
COPILOT_MODEL = os.environ.get("COPILOT_MODEL", "gpt-4o-mini")
# 调试时可将 COPILOT_LOG_LEVEL 设为 "info" 以查看 Copilot CLI 输出
COPILOT_LOG_LEVEL = os.environ.get("COPILOT_LOG_LEVEL", "error")

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
        json.dump({"ids": sorted(id for id in ids if id is not None)}, f, indent=2, ensure_ascii=False)


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
        response = client.get_user_tweets(handle, total=FETCH_TWEETS_PER_USER)
        if not response or not response.get("data"):
            logger.warning(f"Twitter 用户未找到或无推文：@{handle}")
            return []

        raw_count = len(response["data"])
        results = []
        skipped_no_text = skipped_no_id = skipped_rt = skipped_reply = 0
        for item in response["data"]:
            tweet = Tweet(item)
            # tweeterpy's find_nested_key() can return a list when a tweet
            # contains nested tweet data (e.g. quoted tweets).  Normalise to
            # a plain string by taking the first element so that downstream
            # str operations like .startswith() don't raise TypeError.
            full_text = tweet.full_text
            if isinstance(full_text, list):
                full_text = full_text[0] if full_text else None

            if not full_text:
                skipped_no_text += 1
                continue
            # 排除转推
            if full_text.startswith("RT @"):
                skipped_rt += 1
                continue
            # 排除回复 — in_reply_to_status_id_str may also be a list
            reply_id = tweet.in_reply_to_status_id_str
            if isinstance(reply_id, list):
                reply_id = reply_id[0] if reply_id else None
            if reply_id:
                skipped_reply += 1
                continue

            tweet_id = tweet.rest_id or tweet.id_str
            if not tweet_id:
                skipped_no_id += 1
                continue
            results.append({
                "id": tweet_id,
                "text": full_text,
                "created_at": _parse_twitter_date(tweet.created_at),
                "url": f"https://x.com/{handle}/status/{tweet_id}",
            })

        logger.info(
            f"  @{handle}: 原始={raw_count}, 无文本={skipped_no_text}, 无ID={skipped_no_id}, "
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


async def _do_translate(tweet_text: str, github_token: str) -> str:
    """
    用 GitHub Copilot SDK 发起一次翻译请求，返回 LLM 的原始文本响应。
    每次调用都会启动一个 Copilot CLI 进程并在结束时关闭。
    """
    config = SubprocessConfig(github_token=github_token, log_level=COPILOT_LOG_LEVEL)
    async with CopilotClient(config) as client:
        async with await client.create_session(
            on_permission_request=PermissionHandler.approve_all,
            model=COPILOT_MODEL,
        ) as session:
            event = await session.send_and_wait(
                PROMPT_TEMPLATE.format(tweet_text=tweet_text),
                timeout=120.0,
            )
    return event.data.content if event and event.data.content else ""


def translate(tweet: dict, github_token: str) -> dict[str, str]:
    """
    用 GitHub Copilot SDK 翻译一条推文。
    - 遇到 429 速率限制时自动等待 60 秒后重试，最多 3 次。
    - API 最终失败时返回原文 fallback（不抛出异常）。

    返回格式：{"title": str, "summary": str, "original": str}
    """
    fallback = {
        "title": "（翻译失败）",
        "summary": tweet["text"],
        "original": tweet["text"],
    }

    for attempt in range(3):
        try:
            raw = asyncio.run(_do_translate(tweet["text"], github_token))
            if not raw:
                raise ValueError("Copilot 返回空响应")
            parsed = parse_llm_output(raw, tweet["text"])
            return {**parsed, "original": tweet["text"]}
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and attempt < 2:
                logger.warning(f"  触发速率限制，等待 60s 后重试（第 {attempt + 1} 次）…")
                time.sleep(60)
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
    github_token = os.environ.get("GITHUB_TOKEN")

    if not github_token:
        raise ValueError("GITHUB_TOKEN 环境变量未设置")

    # log_level="CRITICAL" suppresses TweeterPy's INFO/WARNING/ERROR noise
    # (including the harmless "[Errno 2] No such file or directory: '/tmp/tweeterpy_api.json'"
    # warning that appears on every clean CI run, and the rate-limit ERROR messages
    # that TweeterPy logs internally when it hits HTTP 429 — those would otherwise
    # appear as ##[error] annotations in GitHub Actions even though the pipeline
    # handles them gracefully).  TweeterPy's constructor also calls set_log_level()
    # which resets ALL named loggers; restore our level immediately afterwards so
    # pipeline INFO messages remain visible.
    twitter_client = TweeterPy(log_level="CRITICAL")
    logger.setLevel(logging.INFO)

    # Authenticated sessions have access to the full current timeline.
    # Guest sessions are limited to historical/cached data (~ Nov 2025 cutoff)
    # which causes the pipeline to find 0 new tweets with a short lookback window.
    # Set TWITTER_AUTH_TOKEN (GitHub Secret) to an auth_token cookie value from a
    # logged-in Twitter/X session to unlock the real-time timeline.
    twitter_auth_token = os.environ.get("TWITTER_AUTH_TOKEN")
    if twitter_auth_token:
        twitter_client.generate_session(auth_token=twitter_auth_token)
        logger.info("Twitter 会话：已认证（TWITTER_AUTH_TOKEN 已配置）")
    else:
        logger.warning(
            "TWITTER_AUTH_TOKEN 未设置，使用 Guest Session。"
            " Guest Session 仅能访问约 2025-11 以前的历史推文，无法获取当前内容。"
            " 请在 GitHub Secrets 中配置 TWITTER_AUTH_TOKEN 以获取最新推文。"
        )

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
    total_skipped_ids = 0       # already in processed_ids
    total_skipped_lookback = 0  # outside rolling lookback window

    for person in people:
        if not person.twitter_enabled:
            continue

        logger.info(f"抓取 @{person.twitter_handle} 的推文…")
        tweets = fetch_tweets(person.twitter_handle, twitter_client)

        for tweet in tweets:
            if tweet["id"] in processed_ids:
                total_skipped_ids += 1
                continue
            # Skip tweets that fall outside the rolling lookback window so that
            # an ever-growing processed_ids.json does not cause all fetched tweets
            # to look "already processed" after a few pipeline runs.
            # Always process all tweets on the first run (processed_ids empty).
            if not is_first_run and not _is_within_lookback(tweet["created_at"]):
                logger.info(
                    f"  跳过推文 {tweet['id']}（日期 {tweet['created_at']} 超出 "
                    f"{LOOKBACK_DAYS} 天窗口）— 需配置 TWITTER_AUTH_TOKEN 以获取最新推文"
                )
                total_skipped_lookback += 1
                continue

            logger.info(f"  翻译推文 {tweet['id']}")
            translated = translate(tweet, github_token)

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
    if entries:
        (ARCHIVE_DIR / f"{today}.html").write_text(
            render_html(entries, today, archive_url="./"), encoding="utf-8"
        )

    archive_index = render_archive_index(ARCHIVE_DIR)
    (ARCHIVE_DIR / "index.html").write_text(archive_index, encoding="utf-8")

    # 更新去重状态
    save_processed_ids(processed_ids | new_ids, PROCESSED_IDS_FILE)

    if total_skipped_ids or total_skipped_lookback:
        logger.info(
            f"过滤摘要：{total_skipped_ids} 条已处理（processed_ids），"
            f"{total_skipped_lookback} 条超出 {LOOKBACK_DAYS} 天窗口"
        )

    logger.info(f"完成：{len(entries)} 条新内容，日期 {today}")


if __name__ == "__main__":
    main()
