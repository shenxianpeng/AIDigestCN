"""
AI 领袖动态 — 每日自动抓取、翻译、发布到 GitHub Pages

数据流：
  people.yml
      │
      ▼
  fetch_tweets()      ← Twitter API Free Tier (tweepy)
      │ new tweets only (去重: processed_ids.json)
      ▼
  translate()         ← Claude Haiku (TITLE: / SUMMARY: 格式)
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
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import anthropic
import tweepy
import yaml
from jinja2 import Environment, FileSystemLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
PEOPLE_FILE = ROOT / "people.yml"
PROCESSED_IDS_FILE = ROOT / "processed_ids.json"
DOCS_DIR = ROOT / "docs"
ARCHIVE_DIR = DOCS_DIR / "archive"
TEMPLATES_DIR = ROOT / "templates"

# LLM prompt — 固定模板，稳定输出格式
PROMPT_TEMPLATE = """\
将以下推文翻译成中文。

格式要求（必须严格遵守，不要添加任何前缀或额外文字）：
TITLE: [一句话核心观点，不超过 20 字]
SUMMARY: [中文翻译，保留原意，不超过 100 字]

推文原文：
{tweet_text}"""


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

def fetch_tweets(handle: str, client: tweepy.Client) -> list[dict]:
    """
    抓取指定 Twitter 用户的最近推文（排除转推和回复）。
    任何错误都返回空列表，不抛出异常（单人失败不影响整体流程）。

    返回格式：[{"id": str, "text": str, "created_at": str, "url": str}]
    """
    try:
        user_resp = client.get_user(username=handle)
        if not user_resp.data:
            logger.warning(f"Twitter 用户未找到：@{handle}")
            return []

        user_id = user_resp.data.id
        tweets_resp = client.get_users_tweets(
            id=user_id,
            max_results=10,
            tweet_fields=["created_at", "text"],
            exclude=["retweets", "replies"],
        )

        if not tweets_resp.data:
            return []

        return [
            {
                "id": str(t.id),
                "text": t.text,
                "created_at": t.created_at.strftime("%Y-%m-%d %H:%M") if t.created_at else "",
                "url": f"https://twitter.com/{handle}/status/{t.id}",
            }
            for t in tweets_resp.data
        ]
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


def translate(tweet: dict, client: anthropic.Anthropic) -> dict[str, str]:
    """
    用 Claude Haiku 翻译一条推文。
    API 失败时返回原文 fallback（不抛出异常）。

    返回格式：{"title": str, "summary": str, "original": str}
    """
    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": PROMPT_TEMPLATE.format(tweet_text=tweet["text"]),
            }],
        )
        raw = message.content[0].text
        parsed = parse_llm_output(raw, tweet["text"])
        return {**parsed, "original": tweet["text"]}
    except Exception as e:
        logger.warning(f"LLM 翻译失败（使用原文 fallback）：{e}")
        return {
            "title": "（翻译失败）",
            "summary": tweet["text"],
            "original": tweet["text"],
        }


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def render_html(
    entries: list[TweetEntry],
    today: str,
    templates_dir: Path = TEMPLATES_DIR,
) -> str:
    """用 Jinja2 模板渲染 HTML。entries 为空时渲染空状态页面。"""
    env = Environment(loader=FileSystemLoader(str(templates_dir)), autoescape=True)
    template = env.get_template("day.html.j2")
    return template.render(entries=entries, date=today)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not bearer_token:
        raise ValueError("TWITTER_BEARER_TOKEN 环境变量未设置")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY 环境变量未设置")

    twitter_client = tweepy.Client(bearer_token=bearer_token)
    anthropic_client = anthropic.Anthropic(api_key=api_key)

    people = load_config()
    processed_ids = load_processed_ids()
    today = date.today().isoformat()

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

            logger.info(f"  翻译推文 {tweet['id']}")
            translated = translate(tweet, anthropic_client)

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

    html = render_html(entries, today)
    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")
    (ARCHIVE_DIR / f"{today}.html").write_text(html, encoding="utf-8")

    # 更新去重状态
    save_processed_ids(processed_ids | new_ids)

    logger.info(f"完成：{len(entries)} 条新内容，日期 {today}")


if __name__ == "__main__":
    main()
