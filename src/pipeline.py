"""
AI 领袖动态 — 每日自动抓取、翻译、发布到 GitHub Pages

数据流：
  people.yml
      │
      ▼
  fetch_tweets()      ← TweeterPy (无需 Bearer Token，使用 Guest Session)
      │ new tweets only (去重: processed_ids.json + LOOKBACK_DAYS 窗口)
      ▼
  translate_batch()   ← OpenAI API (gpt-4o-mini，批量翻译降低调用次数)
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

from openai import OpenAI
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

# 开关：是否跳过纯转推（即转发他人推文但未添加任何评论的帖子）。
# 设为 "true"（默认）时，纯 RT 不会被翻译和展示；设为 "false" 时保留所有转推。
SKIP_PURE_REPOSTS = os.environ.get("SKIP_PURE_REPOSTS", "true").lower() == "true"

# OpenAI 配置
# gpt-4o-mini 具有较高的速率限制配额，适合批量翻译场景
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
# 每批翻译的推文数量（减少 API 调用次数，降低触发速率限制的风险）
TRANSLATE_BATCH_SIZE = int(os.environ.get("TRANSLATE_BATCH_SIZE", "10"))

# 批量翻译 prompt — 一次 API 调用处理多条推文
BATCH_PROMPT_TEMPLATE = """\
将以下 {n} 条推文分别翻译成中文。
部分推文含有引用或转发内容，请结合全文理解完整信息。

格式要求（严格按编号顺序输出，每条用 [编号] 分隔，不添加任何额外文字）：
[1]
TITLE: 一句话核心观点（不超过 20 字）
SUMMARY: 中文翻译（保留原意，如有引用请概括完整背景，不超过 150 字）
CONTEXT: 被转发/引用原帖的直接中文翻译（仅当原文中含有转发或引用内容时填写，否则留空）

[2]
TITLE: ...
SUMMARY: ...
CONTEXT: ...

推文原文：
{tweets}"""


def _build_tweet_prompt_text(tweet: dict) -> str:
    """
    为 LLM 翻译构建上下文感知的文本。
    若推文包含引用/转发内容，将其纳入提示词以生成更准确的翻译。
    """
    text = tweet.get("text", "")
    context_text = tweet.get("context_text", "")
    context_author = tweet.get("context_author", "")
    is_repost = tweet.get("is_repost", False)

    if not context_text:
        return text

    author_note = f"@{context_author}" if context_author else "某用户"
    if is_repost:
        return f"（转发自 {author_note}）\n{text}"
    else:
        return f"（评论/引用了 {author_note} 的推文）\n原帖：{context_text}\n回应：{text}"


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
    role: str = ""                # 角色/职位（可选，显示在卡片上）


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
    twitter_handle: str = ""      # Twitter @handle（用于头像路径）
    person_role: str = ""         # 角色/职位（来自 people.yml 的 role 字段）
    context_text: str = ""        # 被引用/转发推文的内容（英文原文）
    context_author: str = ""      # 被引用/转发推文的作者 handle
    context_url: str = ""         # 被引用推文的链接
    is_repost: bool = False       # True 表示纯转推（无评论）
    context_translated: str = ""  # 被引用/转发推文的中文翻译


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
            role=item.get("role", ""),
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
    抓取指定 Twitter 用户的最近推文（排除无法解析的转推和回复）。
    对于含引用推文的帖子，提取被引用内容作为上下文（context_text/context_author/context_url）。
    对于转推（RT @），提取被转推的原始内容，以便展示完整信息。
    任何错误都返回空列表，不抛出异常（单人失败不影响整体流程）。

    返回格式：[{
        "id": str, "text": str, "created_at": str, "url": str,
        "context_text": str, "context_author": str, "context_url": str,
        "is_repost": bool,
    }]
    """
    try:
        response = client.get_user_tweets(handle, total=FETCH_TWEETS_PER_USER)
        if not response or not response.get("data"):
            logger.warning(f"Twitter 用户未找到或无推文：@{handle}")
            return []

        raw_count = len(response["data"])
        results = []
        skipped_no_text = skipped_no_id = skipped_rt = skipped_reply = skipped_pure_repost = 0
        repost_count = 0
        for item in response["data"]:
            tweet = Tweet(item)
            # tweeterpy's find_nested_key() can return a list when a tweet
            # contains nested tweet data (e.g. quoted tweets).  The first
            # element is the main tweet text; the second (if present) is the
            # quoted tweet's text which we capture as context.
            full_texts = tweet.full_text
            context_text = ""
            context_author = ""
            context_url = ""

            if isinstance(full_texts, list):
                full_text = full_texts[0] if full_texts else None
                if len(full_texts) > 1:
                    context_text = full_texts[1] or ""
            else:
                full_text = full_texts

            if not full_text:
                skipped_no_text += 1
                continue

            # 处理转推 — 提取原始内容和作者，而非直接跳过
            is_repost = False
            if full_text.startswith("RT @"):
                rt_match = re.match(r"RT @(\w+): (.*)", full_text, re.DOTALL)
                if rt_match:
                    rt_handle = rt_match.group(1)
                    rt_content = rt_match.group(2).rstrip("…").strip()
                    # 优先使用嵌套数据中的完整文本（避免 RT 截断问题）
                    if not context_text:
                        context_text = rt_content
                    context_author = rt_handle
                    full_text = context_text  # 用原始内容替换 RT @ 前缀
                    is_repost = True
                    repost_count += 1
                    # 若开关开启，纯转推（无评论）直接跳过，不翻译
                    if SKIP_PURE_REPOSTS:
                        skipped_pure_repost += 1
                        continue
                else:
                    skipped_rt += 1
                    continue

            # 排除回复 — in_reply_to_status_id_str may also be a list
            reply_id = tweet.in_reply_to_status_id_str
            if isinstance(reply_id, list):
                reply_id = reply_id[0] if reply_id else None
            if reply_id:
                skipped_reply += 1
                continue

            # 提取引用推文的作者（当上下文文本存在但作者未知时）
            if context_text and not context_author:
                screen_names = tweet.screen_name
                if isinstance(screen_names, list) and len(screen_names) > 1:
                    context_author = screen_names[1]

            # 处理 rest_id / id_str 可能为列表的情况（多层嵌套推文）
            rest_id_raw = tweet.rest_id
            if isinstance(rest_id_raw, list):
                quoted_rest_id = rest_id_raw[1] if len(rest_id_raw) > 1 else None
                rest_id_val = rest_id_raw[0] if rest_id_raw else None
            else:
                quoted_rest_id = None
                rest_id_val = rest_id_raw

            id_str_raw = tweet.id_str
            id_str_val = id_str_raw[0] if isinstance(id_str_raw, list) else id_str_raw

            tweet_id = rest_id_val or id_str_val
            if not tweet_id:
                skipped_no_id += 1
                continue

            # 构建引用推文的 URL（如有足够信息）
            if context_text and context_author and not context_url and quoted_rest_id:
                context_url = f"https://x.com/{context_author}/status/{quoted_rest_id}"

            results.append({
                "id": tweet_id,
                "text": full_text,
                "created_at": _parse_twitter_date(tweet.created_at),
                "url": f"https://x.com/{handle}/status/{tweet_id}",
                "context_text": context_text,
                "context_author": context_author,
                "context_url": context_url,
                "is_repost": is_repost,
            })

        logger.info(
            f"  @{handle}: 原始={raw_count}, 无文本={skipped_no_text}, 无ID={skipped_no_id}, "
            f"转推(含上下文)={repost_count}, 转推(无法解析)={skipped_rt}, "
            f"纯转推(已跳过)={skipped_pure_repost}, 回复={skipped_reply}, 有效={len(results)}"
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
    context_translated = ""

    for line in text.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("TITLE:"):
            title = stripped[len("TITLE:"):].strip()
        elif stripped.startswith("SUMMARY:"):
            summary = stripped[len("SUMMARY:"):].strip()
        elif stripped.startswith("CONTEXT:"):
            context_translated = stripped[len("CONTEXT:"):].strip()

    if not title and not summary:
        logger.warning("LLM 输出格式不符，使用原文 fallback")
        return {"title": "（原文）", "summary": original, "context_translated": ""}

    return {
        "title": title or "（无标题）",
        "summary": summary or original,
        "context_translated": context_translated,
    }


def _call_openai(prompt: str, api_key: str) -> str:
    """
    调用 OpenAI Chat Completions API，返回模型的原始文本响应。
    此函数是可 mock 的最小单元，不含重试逻辑。
    """
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        timeout=120,
    )
    return response.choices[0].message.content or ""


def _parse_batch_response(text: str, tweets: list[dict]) -> list[dict[str, str]]:
    """
    解析批量翻译响应，按 [编号] 标记拆分并提取各条推文的 TITLE/SUMMARY。
    同时兼容 LLM 可能返回的中文全角括号 【编号】。
    任何缺失条目均使用原文 fallback。
    """
    parts = re.split(r'[\[【](\d+)[\]】]', text.strip())
    # parts: ['preamble', '1', 'content1', '2', 'content2', ...]

    section_map: dict[int, str] = {}
    for i in range(1, len(parts), 2):
        try:
            idx = int(parts[i])
            content = parts[i + 1] if i + 1 < len(parts) else ""
            section_map[idx] = content.strip()
        except (ValueError, IndexError):
            continue

    results = []
    for i, tweet in enumerate(tweets):
        num = i + 1
        content = section_map.get(num, "")
        if content:
            parsed = parse_llm_output(content, tweet["text"])
        else:
            logger.warning(f"批量翻译：第 {num} 条结果缺失，使用原文 fallback")
            parsed = {"title": "（翻译失败）", "summary": tweet["text"]}
        results.append({**parsed, "original": tweet["text"]})

    return results


def translate_batch(tweets: list[dict], api_key: str) -> list[dict[str, str]]:
    """
    用 OpenAI API 批量翻译推文（一次 API 调用处理多条，显著减少调用次数）。
    - 遇到 429 速率限制时自动等待 60 秒后重试，最多 3 次。
    - API 最终失败时所有条目返回原文 fallback，不抛出异常。

    返回格式：[{"title": str, "summary": str, "original": str}, ...]，顺序与输入一致。
    """
    if not tweets:
        return []

    fallback_results = [
        {"title": "（翻译失败）", "summary": t["text"], "original": t["text"]}
        for t in tweets
    ]

    tweets_block = "\n\n".join(
        f"[{i + 1}]\n{_build_tweet_prompt_text(t)}" for i, t in enumerate(tweets)
    )
    prompt = BATCH_PROMPT_TEMPLATE.format(n=len(tweets), tweets=tweets_block)

    for attempt in range(3):
        try:
            raw = _call_openai(prompt, api_key)
            if not raw:
                raise ValueError("OpenAI 返回空响应")
            return _parse_batch_response(raw, tweets)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and attempt < 2:
                logger.warning(f"  触发速率限制，等待 60s 后重试（第 {attempt + 1} 次）…")
                time.sleep(60)
            else:
                logger.warning(f"批量翻译失败（使用原文 fallback）：{e}")
                return fallback_results

    return fallback_results


def translate(tweet: dict, api_key: str) -> dict[str, str]:
    """
    翻译单条推文（translate_batch 的单条包装，保留此接口以便测试调用）。

    返回格式：{"title": str, "summary": str, "original": str}
    """
    results = translate_batch([tweet], api_key)
    return results[0] if results else {
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
    archive_url: str = "archive/",
    avatar_base: str = "avatars/",
    templates_dir: Path = TEMPLATES_DIR,
) -> str:
    """用 Jinja2 模板渲染 HTML。entries 为空时渲染空状态页面。"""
    env = Environment(loader=FileSystemLoader(str(templates_dir)), autoescape=True)
    template = env.get_template("day.html.j2")
    return template.render(entries=entries, date=today, archive_url=archive_url, avatar_base=avatar_base)


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
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY 环境变量未设置")

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

    # Phase 1: fetch and filter tweets from all people
    pending: list[tuple] = []  # [(person, tweet), ...]
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
            pending.append((person, tweet))

    # Phase 2: batch translate all pending tweets to minimise API calls
    for batch_start in range(0, len(pending), TRANSLATE_BATCH_SIZE):
        batch = pending[batch_start:batch_start + TRANSLATE_BATCH_SIZE]
        batch_num = batch_start // TRANSLATE_BATCH_SIZE + 1
        logger.info(f"批量翻译第 {batch_num} 批（{len(batch)} 条）…")

        tweet_batch = [t for _, t in batch]
        translations = translate_batch(tweet_batch, openai_api_key)

        for (person, tweet), translated in zip(batch, translations):
            entries.append(TweetEntry(
                tweet_id=tweet["id"],
                person_id=person.id,
                person_name=person.name,
                twitter_handle=person.twitter_handle,
                person_role=person.role,
                original_text=tweet["text"],
                tweet_url=tweet["url"],
                created_at=tweet["created_at"],
                title=translated["title"],
                summary=translated["summary"],
                context_text=tweet.get("context_text", ""),
                context_author=tweet.get("context_author", ""),
                context_url=tweet.get("context_url", ""),
                is_repost=tweet.get("is_repost", False),
                context_translated=translated.get("context_translated", ""),
            ))
            new_ids.add(tweet["id"])

    # 写出 HTML
    DOCS_DIR.mkdir(exist_ok=True)
    ARCHIVE_DIR.mkdir(exist_ok=True)

    # Main page: archive link is "archive/" (relative to root)
    # Archive date page: archive link is "./" (current directory = archive/)
    (DOCS_DIR / "index.html").write_text(
        render_html(entries, today, archive_url="archive/", avatar_base="avatars/"), encoding="utf-8"
    )
    if entries:
        (ARCHIVE_DIR / f"{today}.html").write_text(
            render_html(entries, today, archive_url="./", avatar_base="../avatars/"), encoding="utf-8"
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
