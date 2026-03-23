"""
pytest 测试 — pipeline.py 核心函数

测试覆盖（见测试计划）：
  test_load_config_valid              — 正常解析 people.yml
  test_load_config_twitter_disabled   — enabled:false 时 twitter_enabled=False
  test_load_config_rss_source         — 解析 RSS source（url/enabled）
  test_load_config_rss_disabled       — rss enabled:false 时 rss_enabled=False
  test_load_processed_ids_new         — 文件不存在时返回空集合
  test_load_processed_ids_existing    — 加载已有 ID 列表
  test_load_processed_ids_malformed   — 格式损坏时返回空集合（不崩溃）
  test_parse_llm_output_valid         — 正常 TITLE:/SUMMARY: 解析
  test_parse_llm_output_fallback      — 格式不符时返回原文 fallback
  test_parse_llm_output_partial       — 只有 SUMMARY 时正常处理
  test_render_html_with_entries       — 有内容时渲染正确 HTML
  test_render_html_empty              — 无内容时渲染空状态 HTML
  test_fetch_tweets_happy_path        — 正常返回推文列表
  test_fetch_tweets_user_not_found    — 用户不存在时返回 []
  test_fetch_tweets_no_tweets         — 无推文时返回 []
  test_fetch_tweets_exception         — 任何异常返回 []（不抛出）
  test_fetch_rss_happy_path           — 正常返回 RSS 条目列表
  test_fetch_rss_no_entries           — feed 无条目时返回 []
  test_fetch_rss_html_stripped        — 正文 HTML 标签被去除
  test_fetch_rss_exception            — 任何异常返回 []（不抛出）
  test_translate_success              — 正常翻译返回 title/summary/original
  test_translate_api_failure          — API 异常时返回 fallback dict
  test_main_missing_twitter_token     — 缺少 TWITTER_BEARER_TOKEN 时抛出 ValueError
  test_main_missing_anthropic_key     — 缺少 ANTHROPIC_API_KEY 时抛出 ValueError
"""

import json
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# 把 src/ 加入 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import (
    TweetEntry,
    fetch_rss,
    fetch_tweets,
    load_config,
    load_processed_ids,
    main,
    parse_llm_output,
    render_html,
    save_processed_ids,
    translate,
)


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

def test_load_config_valid(tmp_path):
    """正常 YAML 解析出正确的 Person 列表。"""
    config = {
        "people": [
            {
                "id": "sam_altman",
                "name": "Sam Altman",
                "twitter_handle": "sama",
                "sources": [{"type": "twitter", "enabled": True}],
            }
        ]
    }
    p = tmp_path / "people.yml"
    p.write_text(yaml.dump(config), encoding="utf-8")

    people = load_config(p)

    assert len(people) == 1
    assert people[0].id == "sam_altman"
    assert people[0].name == "Sam Altman"
    assert people[0].twitter_handle == "sama"
    assert people[0].twitter_enabled is True


def test_load_config_twitter_disabled(tmp_path):
    """enabled: false 时 twitter_enabled 应为 False。"""
    config = {
        "people": [
            {
                "id": "test_person",
                "name": "Test",
                "twitter_handle": "test",
                "sources": [{"type": "twitter", "enabled": False}],
            }
        ]
    }
    p = tmp_path / "people.yml"
    p.write_text(yaml.dump(config), encoding="utf-8")

    people = load_config(p)
    assert people[0].twitter_enabled is False


def test_load_config_empty_people(tmp_path):
    """people 列表为空时返回空列表，不崩溃。"""
    p = tmp_path / "people.yml"
    p.write_text(yaml.dump({"people": []}), encoding="utf-8")

    assert load_config(p) == []


def test_load_config_rss_source(tmp_path):
    """rss source 的 url 和 enabled 被正确解析。"""
    config = {
        "people": [
            {
                "id": "sam_altman",
                "name": "Sam Altman",
                "twitter_handle": "sama",
                "sources": [
                    {"type": "twitter", "enabled": False},
                    {"type": "rss", "url": "https://blog.samaltman.com/rss", "enabled": True},
                ],
            }
        ]
    }
    p = tmp_path / "people.yml"
    p.write_text(yaml.dump(config), encoding="utf-8")

    people = load_config(p)
    assert people[0].rss_enabled is True
    assert people[0].rss_url == "https://blog.samaltman.com/rss"
    assert people[0].twitter_enabled is False


def test_load_config_rss_disabled(tmp_path):
    """rss enabled:false 时 rss_enabled 应为 False，rss_url 为空字符串。"""
    config = {
        "people": [
            {
                "id": "test_person",
                "name": "Test",
                "twitter_handle": "test",
                "sources": [{"type": "twitter", "enabled": True}],
            }
        ]
    }
    p = tmp_path / "people.yml"
    p.write_text(yaml.dump(config), encoding="utf-8")

    people = load_config(p)
    assert people[0].rss_enabled is False
    assert people[0].rss_url == ""


# ---------------------------------------------------------------------------
# load_processed_ids / save_processed_ids
# ---------------------------------------------------------------------------

def test_load_processed_ids_new(tmp_path):
    """文件不存在时返回空集合。"""
    ids = load_processed_ids(tmp_path / "nonexistent.json")
    assert ids == set()


def test_load_processed_ids_existing(tmp_path):
    """正常加载已有 ID 列表。"""
    p = tmp_path / "processed_ids.json"
    p.write_text(json.dumps({"ids": ["123", "456"]}))

    ids = load_processed_ids(p)
    assert ids == {"123", "456"}


def test_load_processed_ids_malformed(tmp_path):
    """JSON 格式损坏时返回空集合，不抛出异常。"""
    p = tmp_path / "processed_ids.json"
    p.write_text("this is not valid json {{{{")

    ids = load_processed_ids(p)
    assert ids == set()


def test_save_and_reload_processed_ids(tmp_path):
    """save + load 往返一致。"""
    p = tmp_path / "processed_ids.json"
    original = {"aaa", "bbb", "ccc"}

    save_processed_ids(original, p)
    reloaded = load_processed_ids(p)

    assert reloaded == original


# ---------------------------------------------------------------------------
# parse_llm_output
# ---------------------------------------------------------------------------

def test_parse_llm_output_valid():
    """正常格式：提取 TITLE 和 SUMMARY。"""
    text = "TITLE: 关于 AGI 时间线的思考\nSUMMARY: Altman 认为 AGI 比预期更近。"
    result = parse_llm_output(text, original="The original tweet")

    assert result["title"] == "关于 AGI 时间线的思考"
    assert result["summary"] == "Altman 认为 AGI 比预期更近。"


def test_parse_llm_output_fallback():
    """格式完全不符时，返回原文 fallback（不跳过条目）。"""
    text = "Here is my translation: 这是翻译内容，但没有遵守格式。"
    original = "The original tweet text"

    result = parse_llm_output(text, original)

    assert result["title"] == "（原文）"
    assert result["summary"] == original


def test_parse_llm_output_partial_summary_only():
    """只有 SUMMARY 没有 TITLE 时，title 填充占位符。"""
    text = "SUMMARY: 这是摘要内容。"
    result = parse_llm_output(text, original="original")

    assert result["summary"] == "这是摘要内容。"
    assert result["title"] == "（无标题）"


def test_parse_llm_output_extra_prefix():
    """LLM 在 TITLE: 前加了多余的前缀行，仍能正确提取。"""
    text = textwrap.dedent("""\
        好的，这是翻译：
        TITLE: 核心标题
        SUMMARY: 核心摘要内容。
    """)
    result = parse_llm_output(text, original="original")

    assert result["title"] == "核心标题"
    assert result["summary"] == "核心摘要内容。"


# ---------------------------------------------------------------------------
# render_html
# ---------------------------------------------------------------------------

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


def test_render_html_with_entries():
    """有内容时，HTML 包含条目的 person_name 和 summary。"""
    entries = [
        TweetEntry(
            tweet_id="1",
            person_id="sam_altman",
            person_name="Sam Altman",
            original_text="Original tweet",
            tweet_url="https://twitter.com/sama/status/1",
            created_at="2026-03-22 09:00",
            title="关于 AGI 的思考",
            summary="这是中文摘要。",
        )
    ]

    html = render_html(entries, today="2026-03-22", templates_dir=TEMPLATES_DIR)

    assert "Sam Altman" in html
    assert "这是中文摘要。" in html
    assert "Original tweet" in html
    assert "https://twitter.com/sama/status/1" in html
    assert "今日无新内容" not in html


def test_render_html_empty():
    """无内容时，HTML 包含空状态提示，不包含条目结构。"""
    html = render_html([], today="2026-03-22", templates_dir=TEMPLATES_DIR)

    assert "今日无新内容" in html
    assert "Sam Altman" not in html


# ---------------------------------------------------------------------------
# fetch_tweets
# ---------------------------------------------------------------------------

def _make_tweet(id_="111", text="hello", created_at=None):
    """构造 mock 推文对象。"""
    from datetime import datetime
    t = MagicMock()
    t.id = id_
    t.text = text
    t.created_at = created_at or datetime(2026, 3, 22, 9, 0)
    return t


def test_fetch_tweets_happy_path():
    """正常情况：返回格式化的推文字典列表。"""
    mock_client = MagicMock()
    mock_client.get_user.return_value.data.id = "123"
    tweet = _make_tweet("111", "AGI is near")
    mock_client.get_users_tweets.return_value.data = [tweet]

    result = fetch_tweets("sama", mock_client)

    assert len(result) == 1
    assert result[0]["id"] == "111"
    assert result[0]["text"] == "AGI is near"
    assert "sama/status/111" in result[0]["url"]


def test_fetch_tweets_user_not_found():
    """用户不存在（data=None）时返回空列表，不抛出异常。"""
    mock_client = MagicMock()
    mock_client.get_user.return_value.data = None

    result = fetch_tweets("nonexistent_user", mock_client)

    assert result == []


def test_fetch_tweets_no_tweets():
    """用户存在但无推文（tweets_resp.data=None）时返回空列表。"""
    mock_client = MagicMock()
    mock_client.get_user.return_value.data.id = "123"
    mock_client.get_users_tweets.return_value.data = None

    result = fetch_tweets("sama", mock_client)

    assert result == []


def test_fetch_tweets_exception():
    """任何 API 异常都返回空列表，不向上抛出。"""
    mock_client = MagicMock()
    mock_client.get_user.side_effect = Exception("network error")

    result = fetch_tweets("sama", mock_client)

    assert result == []


# ---------------------------------------------------------------------------
# fetch_rss
# ---------------------------------------------------------------------------

def _make_feed(entries):
    """构造 feedparser.FeedParserDict 风格的 mock feed。"""
    feed = MagicMock()
    feed.entries = entries
    return feed


def _make_rss_entry(
    id_="https://example.com/post/1",
    summary="Hello world content",
    link="https://example.com/post/1",
    published_parsed=(2026, 3, 22, 9, 0, 0, 0, 0, 0),
    content=None,
):
    entry = MagicMock()
    entry.get = lambda key, default=None: {
        "id": id_,
        "link": link,
        "summary": summary,
        "published_parsed": published_parsed,
        "content": content,
    }.get(key, default)
    entry.id = id_
    entry.summary = summary
    entry.link = link
    entry.published_parsed = published_parsed
    entry.content = content or []
    return entry


def test_fetch_rss_happy_path():
    """正常情况：返回格式化的 RSS 条目列表。"""
    entry = _make_rss_entry(
        id_="https://blog.example.com/post/1",
        summary="AI is transforming the world.",
        link="https://blog.example.com/post/1",
    )
    mock_feed = _make_feed([entry])

    with patch("pipeline.feedparser.parse", return_value=mock_feed):
        result = fetch_rss("https://blog.example.com/rss")

    assert len(result) == 1
    assert result[0]["id"] == "https://blog.example.com/post/1"
    assert result[0]["text"] == "AI is transforming the world."
    assert result[0]["url"] == "https://blog.example.com/post/1"
    assert result[0]["created_at"] == "2026-03-22 09:00"


def test_fetch_rss_no_entries():
    """feed 无条目时返回空列表。"""
    mock_feed = _make_feed([])

    with patch("pipeline.feedparser.parse", return_value=mock_feed):
        result = fetch_rss("https://blog.example.com/rss")

    assert result == []


def test_fetch_rss_html_stripped():
    """正文中的 HTML 标签应被去除。"""
    entry = _make_rss_entry(
        id_="https://example.com/1",
        summary="<p>This is <strong>important</strong> news.</p>",
        link="https://example.com/1",
    )
    mock_feed = _make_feed([entry])

    with patch("pipeline.feedparser.parse", return_value=mock_feed):
        result = fetch_rss("https://example.com/rss")

    assert "<p>" not in result[0]["text"]
    assert "<strong>" not in result[0]["text"]
    assert "This is important news." in result[0]["text"]


def test_fetch_rss_exception():
    """任何异常都返回空列表，不向上抛出。"""
    with patch("pipeline.feedparser.parse", side_effect=Exception("connection error")):
        result = fetch_rss("https://blog.example.com/rss")

    assert result == []


# ---------------------------------------------------------------------------
# translate
# ---------------------------------------------------------------------------

def test_translate_success():
    """正常翻译：返回含 title、summary、original 的字典。"""
    mock_client = MagicMock()
    mock_content = MagicMock()
    mock_content.text = "TITLE: AGI 近了\nSUMMARY: Altman 认为 AGI 比预期更近。"
    mock_client.messages.create.return_value.content = [mock_content]

    tweet = {"id": "1", "text": "AGI is coming sooner than expected.", "created_at": "2026-03-22 09:00"}
    result = translate(tweet, mock_client)

    assert result["title"] == "AGI 近了"
    assert result["summary"] == "Altman 认为 AGI 比预期更近。"
    assert result["original"] == tweet["text"]


def test_translate_api_failure():
    """API 异常时返回 fallback dict，不向上抛出。"""
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = Exception("api timeout")

    tweet = {"id": "1", "text": "The original tweet text.", "created_at": "2026-03-22 09:00"}
    result = translate(tweet, mock_client)

    assert result["title"] == "（翻译失败）"
    assert result["summary"] == tweet["text"]
    assert result["original"] == tweet["text"]


# ---------------------------------------------------------------------------
# main — 环境变量缺失检查
# ---------------------------------------------------------------------------

def test_main_missing_twitter_token():
    """缺少 TWITTER_BEARER_TOKEN 时抛出 ValueError。"""
    with patch.dict("os.environ", {"TWITTER_BEARER_TOKEN": "", "ANTHROPIC_API_KEY": "sk-test"}, clear=False):
        # 确保 key 不存在
        import os
        env = {"ANTHROPIC_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="TWITTER_BEARER_TOKEN"):
                main()


def test_main_missing_anthropic_key():
    """缺少 ANTHROPIC_API_KEY 时抛出 ValueError。"""
    import os
    env = {"TWITTER_BEARER_TOKEN": "tok-test"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            main()
