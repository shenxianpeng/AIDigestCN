"""
pytest 测试 — pipeline.py 核心函数

测试覆盖（见测试计划）：
  test_load_config_valid              — 正常解析 people.yml
  test_load_config_twitter_disabled   — enabled:false 时 twitter_enabled=False
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
  test_fetch_tweets_skip_retweets     — 过滤转推
  test_fetch_tweets_skip_replies      — 过滤回复
  test_translate_success              — 正常翻译返回 title/summary/original
  test_translate_api_failure          — API 异常时返回 fallback dict
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
    _parse_twitter_date,
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
# _parse_twitter_date
# ---------------------------------------------------------------------------

def test_parse_twitter_date_valid():
    """正常格式的 Twitter 日期字符串应正确转换。"""
    result = _parse_twitter_date("Mon Mar 22 09:00:00 +0000 2026")
    assert result == "2026-03-22 09:00"


def test_parse_twitter_date_invalid():
    """无效格式时返回空字符串，不抛出异常。"""
    assert _parse_twitter_date("not a date") == ""
    assert _parse_twitter_date(None) == ""


# ---------------------------------------------------------------------------
# fetch_tweets
# ---------------------------------------------------------------------------

def _make_tweet_item(
    rest_id="111",
    full_text="hello",
    created_at="Mon Mar 22 09:00:00 +0000 2026",
    in_reply_to_status_id_str=None,
    screen_name="sama",
):
    """构造 mock Tweet 对象（模拟 tweeterpy.util.Tweet）。"""
    tweet = MagicMock()
    tweet.rest_id = rest_id
    tweet.id_str = rest_id
    tweet.full_text = full_text
    tweet.created_at = created_at
    tweet.in_reply_to_status_id_str = in_reply_to_status_id_str
    tweet.screen_name = screen_name
    tweet.tweet_url = f"https://x.com/{screen_name}/status/{rest_id}"
    return tweet


def test_fetch_tweets_happy_path():
    """正常情况：返回格式化的推文字典列表。"""
    mock_client = MagicMock()
    mock_tweet = _make_tweet_item("111", "AGI is near")

    with patch("pipeline.Tweet", return_value=mock_tweet):
        mock_client.get_user_tweets.return_value = {"data": [MagicMock()]}
        result = fetch_tweets("sama", mock_client)

    assert len(result) == 1
    assert result[0]["id"] == "111"
    assert result[0]["text"] == "AGI is near"
    assert "sama/status/111" in result[0]["url"]


def test_fetch_tweets_user_not_found():
    """响应为空时返回空列表，不抛出异常。"""
    mock_client = MagicMock()
    mock_client.get_user_tweets.return_value = None

    result = fetch_tweets("nonexistent_user", mock_client)

    assert result == []


def test_fetch_tweets_no_tweets():
    """data 列表为空时返回空列表。"""
    mock_client = MagicMock()
    mock_client.get_user_tweets.return_value = {"data": []}

    result = fetch_tweets("sama", mock_client)

    assert result == []


def test_fetch_tweets_exception():
    """任何 API 异常都返回空列表，不向上抛出。"""
    mock_client = MagicMock()
    mock_client.get_user_tweets.side_effect = Exception("network error")

    result = fetch_tweets("sama", mock_client)

    assert result == []


def test_fetch_tweets_skip_retweets():
    """以 'RT @' 开头的推文应被过滤。"""
    mock_client = MagicMock()
    mock_tweet = _make_tweet_item("111", "RT @someone: some text")

    with patch("pipeline.Tweet", return_value=mock_tweet):
        mock_client.get_user_tweets.return_value = {"data": [MagicMock()]}
        result = fetch_tweets("sama", mock_client)

    assert result == []


def test_fetch_tweets_skip_replies():
    """有 in_reply_to_status_id_str 的推文（回复）应被过滤。"""
    mock_client = MagicMock()
    mock_tweet = _make_tweet_item("111", "This is a reply", in_reply_to_status_id_str="999")

    with patch("pipeline.Tweet", return_value=mock_tweet):
        mock_client.get_user_tweets.return_value = {"data": [MagicMock()]}
        result = fetch_tweets("sama", mock_client)

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

def test_main_missing_anthropic_key():
    """缺少 ANTHROPIC_API_KEY 时抛出 ValueError。"""
    import os
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            main()
