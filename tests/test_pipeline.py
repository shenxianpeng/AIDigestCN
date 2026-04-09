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
  test_fetch_tweets_includes_retweets_with_context — 转推包含上下文
  test_fetch_tweets_skip_replies      — 过滤回复
  test_translate_success              — 正常翻译返回 title/summary/original
  test_translate_api_failure          — API 异常时返回 fallback dict
  test_main_missing_github_token      — 缺少 GITHUB_TOKEN 时抛出 ValueError
"""

import json
import os
import sys
import textwrap
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# 把 src/ 加入 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import (
    TweetEntry,
    _is_within_lookback,
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



def test_save_processed_ids_with_none(tmp_path):
    """None 值应被过滤，不引发 TypeError。"""
    p = tmp_path / "processed_ids.json"
    ids_with_none = {"aaa", None, "bbb"}

    save_processed_ids(ids_with_none, p)
    reloaded = load_processed_ids(p)

    assert None not in reloaded
    assert reloaded == {"aaa", "bbb"}


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


def test_parse_llm_output_with_context():
    """含 CONTEXT: 字段时正确提取引用内容的中文翻译。"""
    text = textwrap.dedent("""\
        TITLE: 关于大胆程度的理论
        SUMMARY: Swyx 转发了关于大胆程度的思考。
        CONTEXT: 生活会在你大胆的程度上与你相遇。
    """)
    result = parse_llm_output(text, original="original")

    assert result["title"] == "关于大胆程度的理论"
    assert result["summary"] == "Swyx 转发了关于大胆程度的思考。"
    assert result["context_translated"] == "生活会在你大胆的程度上与你相遇。"


def test_parse_llm_output_empty_context():
    """CONTEXT: 为空时 context_translated 为空字符串。"""
    text = "TITLE: 标题\nSUMMARY: 摘要。\nCONTEXT:"
    result = parse_llm_output(text, original="original")

    assert result["context_translated"] == ""


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
    assert "今日暂无新内容" not in html


def test_render_html_empty():
    """无内容时，HTML 包含空状态提示，不包含条目结构。"""
    html = render_html([], today="2026-03-22", templates_dir=TEMPLATES_DIR)

    assert "今日暂无新内容" in html
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
# _is_within_lookback
# ---------------------------------------------------------------------------

from pipeline import LOOKBACK_DAYS


def test_is_within_lookback_today():
    """今天的推文应在窗口内。"""
    today = date.today().isoformat()
    assert _is_within_lookback(today) is True


def test_is_within_lookback_recent_with_time():
    """'YYYY-MM-DD HH:MM' 格式的近期推文应在窗口内。"""
    today = date.today().isoformat() + " 12:00"
    assert _is_within_lookback(today) is True


def test_is_within_lookback_old_tweet():
    """超出窗口期的旧推文应返回 False。"""
    old = (date.today() - timedelta(days=LOOKBACK_DAYS + 1)).isoformat()
    assert _is_within_lookback(old) is False


def test_is_within_lookback_empty_date():
    """无日期信息时应返回 True（不过滤，以免丢失数据）。"""
    assert _is_within_lookback("") is True
    assert _is_within_lookback(None) is True


def test_is_within_lookback_boundary():
    """恰好在窗口边界当天的推文应视为在窗口内。"""
    boundary = (date.today() - timedelta(days=LOOKBACK_DAYS)).isoformat()
    assert _is_within_lookback(boundary) is True


# ---------------------------------------------------------------------------
# fetch_tweets
# ---------------------------------------------------------------------------

# Twitter API 日期格式（用于测试 mock 数据）
_TWITTER_DATE_FMT = "%a %b %d %H:%M:%S +0000 %Y"


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


def test_fetch_tweets_uses_configured_total():
    """抓取条数应使用 FETCH_TWEETS_PER_USER 配置。"""
    mock_client = MagicMock()
    mock_tweet = _make_tweet_item("111", "AGI is near")

    with (
        patch("pipeline.FETCH_TWEETS_PER_USER", 40),
        patch("pipeline.Tweet", return_value=mock_tweet),
    ):
        mock_client.get_user_tweets.return_value = {"data": [MagicMock()]}
        fetch_tweets("sama", mock_client)

    mock_client.get_user_tweets.assert_called_once_with("sama", total=40)


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


def test_fetch_tweets_includes_retweets_with_context():
    """SKIP_PURE_REPOSTS=False 时转推应被包含，并提取 context_author 和 context_text。"""
    mock_client = MagicMock()
    mock_tweet = _make_tweet_item("111", "RT @someone: some text")

    with patch("pipeline.Tweet", return_value=mock_tweet), \
         patch("pipeline.SKIP_PURE_REPOSTS", False):
        mock_client.get_user_tweets.return_value = {"data": [MagicMock()]}
        result = fetch_tweets("sama", mock_client)

    assert len(result) == 1
    assert result[0]["text"] == "some text"
    assert result[0]["context_author"] == "someone"
    assert result[0]["context_text"] == "some text"
    assert result[0]["is_repost"] is True


def test_fetch_tweets_skip_pure_reposts_enabled():
    """SKIP_PURE_REPOSTS=True（默认）时纯转推应被过滤，不返回结果。"""
    mock_client = MagicMock()
    mock_tweet = _make_tweet_item("111", "RT @someone: some text")

    with patch("pipeline.Tweet", return_value=mock_tweet), \
         patch("pipeline.SKIP_PURE_REPOSTS", True):
        mock_client.get_user_tweets.return_value = {"data": [MagicMock()]}
        result = fetch_tweets("sama", mock_client)

    assert result == []


def test_fetch_tweets_skip_pure_reposts_disabled_keeps_repost():
    """SKIP_PURE_REPOSTS=False 时纯转推应被保留并标记 is_repost=True。"""
    mock_client = MagicMock()
    mock_tweet = _make_tweet_item("222", "RT @openai: Big news today")

    with patch("pipeline.Tweet", return_value=mock_tweet), \
         patch("pipeline.SKIP_PURE_REPOSTS", False):
        mock_client.get_user_tweets.return_value = {"data": [MagicMock()]}
        result = fetch_tweets("sama", mock_client)

    assert len(result) == 1
    assert result[0]["is_repost"] is True
    assert result[0]["context_author"] == "openai"


def test_fetch_tweets_skip_replies():
    """有 in_reply_to_status_id_str 的推文（回复）应被过滤。"""
    mock_client = MagicMock()
    mock_tweet = _make_tweet_item("111", "This is a reply", in_reply_to_status_id_str="999")

    with patch("pipeline.Tweet", return_value=mock_tweet):
        mock_client.get_user_tweets.return_value = {"data": [MagicMock()]}
        result = fetch_tweets("sama", mock_client)

    assert result == []


def test_fetch_tweets_full_text_is_list_captures_context():
    """tweeterpy 返回 full_text 为列表时，第二个元素应被捕获为引用上下文。"""
    mock_client = MagicMock()
    mock_tweet = _make_tweet_item("111", ["AGI is near", "quoted tweet text"])

    with patch("pipeline.Tweet", return_value=mock_tweet):
        mock_client.get_user_tweets.return_value = {"data": [MagicMock()]}
        result = fetch_tweets("sama", mock_client)

    assert len(result) == 1
    assert result[0]["text"] == "AGI is near"
    assert result[0]["context_text"] == "quoted tweet text"
    assert result[0]["is_repost"] is False


def test_fetch_tweets_full_text_list_is_retweet():
    """full_text 为列表且第一个元素是转推时，SKIP_PURE_REPOSTS=False 时应提取原始内容并标记为转推。"""
    mock_client = MagicMock()
    mock_tweet = _make_tweet_item("111", ["RT @someone: original", "original"])

    with patch("pipeline.Tweet", return_value=mock_tweet), \
         patch("pipeline.SKIP_PURE_REPOSTS", False):
        mock_client.get_user_tweets.return_value = {"data": [MagicMock()]}
        result = fetch_tweets("sama", mock_client)

    assert len(result) == 1
    assert result[0]["is_repost"] is True
    assert result[0]["context_author"] == "someone"


def test_fetch_tweets_in_reply_to_is_list():
    """in_reply_to_status_id_str 为列表时，应跳过该回复推文。"""
    mock_client = MagicMock()
    mock_tweet = _make_tweet_item("111", "This is a reply", in_reply_to_status_id_str=["999"])

    with patch("pipeline.Tweet", return_value=mock_tweet):
        mock_client.get_user_tweets.return_value = {"data": [MagicMock()]}
        result = fetch_tweets("sama", mock_client)

    assert result == []



def test_fetch_tweets_skip_none_tweet_id():
    """rest_id 和 id_str 均为 None 时，该推文应被跳过，不加入结果。"""
    mock_client = MagicMock()
    mock_tweet = _make_tweet_item(None, "AGI is near")

    with patch("pipeline.Tweet", return_value=mock_tweet):
        mock_client.get_user_tweets.return_value = {"data": [MagicMock()]}
        result = fetch_tweets("sama", mock_client)

    assert result == []


# ---------------------------------------------------------------------------
# translate
# ---------------------------------------------------------------------------

def test_translate_success():
    """正常翻译：返回含 title、summary、original 的字典。"""
    tweet = {"id": "1", "text": "AGI is coming sooner than expected.", "created_at": "2026-03-22 09:00"}

    with patch("pipeline._call_openai", return_value="[1]\nTITLE: AGI 近了\nSUMMARY: Altman 认为 AGI 比预期更近。"):
        result = translate(tweet, "fake-token")

    assert result["title"] == "AGI 近了"
    assert result["summary"] == "Altman 认为 AGI 比预期更近。"
    assert result["original"] == tweet["text"]


def test_translate_api_failure():
    """API 异常时返回 fallback dict，不向上抛出。"""
    tweet = {"id": "1", "text": "The original tweet text.", "created_at": "2026-03-22 09:00"}

    with patch("pipeline._call_openai", side_effect=Exception("api timeout")):
        result = translate(tweet, "fake-token")

    assert result["title"] == "（翻译失败）"
    assert result["summary"] == tweet["text"]
    assert result["original"] == tweet["text"]


# ---------------------------------------------------------------------------
# main — 环境变量缺失检查
# ---------------------------------------------------------------------------

def test_main_missing_openai_api_key():
    """缺少 OPENAI_API_KEY 时抛出 ValueError。"""
    import os
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            main()


# ---------------------------------------------------------------------------
# main — first run vs. subsequent run lookback behavior
# ---------------------------------------------------------------------------

def _make_main_mocks(
    tmp_path,
    people_yml,
    processed_ids_content,
    tweet_items,
):
    """Helper: write config files and return (people_file, ids_file, mock_twitter)."""
    people_file = tmp_path / "people.yml"
    people_file.write_text(yaml.dump(people_yml), encoding="utf-8")
    ids_file = tmp_path / "processed_ids.json"
    ids_file.write_text(json.dumps({"ids": processed_ids_content}))

    mock_twitter = MagicMock()
    mock_twitter.get_user_tweets.return_value = {"data": [MagicMock() for _ in tweet_items]}

    return people_file, ids_file, mock_twitter


def test_main_first_run_processes_old_tweets(tmp_path):
    """首次运行（processed_ids 为空）时，超出 lookback 窗口的旧推文也应被处理。"""
    people_yml = {
        "people": [{
            "id": "test_user",
            "name": "Test User",
            "twitter_handle": "testuser",
            "sources": [{"type": "twitter", "enabled": True}],
        }]
    }

    # Tweet created well outside the lookback window
    old_date = (date.today() - timedelta(days=LOOKBACK_DAYS + 10)).strftime(
        _TWITTER_DATE_FMT
    )
    mock_tweet = _make_tweet_item("999", "Old but valid tweet", created_at=old_date)

    people_file, ids_file, mock_twitter = _make_main_mocks(
        tmp_path,
        people_yml,
        processed_ids_content=[],  # ← 空：首次运行
        tweet_items=[mock_tweet],
    )
    docs_dir = tmp_path / "docs"
    archive_dir = docs_dir / "archive"

    with (
        patch("pipeline.Tweet", return_value=mock_tweet),
        patch("pipeline.PEOPLE_FILE", people_file),
        patch("pipeline.PROCESSED_IDS_FILE", ids_file),
        patch("pipeline.DOCS_DIR", docs_dir),
        patch("pipeline.ARCHIVE_DIR", archive_dir),
        patch("pipeline.TEMPLATES_DIR", TEMPLATES_DIR),
        patch("pipeline.TweeterPy", return_value=mock_twitter),
        patch("pipeline._call_openai", return_value="[1]\nTITLE: 标题\nSUMMARY: 摘要"),
        patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}),
    ):
        main()

    # processed_ids should now contain the old tweet's ID
    saved = load_processed_ids(ids_file)
    assert "999" in saved, "首次运行应处理超出 lookback 窗口的旧推文"


def test_main_subsequent_run_skips_old_tweets(tmp_path):
    """后续运行时，超出 lookback 窗口的旧推文应被跳过。"""
    people_yml = {
        "people": [{
            "id": "test_user",
            "name": "Test User",
            "twitter_handle": "testuser",
            "sources": [{"type": "twitter", "enabled": True}],
        }]
    }

    old_date = (date.today() - timedelta(days=LOOKBACK_DAYS + 10)).strftime(
        _TWITTER_DATE_FMT
    )
    mock_tweet = _make_tweet_item("888", "Old tweet on subsequent run", created_at=old_date)

    people_file, ids_file, mock_twitter = _make_main_mocks(
        tmp_path,
        people_yml,
        processed_ids_content=["111"],  # ← 非空：后续运行
        tweet_items=[mock_tweet],
    )
    docs_dir = tmp_path / "docs"
    archive_dir = docs_dir / "archive"

    with (
        patch("pipeline.Tweet", return_value=mock_tweet),
        patch("pipeline.PEOPLE_FILE", people_file),
        patch("pipeline.PROCESSED_IDS_FILE", ids_file),
        patch("pipeline.DOCS_DIR", docs_dir),
        patch("pipeline.ARCHIVE_DIR", archive_dir),
        patch("pipeline.TEMPLATES_DIR", TEMPLATES_DIR),
        patch("pipeline.TweeterPy", return_value=mock_twitter),
        patch("pipeline._call_openai", return_value="[1]\nTITLE: 标题\nSUMMARY: 摘要"),
        patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}),
    ):
        main()

    # The old tweet should NOT have been added to processed_ids
    saved = load_processed_ids(ids_file)
    assert "888" not in saved, "后续运行应跳过超出 lookback 窗口的旧推文"


# ---------------------------------------------------------------------------
# main — TWITTER_AUTH_TOKEN 认证行为
# ---------------------------------------------------------------------------

def test_main_calls_generate_session_with_auth_token(tmp_path):
    """设置 TWITTER_AUTH_TOKEN 时，main() 应调用 generate_session(auth_token=...)。"""
    people_yml = {
        "people": [{
            "id": "test_user",
            "name": "Test User",
            "twitter_handle": "testuser",
            "sources": [{"type": "twitter", "enabled": True}],
        }]
    }
    people_file = tmp_path / "people.yml"
    people_file.write_text(yaml.dump(people_yml), encoding="utf-8")
    ids_file = tmp_path / "processed_ids.json"
    ids_file.write_text(json.dumps({"ids": []}))

    mock_twitter = MagicMock()
    mock_twitter.get_user_tweets.return_value = {"data": []}

    docs_dir = tmp_path / "docs"
    archive_dir = docs_dir / "archive"

    with (
        patch("pipeline.PEOPLE_FILE", people_file),
        patch("pipeline.PROCESSED_IDS_FILE", ids_file),
        patch("pipeline.DOCS_DIR", docs_dir),
        patch("pipeline.ARCHIVE_DIR", archive_dir),
        patch("pipeline.TEMPLATES_DIR", TEMPLATES_DIR),
        patch("pipeline.TweeterPy", return_value=mock_twitter),
        patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key", "TWITTER_AUTH_TOKEN": "my-secret-token"}),
    ):
        main()

    mock_twitter.generate_session.assert_called_once_with(auth_token="my-secret-token")


def test_main_no_auth_token_skips_generate_session(tmp_path):
    """未设置 TWITTER_AUTH_TOKEN 时，main() 不应调用 generate_session。"""
    people_yml = {
        "people": [{
            "id": "test_user",
            "name": "Test User",
            "twitter_handle": "testuser",
            "sources": [{"type": "twitter", "enabled": True}],
        }]
    }
    people_file = tmp_path / "people.yml"
    people_file.write_text(yaml.dump(people_yml), encoding="utf-8")
    ids_file = tmp_path / "processed_ids.json"
    ids_file.write_text(json.dumps({"ids": []}))

    mock_twitter = MagicMock()
    mock_twitter.get_user_tweets.return_value = {"data": []}

    docs_dir = tmp_path / "docs"
    archive_dir = docs_dir / "archive"

    # Remove TWITTER_AUTH_TOKEN from env entirely
    env_without_token = {k: v for k, v in os.environ.items() if k != "TWITTER_AUTH_TOKEN"}
    env_without_token["OPENAI_API_KEY"] = "fake-key"

    with (
        patch("pipeline.PEOPLE_FILE", people_file),
        patch("pipeline.PROCESSED_IDS_FILE", ids_file),
        patch("pipeline.DOCS_DIR", docs_dir),
        patch("pipeline.ARCHIVE_DIR", archive_dir),
        patch("pipeline.TEMPLATES_DIR", TEMPLATES_DIR),
        patch("pipeline.TweeterPy", return_value=mock_twitter),
        patch.dict(os.environ, env_without_token, clear=True),
    ):
        main()

    mock_twitter.generate_session.assert_not_called()
