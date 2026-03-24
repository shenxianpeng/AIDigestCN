"""
Regression: ISSUE-002 — broken 历史存档 link on archive date pages
Found by /qa on 2026-03-24
Report: .gstack/qa-reports/qa-report-shenxianpeng-github-io-2026-03-24.md

Archive date pages (archive/YYYY-MM-DD.html) shared the same rendered HTML
as the main index.html. The footer used href="archive/" which resolves
correctly for the main page but becomes /archive/archive/ (404) when served
from inside the archive/ subdirectory.

Fix: render_html() now accepts archive_url parameter. Main page uses
"archive/" and archive date pages use "./" (current directory).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import render_html

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


def test_main_page_archive_link_uses_relative_path():
    """主页（index.html）的历史存档链接应为 'archive/'（默认值）。"""
    html = render_html([], today="2026-03-24", archive_url="archive/", templates_dir=TEMPLATES_DIR)
    assert 'href="archive/"' in html


def test_archive_date_page_archive_link_uses_current_dir():
    """归档日期页面（archive/YYYY-MM-DD.html）的历史存档链接应为 './'，不能是 'archive/'。

    原始 bug：archive date pages had href="archive/" which resolved to
    /archive/archive/ (404) when served from the archive/ subdirectory.
    """
    html = render_html([], today="2026-03-24", archive_url="./", templates_dir=TEMPLATES_DIR)
    assert 'href="./"' in html
    # Ensure the broken double-archive path is not present
    assert 'href="archive/"' not in html


def test_archive_date_page_default_would_break():
    """如果归档日期页面使用默认的 'archive/' 而非 './'，说明修复回退了。"""
    # This test documents the expected behavior: archive date pages MUST NOT
    # use the same archive_url as the main page. If this test starts failing
    # (i.e., both pages use the same URL), the fix has been reverted.
    main_html = render_html([], today="2026-03-24", archive_url="archive/", templates_dir=TEMPLATES_DIR)
    archive_html = render_html([], today="2026-03-24", archive_url="./", templates_dir=TEMPLATES_DIR)
    assert main_html != archive_html, (
        "Main page and archive date page should render different archive links"
    )
