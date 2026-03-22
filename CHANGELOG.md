# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0.0] - 2026-03-22

### Added
- README with GitHub Pages link, Fork setup guide, and local run instructions
- `gh-pages` branch deployment: static site now lives on a dedicated orphan branch, keeping `main` clean
- `.nojekyll` in `gh-pages` to prevent Jekyll processing

### Changed
- GitHub Actions workflow split into two steps: commit `processed_ids.json` to `main`, deploy HTML to `gh-pages`
- Redesigned page template: modern card layout with subtle shadows, hover effects, clean typography, and CSS custom properties
- Footer simplified to a single "历史存档" archive link
- `docs/` added to `.gitignore` — generated files are build artifacts, not source

### Removed
- `docs/.gitkeep`, `docs/index.html`, `docs/archive/` from `main` branch tracking

## [0.1.0.0] - 2026-03-22

### Added
- Initial pipeline: daily fetch of tweets from Sam Altman, Dario Amodei, and Andrej Karpathy via Twitter API Free Tier (tweepy)
- Claude Haiku translation with TITLE: / SUMMARY: format and robust fallback parsing for malformed LLM output
- Jinja2 HTML rendering to `docs/index.html` and `docs/archive/YYYY-MM-DD.html` (GitHub Pages compatible)
- Tweet deduplication via `processed_ids.json` committed to repo
- GitHub Actions daily cron at 01:00 UTC (09:00 BJT) with `workflow_dispatch` for manual testing
- Configurable people list via `people.yml` with per-person source enable/disable
- 21 pytest tests covering all core functions including fetch_tweets, translate, and main env-check paths
