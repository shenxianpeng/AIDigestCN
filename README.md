# AI 领袖动态

每日自动抓取顶级 AI 领袖的推文，翻译成中文，发布到 GitHub Pages。

**[→ 查看最新内容](https://shenxianpeng.github.io/new-prod-01/)**

当前关注：Sam Altman · Dario Amodei · Andrej Karpathy

---

## Fork 使用

**1. 添加 Secrets**

在 **Settings → Secrets and variables → Actions** 中添加：

| Secret | 说明 |
|--------|------|
| `TWITTER_BEARER_TOKEN` | Twitter API v2 Bearer Token（免费套餐即可） |
| `ANTHROPIC_API_KEY` | Anthropic API Key |

**2. 开启 GitHub Pages**

在 **Settings → Pages** 中选择 `gh-pages` 分支、根目录（`/`）作为发布源。

**3. 手动触发一次**

在 **Actions → Daily AI Leaders Update → Run workflow** 验证端到端流程。

之后每天北京时间 09:00 自动更新。

---

## 修改关注列表

编辑 [`people.yml`](people.yml)：

```yaml
people:
  - id: sama
    name: Sam Altman
    twitter_handle: sama
    sources:
      - type: twitter
        enabled: true
```

## 本地运行

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export TWITTER_BEARER_TOKEN=xxx ANTHROPIC_API_KEY=xxx
python src/pipeline.py
# 生成到 docs/index.html
```
