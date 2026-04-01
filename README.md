# AI 领袖动态 · AIDigestCN

> 每日自动抓取顶级 AI 领袖的推文，翻译成中文，发布到 GitHub Pages。
>
> *Daily digest of top AI leaders' tweets — automatically translated to Chinese.*

[![Daily Update](https://github.com/shenxianpeng/AIDigestCN/actions/workflows/daily.yml/badge.svg)](https://github.com/shenxianpeng/AIDigestCN/actions/workflows/daily.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/shenxianpeng/AIDigestCN?style=social)](https://github.com/shenxianpeng/AIDigestCN/stargazers)

**[→ 查看最新内容 / View Latest](https://shenxianpeng.github.io/AIDigestCN/)**

---

## 项目简介

AIDigestCN 是一个开箱即用的 GitHub 仓库模板，帮助中文读者无障碍获取全球顶级 AI 领袖的最新动态：

- 🤖 **自动抓取**：每天北京时间 09:00 自动运行，无需人工干预
- 🌏 **中文翻译**：使用 OpenAI gpt-4o-mini 将英文推文翻译成中文，保留原意
- 📰 **静态网站**：生成纯 HTML 站点，发布到 GitHub Pages，无服务器费用
- 🎨 **专业设计**：以人物为主角的阅读体验，支持深色/浅色模式
- 🔀 **一键 Fork**：Fork 仓库、配置两个密钥即可拥有自己的专属 AI 资讯站

---

## 当前关注人物

| 人物 | 角色 | Twitter |
|------|------|---------|
| Sam Altman | OpenAI CEO | [@sama](https://twitter.com/sama) |
| Dario Amodei | Anthropic CEO | [@DarioAmodei](https://twitter.com/DarioAmodei) |
| Andrej Karpathy | AI 研究者 / 前 Tesla AI | [@karpathy](https://twitter.com/karpathy) |
| Swyx | Latent Space 联创 | [@swyx](https://twitter.com/swyx) |
| Josh Woodward | Google Labs VP | [@joshwoodward](https://twitter.com/joshwoodward) |
| Kevin Weil | OpenAI CPO | [@kevinweil](https://twitter.com/kevinweil) |
| Peter Yang | Cursor 产品负责人 | [@petergyang](https://twitter.com/petergyang) |
| Amanda Askell | Anthropic 研究科学家 | [@AmandaAskell](https://twitter.com/AmandaAskell) |
| Cat Wu | Anthropic 工程师 | [@_catwu](https://twitter.com/_catwu) |
| Amjad Masad | Replit CEO | [@amasad](https://twitter.com/amasad) |
| Guillermo Rauch | Vercel CEO | [@rauchg](https://twitter.com/rauchg) |
| Alex Albert | Anthropic 开发者关系 | [@alexalbert__](https://twitter.com/alexalbert__) |
| Aaron Levie | Box CEO | [@levie](https://twitter.com/levie) |
| Garry Tan | Y Combinator CEO | [@garrytan](https://twitter.com/garrytan) |
| Matt Turck | FirstMark Capital 合伙人 | [@mattturck](https://twitter.com/mattturck) |
| Nikunj Kothari | a16z 合伙人 | [@nikunj](https://twitter.com/nikunj) |
| Peter Steinberger | PSPDFKit 创始人 | [@steipete](https://twitter.com/steipete) |
| Dan Shipper | Every 联创 & CEO | [@danshipper](https://twitter.com/danshipper) |
| Aditya Agarwal | Dropbox 前 CTO | [@adityaag](https://twitter.com/adityaag) |

完整配置见 [`people.yml`](people.yml)。

---

## Fork 使用（5 分钟上线）

### 第一步：Fork 仓库

点击页面右上角 **Fork** 按钮，将仓库 Fork 到你的 GitHub 账户。

### 第二步：配置 Secrets

在 **Settings → Secrets and variables → Actions → New repository secret** 中添加：

| Secret | 说明 | 获取方式 |
|--------|------|----------|
| `OPENAI_API_KEY` | OpenAI API Key，用于翻译推文 | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `TWITTER_AUTH_TOKEN` | Twitter 登录 Cookie（`auth_token` 字段） | 浏览器开发者工具 → Application → Cookies → `auth_token` |

> **说明：** 本项目通过 `tweeterpy` 使用 Twitter Guest Session 抓取推文，无需 Twitter Developer 账号，只需一个普通 Twitter 登录 Cookie。

### 第三步：开启 GitHub Pages

在 **Settings → Pages** 中：
- Source 选择 **Deploy from a branch**
- Branch 选择 **`gh-pages`**，目录选择 **`/ (root)`**
- 点击 **Save**

### 第四步：手动触发验证

在 **Actions → Daily AI Leaders Update → Run workflow** 手动触发一次，验证端到端流程正常运行。

之后每天 **北京时间 09:00**（UTC 01:00）自动更新。

---

## 修改关注列表

编辑 [`people.yml`](people.yml)，添加或删除条目：

```yaml
people:
  - id: sama                       # 唯一 ID（字母/下划线）
    name: Sam Altman               # 显示名称
    role: OpenAI CEO               # 角色/职位（显示在卡片上）
    twitter_handle: sama           # Twitter @handle（不含 @）
    sources:
      - type: twitter
        enabled: true              # false 则跳过该数据源
```

修改后提交到 `main` 分支，下次自动运行时生效。

---

## 本地运行

```bash
# 克隆并进入目录
git clone https://github.com/YOUR_USERNAME/AIDigestCN.git
cd AIDigestCN

# 创建虚拟环境并安装依赖
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 设置环境变量
export OPENAI_API_KEY=sk-...
export TWITTER_AUTH_TOKEN=...

# 运行 pipeline（生成 docs/index.html）
python src/pipeline.py

# 运行测试
pytest tests/ -v
```

---

## 环境变量参考

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENAI_API_KEY` | *(必填)* | OpenAI API Key |
| `TWITTER_AUTH_TOKEN` | *(必填)* | Twitter auth_token Cookie |
| `OPENAI_MODEL` | `gpt-4o-mini` | 翻译使用的模型 |
| `TWEETS_PER_USER` | `40` | 每人每次抓取的推文数量上限 |
| `TRANSLATE_BATCH_SIZE` | `10` | 每批翻译的推文数量（减少 API 调用） |

---

## 项目结构

```
AIDigestCN/
├── .github/workflows/daily.yml   # GitHub Actions 自动化工作流
├── src/pipeline.py               # 核心 pipeline：抓取 → 翻译 → 渲染
├── scripts/fetch_avatars.py      # 下载并缓存头像
├── templates/
│   ├── day.html.j2               # 每日主页模板
│   └── archive.html.j2           # 历史存档页模板
├── tests/                        # pytest 测试套件
├── people.yml                    # 关注的 AI 领袖列表（可配置）
├── processed_ids.json            # 推文去重状态（自动更新）
├── requirements.txt              # Python 依赖
├── DESIGN.md                     # UI/UX 设计系统规范
└── CHANGELOG.md                  # 版本变更记录
```

---

## 常见问题

**Q: 为什么使用 Cookie 而不是 Twitter API？**
> Twitter API 免费套餐限制极为严格（每月仅 1500 条读取），无法满足每日抓取需求。使用 Guest Session Cookie 是目前最实用的免费方案，但请注意遵守 Twitter 使用条款。

**Q: OpenAI 费用大概多少？**
> gpt-4o-mini 非常便宜。20 人 × 每人每天约 3 条推文 × 每条约 300 tokens = 约 0.003 美元/天，每月不到 0.1 美元。

**Q: 可以换其他翻译服务吗？**
> 可以。修改 `src/pipeline.py` 中的 `translate_batch()` 函数，替换 OpenAI 调用即可。

**Q: 如何添加/删除关注的人物？**
> 直接编辑 `people.yml`，提交到 `main` 分支后下次运行自动生效。

---

## 贡献

欢迎提交 Issue 和 Pull Request！建议包括：
- 新增高质量 AI 领袖到默认列表
- 翻译质量改进
- UI/UX 优化
- 文档改善

---

## License

[MIT](LICENSE) © [shenxianpeng](https://github.com/shenxianpeng)
