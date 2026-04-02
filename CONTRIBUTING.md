# 贡献指南

感谢你对 **AIDigestCN** 的兴趣！我们欢迎任何形式的贡献——无论是修复 Bug、改进翻译质量、优化 UI，还是新增高质量 AI 人物。

---

## 目录

- [行为准则](#行为准则)
- [如何贡献](#如何贡献)
- [开发环境搭建](#开发环境搭建)
- [代码规范](#代码规范)
- [提交 Pull Request](#提交-pull-request)
- [新增 AI 人物](#新增-ai-人物)
- [报告 Bug](#报告-bug)
- [提出新功能](#提出新功能)

---

## 行为准则

参与本项目即表示你同意遵守友好、尊重和包容的社区规范。请保持专业，尊重不同观点。

---

## 如何贡献

### 最受欢迎的贡献类型

| 类型 | 描述 |
|------|------|
| 🐛 Bug 修复 | 修复 pipeline、模板或工作流中的问题 |
| ✨ 新功能 | 添加新数据源、新 UI 功能 |
| 🌐 翻译质量 | 改进提示词，提升翻译准确性和流畅度 |
| 👤 新增人物 | 向 `people.yml` 提交高质量 AI 领袖 |
| 📖 文档改善 | 修正文档错误、补充说明 |
| 🎨 UI/UX 优化 | 改进模板设计（遵循 `DESIGN.md` 规范） |

---

## 开发环境搭建

```bash
# 1. Fork 并克隆仓库
git clone https://github.com/YOUR_USERNAME/AIDigestCN.git
cd AIDigestCN

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行测试，确保环境正常
pytest tests/ -v
```

### 环境变量（本地开发可选）

```bash
export OPENAI_API_KEY=sk-...          # OpenAI API Key（翻译功能需要）
export TWITTER_AUTH_TOKEN=...         # Twitter auth_token（抓取推文需要）
```

如果只是修改模板或文档，无需配置上述变量。

---

## 代码规范

- **Python**：遵循 PEP 8，函数和类需有 docstring
- **HTML 模板**：遵循 `DESIGN.md` 设计系统规范（强调色、字体、间距等）
- **YAML**：使用 2 空格缩进，保持与 `people.yml` 风格一致
- **提交信息**：使用中文或英文均可，推荐格式：`feat: 新增...` / `fix: 修复...` / `docs: 更新...`

---

## 提交 Pull Request

1. **Fork** 本仓库到你的 GitHub 账户
2. 从 `main` 创建功能分支：`git checkout -b feat/your-feature`
3. 做出修改，并添加必要的测试
4. 运行完整测试套件：`pytest tests/ -v`
5. 提交代码并 push 到你的 fork
6. 在 GitHub 上打开 Pull Request，填写模板中的所有字段
7. 等待 review，根据反馈进行修改

### PR 检查清单

- [ ] 代码通过 `pytest tests/ -v`
- [ ] 新功能有对应测试
- [ ] UI 改动符合 `DESIGN.md` 规范
- [ ] 更新了相关文档（如适用）

---

## 新增 AI 人物

向 `people.yml` 新增人物时，请确保：

1. **知名度**：该人物在 AI 领域具有行业影响力（CEO、核心研究员、知名 KOL 等）
2. **活跃度**：Twitter 账号仍然活跃（近 30 天内有发帖）
3. **格式正确**：

```yaml
- id: unique_id              # 唯一 ID，使用小写字母和下划线
  name: 显示名称
  role: 职位/角色（中文）
  twitter_handle: handle     # 不含 @
  sources:
    - type: twitter
      enabled: true
```

4. 在 PR 描述中说明**为什么**这个人物值得关注

---

## 报告 Bug

请使用 [Bug 报告模板](.github/ISSUE_TEMPLATE/bug_report.md) 提交 Issue，并尽量提供：

- 复现步骤
- 预期行为 vs 实际行为
- 错误日志（GitHub Actions 日志或本地终端输出）
- Python 版本和操作系统

---

## 提出新功能

请使用 [功能请求模板](.github/ISSUE_TEMPLATE/feature_request.md) 提交 Issue，描述：

- 要解决的问题或场景
- 建议的解决方案
- 备选方案（如有）

---

## 有问题？

- 在 [Issues](https://github.com/shenxianpeng/AIDigestCN/issues) 中提问
- 查阅 [README.md](README.md) 和 [DESIGN.md](DESIGN.md)

再次感谢你的贡献！🙏
