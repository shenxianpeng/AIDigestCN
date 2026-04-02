# 安全政策

## 支持版本

本项目只维护最新版本（`main` 分支）。

## 报告安全漏洞

**请勿在公开 Issue 中报告安全漏洞。**

如果你发现了安全漏洞，请通过以下方式私下报告：

1. 发送邮件至项目维护者（可在 GitHub 个人主页找到联系方式）
2. 或通过 GitHub 的 [Private Vulnerability Reporting](https://github.com/shenxianpeng/AIDigestCN/security/advisories/new) 功能报告

请在报告中包含：
- 漏洞描述
- 复现步骤
- 潜在影响范围
- 建议的修复方案（如有）

我们承诺在 **7 个工作日**内回复，并在修复后通知你。

## 安全注意事项

- **不要**将 `OPENAI_API_KEY`、`TWITTER_AUTH_TOKEN` 等密钥提交到版本控制
- 本项目使用 GitHub Actions Secrets 管理所有敏感信息
- `processed_ids.json` 仅包含推文 ID，不含任何敏感数据
