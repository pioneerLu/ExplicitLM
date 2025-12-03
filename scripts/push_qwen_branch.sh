#!/bin/bash
# 推送 Qwen_branch 分支到 GitHub

cd "$(dirname "$0")/.."

echo "=========================================="
echo "🚀 推送 Qwen_branch 分支到 GitHub"
echo "=========================================="
echo ""
echo "远程仓库: https://github.com/pioneerLu/ExplicitLM.git"
echo "分支: Qwen_branch"
echo ""

# 检查当前分支
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "Qwen_branch" ]; then
    echo "⚠️  当前不在 Qwen_branch 分支，切换到 Qwen_branch..."
    git checkout Qwen_branch
fi

# 检查远程仓库
REMOTE_URL=$(git remote get-url origin)
if [[ ! "$REMOTE_URL" == *"pioneerLu"* ]]; then
    echo "设置远程仓库为你的Fork..."
    git remote set-url origin https://github.com/pioneerLu/ExplicitLM.git
fi

echo "当前状态:"
git status --short | head -10
echo ""

# 检查是否有未提交的更改
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  检测到未提交的更改，请先提交："
    echo "   git add ."
    echo "   git commit -m 'your message'"
    exit 1
fi

echo "开始推送..."
echo ""

# 尝试推送
if git push -u origin Qwen_branch; then
    echo ""
    echo "=========================================="
    echo "✅ 推送成功！"
    echo "=========================================="
    echo ""
    echo "查看代码: https://github.com/pioneerLu/ExplicitLM/tree/Qwen_branch"
    echo "创建PR: https://github.com/pioneerLu/ExplicitLM/compare/main...Qwen_branch"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ 推送失败"
    echo "=========================================="
    echo ""
    echo "需要身份验证，请："
    echo "1. 输入用户名: pioneerLu"
    echo "2. 输入密码: 你的GitHub Personal Access Token"
    echo ""
    echo "如果没有Token，请访问: https://github.com/settings/tokens"
    echo "创建新token，勾选 'repo' 权限"
    echo ""
    echo "或者使用SSH方式（如果已配置SSH密钥）："
    echo "  git remote set-url origin git@github.com:pioneerLu/ExplicitLM.git"
    echo "  git push -u origin Qwen_branch"
    echo ""
fi

