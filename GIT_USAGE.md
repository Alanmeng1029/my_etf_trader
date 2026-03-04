# Git 版本控制使用指南

本文档记录如何使用 Git 管理 my_etf 项目的代码版本。

## 快速入门

### 查看当前状态
```bash
git status
```

### 提交代码更改
```bash
# 1. 添加修改的文件到暂存区
git add <文件名>
git add .                 # 当前目录所有文件
git add -A                # 所有修改

# 2. 提交更改
git commit -m "提交信息"

# 3. 推送到远程仓库（如果有配置）
git push origin master
```

## 日常工作流程

### 修改后的标准流程
```bash
# 1. 修改代码文件
# ...进行代码编辑...

# 2. 查看修改状态
git status
git diff <文件名>          # 查看具体修改内容

# 3. 添加到暂存区
git add src/my_etf/config.py

# 4. 提交
git commit -m "优化XGBoost模型参数"

# 5. 推送到远程
git push origin master
```

### 查看提交历史
```bash
git log --oneline          # 简洁显示
git log --graph            # 图形化显示
git log -p -2              # 查看最近两次提交的详细修改
```

## 分支管理

### 创建和使用分支
```bash
# 创建新分支用于新功能
git checkout -b feature/新功能名称

# 查看所有分支
git branch

# 切换分支
git checkout master

# 合并分支
git checkout master
git merge feature/新功能名称

# 删除已合并的分支
git branch -d feature/新功能名称
```

## 常见操作场景

### 场景1：修复 bug
```bash
# 1. 在 master 分支
git checkout master

# 2. 创建修复分支
git checkout -b fix/bug-问题描述

# 3. 修改代码并提交
git add .
git commit -m "修复：解决数据读取时的空指针问题"

# 4. 合并回 master
git checkout master
git merge fix/bug-问题描述

# 5. 删除修复分支
git branch -d fix/bug-问题描述
```

### 场景2：开发新功能
```bash
# 1. 创建功能分支
git checkout -b feature/添加布林带指标

# 2. 开发过程中可以多次提交
git add src/my_etf/indicators/bollinger_bands.py
git commit -m "添加布林带指标计算函数"

git add tests/test_bollinger_bands.py
git commit -m "添加布林带指标单元测试"

# 3. 完成后合并
git checkout master
git merge feature/添加布林带指标
git branch -d feature/添加布林带指标
```

### 场景3：误操作撤销
```bash
# 撤销工作区的修改（未add）
git restore <文件名>

# 撤销暂存区的修改（已add但未commit）
git restore --staged <文件名>

# 撤销最近的commit但保留修改
git reset HEAD~1

# 撤销commit并丢弃所有修改
git reset --hard HEAD~1
```

## 针对本项目的特定建议

### 代码提交类型

| 提交前缀 | 用途 | 示例 |
|---------|------|------|
| `添加` | 新功能 | `添加：新的技术指标计算模块` |
| `修复` | Bug修复 | `修复：解决数据库连接超时问题` |
| `优化` | 性能优化 | `优化：改进XGBoost训练速度` |
| `更新` | 数据/配置更新 | `更新：数据库数据到最新日期` |
| `重构` | 代码重构 | `重构：简化指标计算流程` |
| `文档` | 文档更新 | `文档：更新README说明` |

### 项目文件提交建议

**必须提交：**
- ✅ `src/my_etf/` - 所有源代码
- ✅ `scripts/` - 所有脚本
- ✅ `tests/` - 测试文件
- ✅ `data/etf_data.db` - 数据库文件
- ✅ `*.md` - 文档文件
- ✅ `pyproject.toml`, `requirements.txt` - 依赖配置
- ✅ `.env.example` - 配置示例

**忽略的文件（自动不提交）：**
- ❌ `reports/` - 生成的回测报告
- ❌ `recommendations/` - 生成的推荐报告
- ❌ `index_summary/` - 生成的汇总报告
- ❌ `data/*/*.csv` - 生成的预测数据
- ❌ `*.log` - 日志文件
- ❌ `__pycache__/` - Python 缓存

### 常用提交示例

```bash
# 修改配置文件
git add src/my_etf/config.py
git commit -m "更新：调整回测起始日期为2025-01-01"

# 添加新指标
git add src/my_etf/indicators/advanced_calculator.py
git add tests/test_indicators.py
git commit -m "添加：ATR和波动率指标"

# 修复bug
git add src/my_etf/models/predict.py
git commit -m "修复：解决特征数量不匹配的问题"

# 更新数据库
git add data/etf_data.db
git commit -m "更新：数据库数据更新至2026-03-04"

# 更新文档
git add README.md CLAUDE.md
git commit -m "文档：更新使用说明和开发指南"
```

## 远程仓库操作（可选）

### 添加远程仓库
```bash
# GitHub
git remote add origin https://github.com/your-username/my_etf.git

# Gitee
git remote add origin https://gitee.com/your-username/my_etf.git
```

### 推送到远程
```bash
# 首次推送
git push -u origin master

# 后续推送
git push origin master
```

### 从远程拉取更新
```bash
git fetch origin          # 获取远程更新但不合并
git pull origin master     # 拉取并合并到当前分支
```

## 提交信息规范

### 好的提交信息
- ✅ `添加：新的技术指标计算模块`
- ✅ `修复：解决数据库连接超时问题`
- ✅ `优化：改进XGBoost训练速度50%`
- ✅ `文档：更新CLAUDE.md开发指南`

### 不好的提交信息
- ❌ `更新代码`
- ❌ `修改了一些文件`
- ❌ `fix bug`
- ❌ `update`

## 高级技巧

### 查看文件的修改历史
```bash
git log --follow -p src/my_etf/config.py
```

### 暂存当前工作
```bash
git stash                    # 保存当前工作
git stash pop               # 恢复之前保存的工作
```

### 查看某次提交的详细信息
```bash
git show <commit-hash>      # 例如：git show dcdd6d2
```

### 比较两个版本的文件
```bash
git diff HEAD~1 HEAD src/my_etf/config.py
```

## 常见问题解决

### 问题：push 时需要身份验证
**解决：**
```bash
# 配置用户信息
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"

# 或使用 SSH 密钥（推荐）
ssh-keygen -t ed25519 -C "你的邮箱"
# 然后将公钥添加到 GitHub/Gitee
```

### 问题：合并冲突
**解决：**
```bash
# 1. 手动编辑冲突文件
# 2. 标记为已解决
git add <冲突文件>
# 3. 继续合并
git commit
```

### 问题：想要回退到某个版本
```bash
# 查看历史找到目标提交的 hash
git log --oneline

# 回退（保留修改）
git reset <commit-hash>

# 硬回退（丢弃之后的所有修改）
git reset --hard <commit-hash>
```

## 快速参考

| 操作 | 命令 |
|-----|------|
| 查看状态 | `git status` |
| 查看差异 | `git diff` |
| 添加文件 | `git add .` |
| 提交 | `git commit -m "信息"` |
| 查看历史 | `git log --oneline` |
| 创建分支 | `git branch -b 新分支` |
| 切换分支 | `git checkout 分支` |
| 合并分支 | `git merge 分支` |
| 推送 | `git push origin master` |

## 相关资源

- [Git 官方文档](https://git-scm.com/doc)
- [GitHub Git 指南](https://guides.github.com/)
- [Pro Git 中文版](https://git-scm.com/book/zh/v2)

---

**最后更新：** 2026-03-04
