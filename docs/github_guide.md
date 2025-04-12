# GitHub上传指南

本文档提供了将Bybit高频交易系统上传到GitHub并进行后续更新的详细步骤。

## 目录

1. [准备工作](#准备工作)
2. [创建GitHub仓库](#创建github仓库)
3. [初始化本地仓库](#初始化本地仓库)
4. [添加.gitignore文件](#添加gitignore文件)
5. [提交代码](#提交代码)
6. [推送到GitHub](#推送到github)
7. [后续更新流程](#后续更新流程)
8. [分支管理](#分支管理)
9. [协作开发](#协作开发)
10. [版本发布](#版本发布)

## 准备工作

在开始之前，请确保您已经：

1. 创建了GitHub账号
2. 安装了Git客户端
3. 配置了Git的用户名和邮箱

```bash
# 配置Git用户名和邮箱
git config --global user.name "您的GitHub用户名"
git config --global user.email "您的邮箱地址"

# 可选：配置默认编辑器
git config --global core.editor "vim"  # 或其他您喜欢的编辑器
```

## 创建GitHub仓库

1. 登录GitHub账号
2. 点击右上角的"+"图标，选择"New repository"
3. 填写仓库信息：
   - Repository name: `bybit-trading-system`（或您喜欢的名称）
   - Description: `Bybit高频交易系统，支持实时数据订阅、历史数据回测和策略优化`
   - 选择"Public"或"Private"（根据您的需求）
   - 勾选"Add a README file"
   - 勾选"Add .gitignore"，选择"Python"模板
   - 勾选"Choose a license"，选择适合的开源许可证（如MIT License）
4. 点击"Create repository"按钮

## 初始化本地仓库

如果您是从零开始：

```bash
# 克隆新创建的空仓库
git clone https://github.com/您的用户名/bybit-trading-system.git
cd bybit-trading-system

# 复制项目文件到仓库目录
cp -r /path/to/bybit_trading_system/* .
```

如果您已经有了本地项目：

```bash
# 进入项目目录
cd /path/to/bybit_trading_system

# 初始化Git仓库
git init

# 添加远程仓库
git remote add origin https://github.com/您的用户名/bybit-trading-system.git

# 拉取远程仓库内容（如README和.gitignore）
git pull origin main --allow-unrelated-histories
```

## 添加.gitignore文件

如果您没有在GitHub上选择Python模板，请手动创建.gitignore文件：

```bash
# 创建.gitignore文件
cat > .gitignore << 'EOF'
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/

# IDE settings
.idea/
.vscode/
*.swp
*.swo

# Project specific
config.ini
*.db
data/raw/
data/processed/
logs/
EOF
```

## 提交代码

```bash
# 添加所有文件到暂存区
git add .

# 检查状态
git status

# 提交代码
git commit -m "初始提交：Bybit高频交易系统"
```

## 推送到GitHub

```bash
# 推送到GitHub
git push -u origin main  # 或 master，取决于您的默认分支名称
```

如果遇到分支名称问题：

```bash
# 查看当前分支
git branch

# 如果您的本地分支是master，而GitHub默认是main
git branch -M main  # 将master重命名为main
git push -u origin main
```

## 后续更新流程

每次更新代码后，按照以下步骤提交和推送：

```bash
# 查看修改的文件
git status

# 添加修改的文件
git add 修改的文件  # 或使用 git add . 添加所有修改

# 提交修改
git commit -m "更新说明：简要描述您的修改"

# 推送到GitHub
git push origin main
```

## 分支管理

对于新功能开发或bug修复，建议使用分支：

```bash
# 创建并切换到新分支
git checkout -b feature/新功能名称  # 或 bugfix/问题描述

# 在分支上进行开发...

# 提交修改
git add .
git commit -m "新功能：功能描述"

# 推送分支到GitHub
git push origin feature/新功能名称

# 开发完成后，切回main分支
git checkout main

# 合并功能分支
git merge feature/新功能名称

# 推送更新后的main分支
git push origin main

# 可选：删除本地功能分支
git branch -d feature/新功能名称

# 可选：删除远程功能分支
git push origin --delete feature/新功能名称
```

## 协作开发

如果有多人协作开发：

1. 其他开发者可以fork您的仓库
2. 克隆fork后的仓库到本地
3. 创建功能分支进行开发
4. 提交并推送到自己的fork仓库
5. 在GitHub上创建Pull Request
6. 您审核代码后合并Pull Request

## 版本发布

当系统达到一个稳定版本时，可以创建版本标签：

```bash
# 创建标签
git tag -a v1.0.0 -m "版本1.0.0：初始稳定版本"

# 推送标签到GitHub
git push origin v1.0.0
```

在GitHub上，您可以基于标签创建Release，添加详细的版本说明和二进制文件。

## 自动化CI/CD（可选）

您可以使用GitHub Actions设置自动化测试和部署流程：

1. 在项目根目录创建`.github/workflows`目录
2. 添加workflow文件，如`test.yml`：

```yaml
name: Python Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
    - name: Run tests
      run: |
        pytest
```

## 结论

按照上述步骤，您可以将Bybit高频交易系统上传到GitHub，并进行后续的版本管理和更新。GitHub不仅提供了代码托管，还提供了问题跟踪、Wiki文档、项目管理等功能，可以帮助您更好地管理和维护项目。
