## 介绍

uv 是一个极快的 Python 包管理器，使用 Rust 编写，旨在替代 pip、pip-tools、pipx、poetry、pyenv、virtualenv 等工具。它提供了极速的依赖解析和安装，支持内联脚本元数据，并且能够管理 Python 版本，是现代 Python 项目管理的理想选择。

## 常用指令

### 项目初始化
```bash
uv init          # 初始化新项目
uv init --app    # 初始化应用程序项目
uv init --lib    # 初始化库项目
```

### 依赖管理
```bash
uv add <package>         # 添加依赖
uv add --dev <package>   # 添加开发依赖
uv remove <package>      # 移除依赖
uv sync                  # 同步依赖到虚拟环境
uv lock                  # 锁定依赖版本
```

### 运行命令
```bash
uv run <command>         # 在项目环境中运行命令
uv run python script.py  # 运行 Python 脚本
uv run pytest           # 运行测试
```

### Python 版本管理
```bash
uv python list           # 列出可用的 Python 版本
uv python install 3.12   # 安装特定 Python 版本
uv python pin 3.12       # 为项目固定 Python 版本
```

### 虚拟环境
```bash
uv venv                  # 创建虚拟环境
uv pip install <package> # 在虚拟环境中安装包
uv pip list             # 列出已安装的包
```

### 工具管理
```bash
uv tool install <package>    # 全局安装工具
uv tool run <command>        # 运行工具命令
uv tool list                 # 列出已安装的工具
```