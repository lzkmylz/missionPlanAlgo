# missionplanalgo - Python 包重构

## 概述

本项目已重构为现代 Python 包，支持 `pip install` 安装，提供 CLI 工具和 API 服务。

## 安装

```bash
# 开发模式安装
pip install -e .

# 或生产环境安装
pip install .
```

## 功能特性

### 1. CLI 工具 (`mpa`)

```bash
# 执行调度
mpa schedule run -s scenario.json -c cache.json -a greedy

# 计算可见性
mpa visibility compute -s scenario.json -o windows.json

# 启动 API 服务
mpa serve start --host 0.0.0.0 --port 8000

# 配置管理
mpa config init
mpa config set scheduler.default_algorithm ga
```

### 2. Python API

```python
import missionplanalgo as mpa

# 执行调度
result = mpa.schedule("scenarios/test.json", algorithm="greedy")

# 计算可见性
windows = mpa.compute_visibility("scenarios/test.json")

# 评估结果
metrics = mpa.evaluate_schedule(result)
```

### 3. RESTful API

```bash
# 启动服务
mpa serve start

# 访问 API 文档
open http://127.0.0.1:8000/docs
```

API 端点：
- `POST /api/v1/schedule` - 提交调度任务
- `GET /api/v1/schedule/{task_id}` - 查询任务状态
- `POST /api/v1/visibility/compute` - 计算可见性
- `POST /api/v1/evaluate` - 评估调度结果
- `GET /health` - 健康检查

## 架构

```
src/missionplanalgo/
├── __init__.py          # 包入口，暴露核心 API
├── api.py               # 便捷函数 (schedule, visibility)
├── cli/                 # 命令行工具
│   ├── main.py          # Click 入口
│   └── commands/        # 子命令实现
├── server/              # API 服务
│   ├── app.py           # FastAPI 应用
│   └── routers/         # REST API 路由
├── config/              # 配置管理
│   ├── default.yaml     # 默认配置
│   └── __init__.py      # 配置加载
└── [软链接到现有代码]
    ├── core/            # 核心算法
    ├── scheduler/       # 调度器
    └── ...
```

## 运行模式

### 单机模式（默认）

- 使用 `ProcessPoolExecutor` 执行任务
- 使用 SQLite 存储任务状态
- 无需 Redis，开箱即用

### 分布式模式（可选）

```bash
# 1. 安装 Celery
pip install missionplanalgo[celery]

# 2. 配置 Redis
mpa config set task_backend.type celery
mpa config set task_backend.broker_url redis://localhost:6379/0

# 3. 启动服务
mpa serve start
```

## 配置文件

```yaml
# ~/.config/mpa/config.yaml
scheduler:
  default_algorithm: "greedy"
  enable_frequency: true

task_backend:
  type: "local"
  max_workers: 4

server:
  host: "127.0.0.1"
  port: 8000
```

## 测试

```bash
# 测试 CLI
PYTHONPATH=src python3 -m missionplanalgo.cli.main --help

# 测试 API
PYTHONPATH=src python3 -m missionplanalgo.server.app

# 运行 pytest
pytest tests/
```
