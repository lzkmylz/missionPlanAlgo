"""
FastAPI 主应用

提供 RESTful API 服务。
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from .routers import schedule, visibility, evaluate, admin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动
    logger.info("MPA API 服务启动中...")

    # 初始化任务后端
    from missionplanalgo.server.backends import create_task_backend
    from missionplanalgo.config import load_config

    config = load_config()
    backend_config = config.get("task_backend", {})
    app.state.task_backend = create_task_backend(backend_config)

    logger.info(f"任务后端初始化完成: {type(app.state.task_backend).__name__}")

    yield

    # 关闭
    logger.info("MPA API 服务关闭中...")
    if hasattr(app.state, 'task_backend'):
        if hasattr(app.state.task_backend, 'shutdown'):
            app.state.task_backend.shutdown()


app = FastAPI(
    title="Mission Planning Algorithm API",
    description="卫星任务规划算法 RESTful API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "Mission Planning Algorithm API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "components": {
            "api": "ok",
            "task_backend": type(app.state.task_backend).__name__ if hasattr(app.state, 'task_backend') else "unknown"
        }
    }


# 注册路由
app.include_router(schedule.router, prefix="/api/v1/schedule", tags=["schedule"])
app.include_router(visibility.router, prefix="/api/v1/visibility", tags=["visibility"])
app.include_router(evaluate.router, prefix="/api/v1/evaluate", tags=["evaluate"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
