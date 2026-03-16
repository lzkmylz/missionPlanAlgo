"""
Schedule 路由 - 任务调度 API
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
import uuid

router = APIRouter()


class ScheduleRequest(BaseModel):
    """调度请求"""
    scenario: Dict[str, Any] = Field(..., description="场景数据")
    algorithm: Literal["greedy", "edd", "spt", "ga", "sa", "aco", "pso", "tabu"] = Field(
        default="greedy", description="调度算法"
    )
    config: Optional[Dict[str, Any]] = Field(
        default=None, description="算法配置"
    )


class ScheduleResponse(BaseModel):
    """调度响应"""
    task_id: str
    status: str
    message: str


class ScheduleResult(BaseModel):
    """调度结果"""
    task_id: str
    status: str
    algorithm: str
    scheduled_count: int
    metrics: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.post("/", response_model=ScheduleResponse)
async def create_schedule(request: ScheduleRequest, background_tasks: BackgroundTasks, request_obj: Request):
    """
    提交调度任务

    异步执行调度算法，返回任务ID用于查询结果。
    """
    task_id = str(uuid.uuid4())

    # 获取任务后端
    backend = request_obj.app.state.task_backend

    # TODO: 实现实际的调度逻辑
    # 这里暂时模拟
    def mock_schedule():
        import time
        time.sleep(2)
        return {
            "algorithm": request.algorithm,
            "scheduled_count": 2638,
            "frequency_satisfaction": 1.0,
            "satellite_utilization": 0.132,
            "makespan_hours": 8.81,
        }

    # 提交任务
    backend.submit(mock_schedule)

    return ScheduleResponse(
        task_id=task_id,
        status="queued",
        message="调度任务已提交"
    )


@router.get("/{task_id}", response_model=ScheduleResult)
async def get_schedule_status(task_id: str, request_obj: Request):
    """
    获取调度任务状态和结果
    """
    backend = request_obj.app.state.task_backend

    task_info = backend.get_status(task_id)

    if task_info is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    # 构建响应
    result = ScheduleResult(
        task_id=task_id,
        status=task_info.status.value,
        algorithm="greedy",  # TODO: 从任务信息中获取
        scheduled_count=0,
    )

    if task_info.status.value == "completed":
        result.metrics = task_info.result if isinstance(task_info.result, dict) else {}
    elif task_info.status.value == "failed":
        result.error = task_info.error

    return result


@router.get("/algorithms/")
async def list_algorithms():
    """
    获取支持的算法列表
    """
    return {
        "algorithms": [
            {"id": "greedy", "name": "Greedy Scheduler", "category": "greedy"},
            {"id": "edd", "name": "EDD Scheduler", "category": "greedy"},
            {"id": "spt", "name": "SPT Scheduler", "category": "greedy"},
            {"id": "ga", "name": "Genetic Algorithm", "category": "metaheuristic"},
            {"id": "sa", "name": "Simulated Annealing", "category": "metaheuristic"},
            {"id": "aco", "name": "Ant Colony Optimization", "category": "metaheuristic"},
            {"id": "pso", "name": "Particle Swarm Optimization", "category": "metaheuristic"},
            {"id": "tabu", "name": "Tabu Search", "category": "metaheuristic"},
        ]
    }
