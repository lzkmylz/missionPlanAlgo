"""
Evaluate 路由 - 评估 API
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

router = APIRouter()


class EvaluateRequest(BaseModel):
    """评估请求"""
    schedule_result: Dict[str, Any] = Field(..., description="调度结果")
    scenario: Optional[Dict[str, Any]] = Field(None, description="场景数据（可选）")
    metrics: Optional[List[str]] = Field(
        None,
        description="要计算的指标，如: demand_satisfaction, makespan, utilization"
    )


class EvaluateResponse(BaseModel):
    """评估响应"""
    overall_score: float = Field(..., description="总体评分 (0-1)")
    metrics: Dict[str, Any] = Field(..., description="各项指标")
    pareto_front: Optional[List[Dict[str, Any]]] = None
    recommendations: Optional[List[str]] = None


@router.post("/", response_model=EvaluateResponse)
async def evaluate_schedule(request: EvaluateRequest):
    """
    评估调度结果

    计算调度结果的各种性能指标。
    """
    # TODO: 实现实际的评估逻辑
    # 这里暂时返回模拟结果

    result = request.schedule_result

    metrics = {
        "demand_satisfaction_rate": result.get("frequency_satisfaction", 0.95),
        "makespan_hours": result.get("makespan_hours", 8.5),
        "satellite_utilization": result.get("satellite_utilization", 0.72),
        "scheduled_tasks": result.get("scheduled_count", 0),
        "total_tasks": result.get("total_tasks", 3000),
    }

    # 计算总体评分
    overall_score = (
        metrics["demand_satisfaction_rate"] * 0.4 +
        min(metrics["satellite_utilization"] / 0.5, 1.0) * 0.3 +
        min(24.0 / metrics["makespan_hours"], 1.0) * 0.3
    )

    return EvaluateResponse(
        overall_score=overall_score,
        metrics=metrics,
        recommendations=[
            "建议增加卫星数量以提高覆盖率",
            "当前频次满足率良好，可考虑优化资源利用率"
        ]
    )


@router.post("/compare")
async def compare_schedules(results: List[Dict[str, Any]]):
    """
    对比多个调度结果

    比较不同算法或配置的调度结果。
    """
    # TODO: 实现对比逻辑

    comparison = []
    for i, result in enumerate(results):
        comparison.append({
            "index": i,
            "algorithm": result.get("algorithm", "unknown"),
            "scheduled_count": result.get("scheduled_count", 0),
            "frequency_satisfaction": result.get("frequency_satisfaction", 0),
            "satellite_utilization": result.get("satellite_utilization", 0),
        })

    return {
        "comparison": comparison,
        "best": comparison[0] if comparison else None,
        "analysis": "对比分析结果"
    }
