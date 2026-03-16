"""
API 路由模块

包含所有 RESTful API 路由。
"""

from . import schedule, visibility, evaluate, admin

__all__ = ["schedule", "visibility", "evaluate", "admin"]
