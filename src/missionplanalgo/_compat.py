"""
兼容性处理模块

处理项目路径、环境兼容性等问题。
"""

import sys
import os


def ensure_project_in_path():
    """
    确保项目根目录在 Python 路径中

    这个函数在包初始化时调用一次，避免重复添加路径。
    """
    # 获取项目根目录 (src/missionplanalgo/../..)
    project_root = os.path.dirname(os.path.dirname(__file__))

    # 将项目根目录添加到路径开头（确保优先使用本地代码）
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


# 在导入时自动执行
ensure_project_in_path()
