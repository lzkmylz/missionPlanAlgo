"""
Pytest 配置文件

定义自定义命令行选项和 fixtures
"""

import pytest
import os


def pytest_addoption(parser):
    """添加自定义命令行选项"""
    parser.addoption(
        "--jvm",
        action="store_true",
        default=False,
        help="启用需要真实JVM环境的测试"
    )


def pytest_configure(config):
    """配置pytest，设置环境变量"""
    jvm_enabled = config.getoption("--jvm")
    # 使用环境变量传递状态给测试文件
    os.environ['_PYTEST_JVM_ENABLED'] = '1' if jvm_enabled else '0'


def _is_jvm_enabled():
    """检查是否启用了JVM测试"""
    return os.environ.get('_PYTEST_JVM_ENABLED') == '1'


# 定义requires_jvm标记 - 这个会被测试文件导入
# 使用os.environ.get直接检查环境变量
requires_jvm = pytest.mark.skipif(
    os.environ.get('_PYTEST_JVM_ENABLED') != '1',
    reason="需要真实JVM环境，使用 --jvm 选项启用"
)


@pytest.fixture(scope="session")
def jvm_bridge():
    """Session级JVM桥接器fixture

    所有JVM测试共享同一个OrekitJavaBridge实例，避免：
    1. 重复JVM启动（节省2-5秒/测试）
    2. 重复数据加载（EGM96/IERS/DE440等大数据文件）
    3. 重复引力场模型初始化

    使用示例:
        def test_something(jvm_bridge):
            frame = jvm_bridge.get_frame("EME2000")
            # 不要重置单例，直接使用传入的bridge
    """
    if not _is_jvm_enabled():
        pytest.skip("需要 --jvm 选项启用")

    try:
        from core.orbit.visibility.orekit_java_bridge import OrekitJavaBridge

        # 确保单例不存在时才创建
        if OrekitJavaBridge._instance is None:
            bridge = OrekitJavaBridge()
        else:
            bridge = OrekitJavaBridge._instance

        # 确保JVM已启动（触发延迟初始化）
        bridge._ensure_jvm_started()

        yield bridge

    except Exception as e:
        pytest.skip(f"JVM初始化失败: {e}")
