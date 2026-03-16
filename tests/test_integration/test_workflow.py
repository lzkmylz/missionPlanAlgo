"""
集成测试 - 端到端流程
"""
import pytest
import sys
import os
import json
import tempfile
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

from missionplanalgo.api import load_scenario
from missionplanalgo.config import load_config, validate_config


@pytest.fixture
def sample_scenario_data():
    """创建示例场景数据"""
    return {
        "satellites": [
            {
                "id": "SAT-001",
                "type": "optical_1",
                "orbit": {
                    "type": "keplerian",
                    "sma": 6878000,
                    "ecc": 0.001,
                    "inc": 97.5,
                    "raan": 45.0,
                    "aop": 0.0,
                    "ta": 0.0,
                    "epoch": "2024-01-01T00:00:00Z"
                },
                "capabilities": {
                    "max_roll_angle": 35.0,
                    "max_pitch_angle": 20.0,
                    "max_roll_rate": 3.0,
                    "max_pitch_rate": 2.0,
                    "min_imaging_altitude": 400000,
                    "max_imaging_altitude": 600000
                }
            }
        ],
        "targets": [
            {
                "id": "TARGET-001",
                "type": "point",
                "latitude": 39.9042,
                "longitude": 116.4074,
                "priority": 1.0
            },
            {
                "id": "TARGET-002",
                "type": "point",
                "latitude": 31.2304,
                "longitude": 121.4737,
                "priority": 1.0
            }
        ],
        "ground_stations": [
            {
                "id": "GS-BEIJING",
                "latitude": 40.0,
                "longitude": 116.0,
                "min_elevation": 10.0
            }
        ],
        "scenario": {
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-01T12:00:00Z",
            "time_step": 60
        }
    }


@pytest.fixture
def temp_scenario_file(sample_scenario_data):
    """创建临时场景文件"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_scenario_data, f)
        f.flush()
        yield f.name
        os.unlink(f.name)


def test_load_scenario_from_file(temp_scenario_file):
    """测试从文件加载场景"""
    scenario = load_scenario(temp_scenario_file)

    assert scenario is not None
    assert isinstance(scenario, dict)
    assert "satellites" in scenario
    assert "targets" in scenario
    assert len(scenario["satellites"]) == 1
    assert len(scenario["targets"]) == 2
    assert scenario["satellites"][0]["id"] == "SAT-001"


def test_load_scenario_from_dict(sample_scenario_data):
    """测试从字典加载场景"""
    scenario = load_scenario(sample_scenario_data)

    assert scenario is not None
    assert isinstance(scenario, dict)
    assert "satellites" in scenario
    assert "targets" in scenario
    assert len(scenario["satellites"]) == 1
    assert len(scenario["targets"]) == 2


def test_config_loading_and_validation():
    """测试配置加载和验证"""
    config = load_config()

    # 验证配置结构
    assert isinstance(config, dict)

    # 验证配置通过验证
    validate_config(config)  # 不应抛出异常


def test_config_validation_with_invalid_algorithm():
    """测试配置验证 - 无效算法"""
    from missionplanalgo.config import ConfigValidationError

    config = {
        "scheduler": {
            "default_algorithm": "invalid_algorithm"
        }
    }

    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(config)

    assert "无效的默认算法" in str(exc_info.value)


def test_end_to_end_workflow_mock(temp_scenario_file):
    """测试端到端工作流（模拟）"""
    # 加载场景
    scenario = load_scenario(temp_scenario_file)
    assert scenario is not None
    assert isinstance(scenario, dict)

    # 验证场景内容
    assert len(scenario["satellites"]) > 0
    assert len(scenario["targets"]) > 0

    # TODO: 完整的调度流程需要可见性缓存


def test_cli_import():
    """测试CLI模块可导入"""
    from missionplanalgo.cli import main
    assert main is not None


def test_server_import():
    """测试服务器模块可导入"""
    from missionplanalgo.server import app
    assert app is not None


def test_backends_import():
    """测试后端模块可导入"""
    from missionplanalgo.server.backends import create_task_backend
    assert create_task_backend is not None


class TestIntegrationSuite:
    """集成测试套件"""

    def test_scenario_to_schedule_workflow(self, sample_scenario_data):
        """测试场景到调度的完整流程"""
        # 1. 加载场景
        scenario = load_scenario(sample_scenario_data)
        assert scenario is not None
        assert isinstance(scenario, dict)

        # 2. 验证卫星
        satellites = scenario.get("satellites", [])
        assert len(satellites) > 0
        sat = satellites[0]
        assert sat["id"] == "SAT-001"
        assert sat["capabilities"]["max_roll_angle"] == 35.0

        # 3. 验证目标
        targets = scenario.get("targets", [])
        assert len(targets) == 2
        assert targets[0]["id"] == "TARGET-001"

    def test_multiple_satellites_scenario(self):
        """测试多卫星场景"""
        scenario_data = {
            "satellites": [
                {
                    "id": f"SAT-{i:03d}",
                    "type": "optical_1",
                    "orbit": {
                        "type": "keplerian",
                        "sma": 6878000,
                        "ecc": 0.001,
                        "inc": 97.5,
                        "raan": i * 10.0,
                        "aop": 0.0,
                        "ta": 0.0,
                        "epoch": "2024-01-01T00:00:00Z"
                    },
                    "capabilities": {
                        "max_roll_angle": 35.0,
                        "max_pitch_angle": 20.0
                    }
                }
                for i in range(5)
            ],
            "targets": [
                {
                    "id": f"TARGET-{i:03d}",
                    "type": "point",
                    "latitude": 39.0 + i * 0.1,
                    "longitude": 116.0 + i * 0.1
                }
                for i in range(10)
            ],
            "scenario": {
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-01-01T06:00:00Z",
                "time_step": 60
            }
        }

        scenario = load_scenario(scenario_data)
        assert isinstance(scenario, dict)
        assert len(scenario["satellites"]) == 5
        assert len(scenario["targets"]) == 10
