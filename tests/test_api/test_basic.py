"""
测试顶层 API
"""

import pytest
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

import missionplanalgo as mpa


def test_version():
    """测试版本号"""
    assert mpa.__version__ == "1.0.0"
    assert mpa.__author__ == "赵林"


def test_exports():
    """测试导出的符号"""
    assert 'schedule' in mpa.__all__
    assert 'compute_visibility' in mpa.__all__
    assert 'load_scenario' in mpa.__all__
    assert 'evaluate_schedule' in mpa.__all__


def test_schedule_mock():
    """测试 schedule 函数（模拟数据）"""
    scenario = {
        'satellites': [{'id': 'SAT-001', 'type': 'optical'}],
        'targets': [{'id': 'TGT-001', 'lat': 39.9, 'lon': 116.4}],
    }

    result = mpa.schedule(scenario, algorithm='greedy')

    assert 'scheduled_tasks' in result
    assert 'metrics' in result
    assert result['algorithm'] == 'greedy'


def test_compute_visibility_mock():
    """测试 compute_visibility 函数（模拟数据）"""
    scenario = {
        'satellites': [{'id': 'SAT-001'}],
        'targets': [{'id': 'TGT-001'}],
    }

    result = mpa.compute_visibility(scenario, use_java=False)

    assert 'total_windows' in result
    assert 'satellite_count' in result
    assert result['satellite_count'] == 1


def test_load_scenario_json(tmp_path):
    """测试 load_scenario 函数"""
    import json

    scenario_file = tmp_path / "test_scenario.json"
    scenario_data = {
        'satellites': [{'id': 'SAT-001'}],
        'targets': [{'id': 'TGT-001'}],
    }

    with open(scenario_file, 'w') as f:
        json.dump(scenario_data, f)

    loaded = mpa.load_scenario(scenario_file)
    assert loaded['satellites'][0]['id'] == 'SAT-001'


def test_evaluate_schedule():
    """测试 evaluate_schedule 函数"""
    result = {
        'scheduled_tasks': [{'id': 'task_1'}],
        'metrics': {
            'scheduled_count': 1,
            'frequency_satisfaction': 1.0,
            'satellite_utilization': 0.8,
        }
    }

    metrics = mpa.evaluate_schedule(result)

    assert 'overall_score' in metrics
    assert 'resource_utilization' in metrics
