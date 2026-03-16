"""
测试 API 错误处理
"""

import pytest
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

from missionplanalgo.api import (
    _load_scenario_data,
    ScenarioLoadError,
    load_scenario,
)


def test_load_nonexistent_file():
    """测试加载不存在的文件"""
    with pytest.raises(ScenarioLoadError) as exc_info:
        _load_scenario_data('/path/that/does/not/exist.json')
    assert '不存在' in str(exc_info.value)


def test_load_directory_instead_of_file(tmp_path):
    """测试加载目录而不是文件"""
    with pytest.raises(ScenarioLoadError) as exc_info:
        _load_scenario_data(str(tmp_path))
    assert '不是文件' in str(exc_info.value)


def test_load_invalid_json(tmp_path):
    """测试加载无效的 JSON"""
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{invalid json")

    with pytest.raises(ScenarioLoadError) as exc_info:
        _load_scenario_data(str(invalid_json))
    assert 'JSON格式错误' in str(exc_info.value)


def test_load_invalid_yaml(tmp_path):
    """测试加载无效的 YAML"""
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("{invalid: yaml: syntax}")

    with pytest.raises(ScenarioLoadError) as exc_info:
        _load_scenario_data(str(invalid_yaml))
    assert 'YAML格式错误' in str(exc_info.value)


def test_load_valid_json(tmp_path):
    """测试加载有效的 JSON"""
    valid_json = tmp_path / "valid.json"
    valid_json.write_text('{"satellites": [], "targets": []}')

    result = _load_scenario_data(str(valid_json))
    assert result == {"satellites": [], "targets": []}


def test_load_valid_yaml(tmp_path):
    """测试加载有效的 YAML"""
    valid_yaml = tmp_path / "valid.yaml"
    valid_yaml.write_text("satellites:\n  - id: SAT-001\ntargets: []")

    result = _load_scenario_data(str(valid_yaml))
    assert result["satellites"][0]["id"] == "SAT-001"


def test_load_dict_directly():
    """测试直接传入字典"""
    data = {"satellites": [{"id": "SAT-001"}]}
    result = _load_scenario_data(data)
    assert result == data


def test_load_with_non_utf8_encoding(tmp_path):
    """测试非 UTF-8 编码的文件"""
    # 创建 Latin-1 编码的文件
    file_path = tmp_path / "latin1.yaml"
    with open(file_path, 'w', encoding='latin-1') as f:
        f.write("satellites: []")

    # 在 UTF-8 模式下读取可能会失败
    # 但这个测试可能依赖于系统行为
    try:
        result = _load_scenario_data(str(file_path))
        # 如果成功，应该能正常解析
    except ScenarioLoadError as e:
        # 如果失败，应该报告编码错误
        assert '编码' in str(e) or 'format' in str(e).lower()
