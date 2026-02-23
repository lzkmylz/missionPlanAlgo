"""
YAML场景配置解析器的单元测试
"""

import pytest
import os
import tempfile
from datetime import datetime
from utils.yaml_loader import YamlLoader


class TestYamlLoader:
    """YamlLoader 测试类"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_yaml_content(self):
        """示例YAML配置内容"""
        return """
scenario:
  name: "测试场景"
  duration:
    start: "2024-01-01T00:00:00Z"
    end: "2024-01-02T00:00:00Z"

  satellites:
    - id: "OPT-01"
      type: "optical_1"
      orbit:
        type: "SSO"
        altitude: 500000
        inclination: 97.4
      capabilities:
        imaging_modes: ["push_broom"]
        max_off_nadir: 30.0
        storage_capacity: 500
        power_capacity: 2000

    - id: "SAR-01"
      type: "sar_1"
      capabilities:
        imaging_modes: ["spotlight", "stripmap"]
        storage_capacity: 1000
        power_capacity: 3000

  targets:
    point_group:
      count: 300
      distribution: "clustered"
      regions:
        - name: "华东区域"
          bounds: [118.0, 30.0, 123.0, 35.0]
          density: "high"

    large_area:
      count: 5
      areas:
        - id: "AREA-01"
          vertices: [[116.0, 39.0], [117.0, 39.0], [117.0, 40.0], [116.0, 40.0]]
          resolution_required: 10.0
          priority: 1

  ground_stations:
    - id: "GS-BEIJING"
      location: [116.4, 39.9, 0.0]
      antennas:
        - id: "ANT-01"
          elevation_min: 5.0
          frequency_band: "X"
"""

    @pytest.fixture
    def minimal_yaml_content(self):
        """最小YAML配置内容"""
        return """
scenario:
  name: "最小场景"
  duration:
    start: "2024-01-01T00:00:00Z"
    end: "2024-01-01T12:00:00Z"
  satellites: []
  targets: {}
  ground_stations: []
"""

    def test_load_valid_yaml(self, temp_dir, sample_yaml_content):
        """测试加载有效的YAML文件"""
        yaml_path = os.path.join(temp_dir, "test_scenario.yaml")
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(sample_yaml_content)

        loader = YamlLoader()
        config = loader.load(yaml_path)

        assert config is not None
        assert 'scenario' in config
        assert config['scenario']['name'] == "测试场景"

    def test_load_file_not_found(self):
        """测试加载不存在的文件"""
        loader = YamlLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/scenario.yaml")

    def test_load_invalid_yaml(self, temp_dir):
        """测试加载无效的YAML文件"""
        yaml_path = os.path.join(temp_dir, "invalid.yaml")
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write("invalid: yaml: content: [")

        loader = YamlLoader()
        with pytest.raises(Exception):
            loader.load(yaml_path)

    def test_parse_scenario_basic_info(self, temp_dir, sample_yaml_content):
        """测试解析场景基本信息"""
        yaml_path = os.path.join(temp_dir, "test_scenario.yaml")
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(sample_yaml_content)

        loader = YamlLoader()
        config = loader.load(yaml_path)
        scenario_info = loader.parse_scenario_basic_info(config)

        assert scenario_info['name'] == "测试场景"
        assert isinstance(scenario_info['start_time'], datetime)
        assert isinstance(scenario_info['end_time'], datetime)
        assert scenario_info['start_time'].year == 2024
        assert scenario_info['end_time'].day == 2

    def test_parse_satellites(self, temp_dir, sample_yaml_content):
        """测试解析卫星配置"""
        yaml_path = os.path.join(temp_dir, "test_scenario.yaml")
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(sample_yaml_content)

        loader = YamlLoader()
        config = loader.load(yaml_path)
        satellites = loader.parse_satellites(config)

        assert len(satellites) == 2
        assert satellites[0]['id'] == "OPT-01"
        assert satellites[0]['type'] == "optical_1"
        assert satellites[0]['capabilities']['storage_capacity'] == 500
        assert satellites[1]['id'] == "SAR-01"
        assert "spotlight" in satellites[1]['capabilities']['imaging_modes']

    def test_parse_targets(self, temp_dir, sample_yaml_content):
        """测试解析目标配置"""
        yaml_path = os.path.join(temp_dir, "test_scenario.yaml")
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(sample_yaml_content)

        loader = YamlLoader()
        config = loader.load(yaml_path)
        targets = loader.parse_targets(config)

        assert 'point_group' in targets
        assert targets['point_group']['count'] == 300
        assert targets['point_group']['distribution'] == "clustered"
        assert 'large_area' in targets
        assert len(targets['large_area']['areas']) == 1
        assert targets['large_area']['areas'][0]['id'] == "AREA-01"

    def test_parse_ground_stations(self, temp_dir, sample_yaml_content):
        """测试解析地面站配置"""
        yaml_path = os.path.join(temp_dir, "test_scenario.yaml")
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(sample_yaml_content)

        loader = YamlLoader()
        config = loader.load(yaml_path)
        ground_stations = loader.parse_ground_stations(config)

        assert len(ground_stations) == 1
        assert ground_stations[0]['id'] == "GS-BEIJING"
        assert ground_stations[0]['location'] == [116.4, 39.9, 0.0]
        assert len(ground_stations[0]['antennas']) == 1
        assert ground_stations[0]['antennas'][0]['id'] == "ANT-01"

    def test_load_minimal_scenario(self, temp_dir, minimal_yaml_content):
        """测试加载最小场景配置"""
        yaml_path = os.path.join(temp_dir, "minimal.yaml")
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(minimal_yaml_content)

        loader = YamlLoader()
        config = loader.load(yaml_path)

        assert config['scenario']['name'] == "最小场景"
        satellites = loader.parse_satellites(config)
        assert len(satellites) == 0
        targets = loader.parse_targets(config)
        assert targets == {}
        ground_stations = loader.parse_ground_stations(config)
        assert len(ground_stations) == 0

    def test_parse_empty_scenario(self):
        """测试解析空场景配置"""
        loader = YamlLoader()
        empty_config = {}

        scenario_info = loader.parse_scenario_basic_info(empty_config)
        assert scenario_info == {}

        satellites = loader.parse_satellites(empty_config)
        assert satellites == []

        targets = loader.parse_targets(empty_config)
        assert targets == {}

        ground_stations = loader.parse_ground_stations(empty_config)
        assert ground_stations == []

    def test_parse_scenario_with_missing_optional_fields(self, temp_dir):
        """测试解析缺少可选字段的场景"""
        yaml_content = """
scenario:
  name: "不完整场景"
  duration:
    start: "2024-01-01T00:00:00Z"
    end: "2024-01-02T00:00:00Z"
"""
        yaml_path = os.path.join(temp_dir, "incomplete.yaml")
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)

        loader = YamlLoader()
        config = loader.load(yaml_path)

        # 应该能正常解析，返回空列表/字典
        satellites = loader.parse_satellites(config)
        assert satellites == []
        targets = loader.parse_targets(config)
        assert targets == {}
        ground_stations = loader.parse_ground_stations(config)
        assert ground_stations == []

    def test_load_to_mission(self, temp_dir, sample_yaml_content):
        """测试加载YAML并转换为Mission对象"""
        yaml_path = os.path.join(temp_dir, "test_scenario.yaml")
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(sample_yaml_content)

        loader = YamlLoader()
        mission = loader.load_to_mission(yaml_path)

        assert mission is not None
        assert mission.name == "测试场景"
        assert len(mission.satellites) == 2
        assert len(mission.targets) > 0
        assert len(mission.ground_stations) == 1

    def test_load_to_mission_file_not_found(self):
        """测试加载不存在的YAML文件到Mission"""
        loader = YamlLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_to_mission("/nonexistent/path/scenario.yaml")

    def test_validate_schema_valid(self, temp_dir, sample_yaml_content):
        """测试验证有效的YAML配置"""
        yaml_path = os.path.join(temp_dir, "test_scenario.yaml")
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(sample_yaml_content)

        loader = YamlLoader()
        config = loader.load(yaml_path)
        is_valid, errors = loader.validate_schema(config)

        assert is_valid is True
        assert errors == []

    def test_validate_schema_invalid(self):
        """测试验证无效的YAML配置"""
        loader = YamlLoader()
        invalid_config = {'invalid': 'config'}
        is_valid, errors = loader.validate_schema(invalid_config)

        assert is_valid is False
        assert len(errors) > 0

    def test_validate_schema_missing_required_fields(self):
        """测试验证缺少必需字段的YAML配置"""
        loader = YamlLoader()
        incomplete_config = {
            'scenario': {
                'name': 'test'
                # 缺少 duration
            }
        }
        is_valid, errors = loader.validate_schema(incomplete_config)

        assert is_valid is False
        assert any('duration' in error.lower() for error in errors)
