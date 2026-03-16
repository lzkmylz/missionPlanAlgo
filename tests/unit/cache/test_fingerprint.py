"""
场景指纹测试

测试场景指纹计算和比较功能。
"""

import json
import pytest
from datetime import datetime
from pathlib import Path

from core.cache.fingerprint import ScenarioFingerprint, ComponentHash
from core.cache.fingerprint_calculator import FingerprintCalculator, FingerprintComparator


class TestComponentHash:
    """测试组件哈希"""

    def test_component_hash_creation(self):
        """测试组件哈希创建"""
        hash_obj = ComponentHash(
            hash_value="abcd1234efgh5678",
            component_type="satellites",
            item_count=5,
            item_ids={"SAT-01", "SAT-02", "SAT-03"}
        )

        assert hash_obj.hash_value == "abcd1234efgh5678"
        assert hash_obj.component_type == "satellites"
        assert hash_obj.item_count == 5
        assert len(hash_obj.item_ids) == 3

    def test_component_hash_to_dict(self):
        """测试组件哈希序列化"""
        hash_obj = ComponentHash(
            hash_value="abcd1234",
            component_type="targets",
            item_count=100,
            item_ids={"TGT-01", "TGT-02"}
        )

        data = hash_obj.to_dict()
        assert data['hash'] == "abcd1234"
        assert data['type'] == "targets"
        assert data['count'] == 100
        assert 'TGT-01' in data['ids']

    def test_component_hash_from_dict(self):
        """测试组件哈希反序列化"""
        data = {
            'hash': 'test1234',
            'type': 'ground_stations',
            'count': 3,
            'ids': ['GS-01', 'GS-02', 'GS-03']
        }

        hash_obj = ComponentHash.from_dict(data)
        assert hash_obj.hash_value == 'test1234'
        assert hash_obj.component_type == 'ground_stations'
        assert hash_obj.item_count == 3
        assert hash_obj.item_ids == {'GS-01', 'GS-02', 'GS-03'}


class TestScenarioFingerprint:
    """测试场景指纹"""

    @pytest.fixture
    def sample_fingerprint(self):
        """创建示例指纹"""
        return ScenarioFingerprint(
            full_hash="a1b2c3d4e5f6789012345678901234567890abcd",
            satellites=ComponentHash(
                hash_value="sat1234567890abcd",
                component_type="satellites",
                item_count=2,
                item_ids={"SAT-01", "SAT-02"}
            ),
            ground_stations=ComponentHash(
                hash_value="gs1234567890abcd",
                component_type="ground_stations",
                item_count=1,
                item_ids={"GS-01"}
            ),
            targets=ComponentHash(
                hash_value="tgt1234567890abcd",
                component_type="targets",
                item_count=10,
                item_ids={f"TGT-{i:02d}" for i in range(1, 11)}
            ),
            time_range=ComponentHash(
                hash_value="tr1234567890abcd",
                component_type="time_range",
                item_count=1,
                item_ids=set()
            ),
            scenario_name="Test Scenario",
            created_at=datetime.now()
        )

    def test_get_cache_key(self, sample_fingerprint):
        """测试缓存键生成"""
        key = sample_fingerprint.get_cache_key()
        assert key == "a1b2c3d4e5f67890"
        assert len(key) == 16

    def test_get_partial_key(self, sample_fingerprint):
        """测试部分缓存键生成"""
        # 单组件
        key = sample_fingerprint.get_partial_key(['satellites'])
        assert "sat12345" in key

        # 多组件
        key = sample_fingerprint.get_partial_key(['satellites', 'targets'])
        assert "sat12345" in key
        assert "tgt12345" in key

    def test_fingerprint_to_dict(self, sample_fingerprint):
        """测试指纹序列化"""
        data = sample_fingerprint.to_dict()

        assert data['full_hash'] == sample_fingerprint.full_hash
        assert data['scenario_name'] == "Test Scenario"
        assert 'satellites' in data
        assert 'ground_stations' in data
        assert 'targets' in data
        assert 'time_range' in data

    def test_fingerprint_from_dict(self, sample_fingerprint):
        """测试指纹反序列化"""
        data = sample_fingerprint.to_dict()
        restored = ScenarioFingerprint.from_dict(data)

        assert restored.full_hash == sample_fingerprint.full_hash
        assert restored.scenario_name == sample_fingerprint.scenario_name
        assert restored.satellites.hash_value == sample_fingerprint.satellites.hash_value
        assert restored.targets.item_count == sample_fingerprint.targets.item_count


class TestFingerprintCalculator:
    """测试指纹计算器"""

    @pytest.fixture
    def sample_scenario_data(self):
        """创建示例场景数据"""
        return {
            "name": "Test Scenario",
            "satellites": [
                {
                    "id": "SAT-01",
                    "orbit": {
                        "semi_major_axis": 6871000.0,
                        "eccentricity": 0.001,
                        "inclination": 55.0,
                        "raan": 0.0,
                        "arg_of_perigee": 0.0,
                        "mean_anomaly": 0.0,
                        "epoch": "2024-03-15T00:00:00Z"
                    },
                    "capabilities": {
                        "max_roll_angle": 35.0,
                        "max_pitch_angle": 20.0,
                        "resolution": 1.0,
                        "swath_width": 15000.0
                    }
                }
            ],
            "ground_stations": [
                {
                    "id": "GS-01",
                    "latitude": 39.9,
                    "longitude": 116.4,
                    "altitude": 0.0,
                    "min_elevation": 5.0
                }
            ],
            "targets": [
                {"id": "TGT-01", "latitude": 30.0, "longitude": 120.0, "type": "point"}
            ],
            "duration": {
                "start": "2024-03-15T00:00:00Z",
                "end": "2024-03-16T00:00:00Z"
            }
        }

    def test_calculate_from_data(self, sample_scenario_data):
        """测试从数据计算指纹"""
        calculator = FingerprintCalculator()
        fingerprint = calculator.calculate_from_data(sample_scenario_data)

        assert fingerprint.scenario_name == "Test Scenario"
        assert fingerprint.satellites.item_count == 1
        assert fingerprint.ground_stations.item_count == 1
        assert fingerprint.targets.item_count == 1
        assert len(fingerprint.full_hash) == 64  # SHA256长度

    def test_identical_scenarios_same_hash(self, tmp_path, sample_scenario_data):
        """测试相同场景产生相同哈希"""
        # 创建两个内容相同的场景文件
        scene1 = tmp_path / "scene1.json"
        scene2 = tmp_path / "scene2.json"

        scene1.write_text(json.dumps(sample_scenario_data))
        scene2.write_text(json.dumps(sample_scenario_data))

        calculator = FingerprintCalculator()
        fp1 = calculator.calculate(str(scene1))
        fp2 = calculator.calculate(str(scene2))

        assert fp1.full_hash == fp2.full_hash
        assert fp1.satellites.hash_value == fp2.satellites.hash_value

    def test_different_satellites_different_hash(self, tmp_path, sample_scenario_data):
        """测试不同卫星配置产生不同哈希"""
        scene1 = tmp_path / "scene1.json"
        scene2 = tmp_path / "scene2.json"

        scene1.write_text(json.dumps(sample_scenario_data))

        # 修改卫星配置
        sample_scenario_data["satellites"][0]["orbit"]["inclination"] = 60.0
        scene2.write_text(json.dumps(sample_scenario_data))

        calculator = FingerprintCalculator()
        fp1 = calculator.calculate(str(scene1))
        fp2 = calculator.calculate(str(scene2))

        assert fp1.full_hash != fp2.full_hash
        assert fp1.satellites.hash_value != fp2.satellites.hash_value

    def test_component_isolation(self, sample_scenario_data):
        """测试组件哈希相互独立"""
        calculator = FingerprintCalculator()
        fingerprint = calculator.calculate_from_data(sample_scenario_data)

        # 各组件哈希应该不同（内容不同）
        hashes = [
            fingerprint.satellites.hash_value,
            fingerprint.ground_stations.hash_value,
            fingerprint.targets.hash_value,
            fingerprint.time_range.hash_value
        ]

        # 至少大部分应该不同
        unique_hashes = set(hashes)
        assert len(unique_hashes) >= 3  # 允许时间范围和某个组件可能相似


class TestFingerprintComparator:
    """测试指纹比较器"""

    @pytest.fixture
    def base_fingerprint(self):
        """创建基础指纹"""
        return ScenarioFingerprint(
            full_hash="hash1abcdef1234567890",
            satellites=ComponentHash("sat_hash1", "satellites", 2, {"SAT-01", "SAT-02"}),
            ground_stations=ComponentHash("gs_hash1", "ground_stations", 1, {"GS-01"}),
            targets=ComponentHash("tgt_hash1", "targets", 10, {f"TGT-{i:02d}" for i in range(1, 11)}),
            time_range=ComponentHash("tr_hash1", "time_range", 1, set()),
            scenario_name="Base Scenario",
            created_at=datetime.now()
        )

    def test_identical_scenes(self, base_fingerprint):
        """测试相同场景"""
        comparator = FingerprintComparator()
        result = comparator.compare(base_fingerprint, base_fingerprint)

        assert result['identical'] is True
        assert result['same_satellites'] is True
        assert result['recommendation'] == 'Scenes are identical, reuse full cache'
        assert 'all' in result['reusable_components']

    def test_different_satellites(self, base_fingerprint):
        """测试不同卫星"""
        other = ScenarioFingerprint(
            full_hash="hash2abcdef1234567890",
            satellites=ComponentHash("sat_hash2", "satellites", 2, {"SAT-03", "SAT-04"}),
            ground_stations=base_fingerprint.ground_stations,
            targets=base_fingerprint.targets,
            time_range=base_fingerprint.time_range,
            scenario_name="Other Scenario",
            created_at=datetime.now()
        )

        comparator = FingerprintComparator()
        result = comparator.compare(base_fingerprint, other)

        assert result['identical'] is False
        assert result['same_satellites'] is False
        assert len(result['common_satellites']) == 0
        assert 'No cache reuse possible' in result['recommendation']

    def test_partial_common_satellites(self, base_fingerprint):
        """测试部分共同卫星"""
        other = ScenarioFingerprint(
            full_hash="hash2abcdef1234567890",
            satellites=ComponentHash("sat_hash2", "satellites", 3, {"SAT-01", "SAT-03", "SAT-04"}),
            ground_stations=base_fingerprint.ground_stations,
            targets=base_fingerprint.targets,
            time_range=base_fingerprint.time_range,
            scenario_name="Other Scenario",
            created_at=datetime.now()
        )

        comparator = FingerprintComparator()
        result = comparator.compare(base_fingerprint, other)

        assert result['identical'] is False
        assert result['same_satellites'] is False
        assert len(result['common_satellites']) == 1
        assert 'SAT-01' in result['common_satellites']
        assert 'partial_orbit' in result['reusable_components']

    def test_same_satellites_different_targets(self, base_fingerprint):
        """测试相同卫星不同目标"""
        other = ScenarioFingerprint(
            full_hash="hash2abcdef1234567890",
            satellites=base_fingerprint.satellites,
            ground_stations=base_fingerprint.ground_stations,
            targets=ComponentHash("tgt_hash2", "targets", 5, {f"TGT-{i:02d}" for i in range(11, 16)}),
            time_range=base_fingerprint.time_range,
            scenario_name="Other Scenario",
            created_at=datetime.now()
        )

        comparator = FingerprintComparator()
        result = comparator.compare(base_fingerprint, other)

        assert result['identical'] is False
        assert result['same_satellites'] is True
        assert result['same_targets'] is False
        # 由于时间范围相同，应该可以复用轨道
        assert 'orbit' in result['reusable_components'] or 'satellite_config' in result['reusable_components']
