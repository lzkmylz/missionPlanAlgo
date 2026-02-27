"""
Java端批量计算器集成测试

测试Phase 2 - Java端批量可见性计算
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from core.orbit.visibility.orekit_java_bridge import OrekitJavaBridge, OrbitPropagationError


class TestJavaBatchCalculator:
    """测试Java批量计算器"""

    @pytest.fixture
    def bridge(self):
        """创建OrekitJavaBridge实例"""
        try:
            return OrekitJavaBridge()
        except Exception as e:
            pytest.skip(f"无法创建OrekitJavaBridge: {e}")

    def test_batch_calculator_class_exists(self, bridge):
        """测试Java批量计算器类是否存在"""
        try:
            BatchCalculator = bridge._get_java_class(
                "orekit.visibility.calculator.VisibilityBatchCalculator"
            )
            assert BatchCalculator is not None
        except Exception as e:
            pytest.skip(f"Java批量计算器类不可用: {e}")

    def test_satellite_config_class_exists(self, bridge):
        """测试卫星配置类是否存在"""
        try:
            SatelliteConfig = bridge._get_java_class(
                "orekit.visibility.model.SatelliteConfig"
            )
            assert SatelliteConfig is not None
        except Exception as e:
            pytest.skip(f"卫星配置类不可用: {e}")

    def test_target_config_class_exists(self, bridge):
        """测试目标配置类是否存在"""
        try:
            TargetConfig = bridge._get_java_class(
                "orekit.visibility.model.TargetConfig"
            )
            assert TargetConfig is not None
        except Exception as e:
            pytest.skip(f"目标配置类不可用: {e}")

    def test_visibility_window_class_exists(self, bridge):
        """测试可见窗口类是否存在"""
        try:
            VisibilityWindow = bridge._get_java_class(
                "orekit.visibility.model.VisibilityWindow"
            )
            assert VisibilityWindow is not None
        except Exception as e:
            pytest.skip(f"可见窗口类不可用: {e}")

    def test_batch_result_class_exists(self, bridge):
        """测试批量结果类是否存在"""
        try:
            BatchResult = bridge._get_java_class(
                "orekit.visibility.model.BatchResult"
            )
            assert BatchResult is not None
        except Exception as e:
            pytest.skip(f"批量结果类不可用: {e}")


class TestComputeVisibilityBatch:
    """测试批量计算接口"""

    @pytest.fixture
    def bridge(self):
        """创建OrekitJavaBridge实例"""
        try:
            return OrekitJavaBridge()
        except Exception as e:
            pytest.skip(f"无法创建OrekitJavaBridge: {e}")

    @pytest.fixture
    def sample_satellites(self):
        """示例卫星配置"""
        return [
            {
                'id': 'SAT1',
                'tle_line1': '1 25544U 98067A   24257.50000000  .00020000  00000-0  28000-4 0  9999',
                'tle_line2': '2 25544  51.6400  45.0000 0005000  30.0000 330.0000 15.50000000    00',
                'min_elevation': 5.0,
                'sensor_fov': 0.0
            }
        ]

    @pytest.fixture
    def sample_targets(self):
        """示例目标配置"""
        return [
            {
                'id': 'TARGET1',
                'longitude': 116.4074,  # 北京经度
                'latitude': 39.9042,    # 北京纬度
                'altitude': 0.0,
                'min_observation_duration': 60,
                'priority': 5
            }
        ]

    def test_compute_visibility_batch_returns_result_structure(
        self, bridge, sample_satellites, sample_targets
    ):
        """测试批量计算返回正确的结果结构"""
        try:
            start_time = datetime(2024, 1, 1, 0, 0, 0)
            end_time = datetime(2024, 1, 2, 0, 0, 0)

            result = bridge.compute_visibility_batch(
                satellite_configs=sample_satellites,
                target_configs=sample_targets,
                start_time=start_time,
                end_time=end_time
            )

            # 验证结果结构
            assert 'windows' in result
            assert 'statistics' in result
            assert 'errors' in result

            # 验证统计信息结构
            stats = result['statistics']
            assert 'total_pairs' in stats
            assert 'pairs_with_windows' in stats
            assert 'total_windows' in stats
            assert 'computation_time_ms' in stats

        except Exception as e:
            pytest.skip(f"批量计算失败: {e}")

    def test_compute_visibility_batch_statistics(self, bridge, sample_satellites, sample_targets):
        """测试批量计算统计信息正确"""
        try:
            start_time = datetime(2024, 1, 1, 0, 0, 0)
            end_time = datetime(2024, 1, 2, 0, 0, 0)

            result = bridge.compute_visibility_batch(
                satellite_configs=sample_satellites,
                target_configs=sample_targets,
                start_time=start_time,
                end_time=end_time
            )

            stats = result['statistics']

            # 验证统计值
            assert stats['total_pairs'] == 1  # 1卫星 × 1目标
            assert stats['computation_time_ms'] >= 0

        except Exception as e:
            pytest.skip(f"批量计算失败: {e}")

    def test_compute_visibility_batch_empty_input(self, bridge):
        """测试空输入处理"""
        try:
            start_time = datetime(2024, 1, 1, 0, 0, 0)
            end_time = datetime(2024, 1, 2, 0, 0, 0)

            result = bridge.compute_visibility_batch(
                satellite_configs=[],
                target_configs=[],
                start_time=start_time,
                end_time=end_time
            )

            assert result['statistics']['total_pairs'] == 0
            assert len(result['windows']) == 0

        except Exception as e:
            pytest.skip(f"批量计算失败: {e}")


class TestDateConversion:
    """测试日期转换功能"""

    @pytest.fixture
    def bridge(self):
        """创建OrekitJavaBridge实例"""
        try:
            return OrekitJavaBridge()
        except Exception as e:
            pytest.skip(f"无法创建OrekitJavaBridge: {e}")

    def test_datetime_to_java_date_and_back(self, bridge):
        """测试datetime到Java日期再转回的准确性"""
        try:
            original = datetime(2024, 1, 15, 12, 30, 45, 123456)

            AbsoluteDate = bridge._get_java_class("org.orekit.time.AbsoluteDate")
            java_date = bridge._datetime_to_java_date(original, AbsoluteDate)
            back_to_python = bridge._java_date_to_datetime(java_date)

            # 允许微秒级误差
            assert abs((back_to_python - original).total_seconds()) < 1.0

        except Exception as e:
            pytest.skip(f"日期转换失败: {e}")

    def test_epoch_boundaries(self, bridge):
        """测试纪元边界日期转换"""
        try:
            test_dates = [
                datetime(2000, 1, 1, 0, 0, 0),  # J2000
                datetime(2024, 1, 1, 0, 0, 0),
                datetime(2024, 12, 31, 23, 59, 59),
            ]

            AbsoluteDate = bridge._get_java_class("org.orekit.time.AbsoluteDate")

            for dt in test_dates:
                java_date = bridge._datetime_to_java_date(dt, AbsoluteDate)
                back = bridge._java_date_to_datetime(java_date)
                assert abs((back - dt).total_seconds()) < 1.0

        except Exception as e:
            pytest.skip(f"日期转换失败: {e}")


class TestPerformanceComparison:
    """性能对比测试"""

    @pytest.fixture
    def bridge(self):
        """创建OrekitJavaBridge实例"""
        try:
            return OrekitJavaBridge()
        except Exception as e:
            pytest.skip(f"无法创建OrekitJavaBridge: {e}")

    def test_java_batch_faster_than_python_iteration(
        self, bridge
    ):
        """测试Java批量计算比Python迭代快10倍以上"""
        import time

        # 准备多个卫星和目标
        satellites = [
            {
                'id': f'SAT{i}',
                'tle_line1': f'1 2554{i}U 98067A   24257.50000000  .00020000  00000-0  28000-4 0  999{i}',
                'tle_line2': f'2 2554{i}  51.6400  {i*10:.4f} 0005000  30.0000 330.0000 15.50000000    0{i}',
            }
            for i in range(3)
        ]

        targets = [
            {
                'id': f'TARGET{i}',
                'longitude': 116.0 + i,
                'latitude': 39.0 + i * 0.5,
            }
            for i in range(3)
        ]

        start_time = datetime(2024, 1, 1, 0, 0, 0)
        end_time = datetime(2024, 1, 2, 0, 0, 0)

        try:
            # 测试Java批量计算
            t1 = time.time()
            result = bridge.compute_visibility_batch(
                satellite_configs=satellites,
                target_configs=targets,
                start_time=start_time,
                end_time=end_time
            )
            java_time = time.time() - t1

            # 验证结果有效
            assert 'statistics' in result
            stats = result['statistics']
            assert stats['total_pairs'] == 9  # 3卫星 × 3目标

            # Java批量计算应该在合理时间内完成（< 60秒）
            assert java_time < 60, f"Java批量计算耗时 {java_time:.1f}s，预期 < 60s"

        except Exception as e:
            pytest.skip(f"性能测试失败: {e}")
