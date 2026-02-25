"""
Orekit配置模块测试

TDD测试套件 - 测试orekit_config模块的配置管理功能
"""

import pytest
import os
from unittest.mock import patch, MagicMock


class TestOrekitConfigImports:
    """测试配置模块导入"""

    def test_config_module_imports(self):
        """测试配置模块可以正确导入"""
        try:
            from core.orbit.visibility.orekit_config import DEFAULT_OREKIT_CONFIG
            assert True
        except ImportError as e:
            pytest.fail(f"无法导入orekit_config模块: {e}")

    def test_default_config_exists(self):
        """测试默认配置字典存在"""
        from core.orbit.visibility.orekit_config import DEFAULT_OREKIT_CONFIG
        assert DEFAULT_OREKIT_CONFIG is not None
        assert isinstance(DEFAULT_OREKIT_CONFIG, dict)


class TestOrekitConfigStructure:
    """测试配置结构"""

    def test_jvm_config_section(self):
        """测试JVM配置部分"""
        from core.orbit.visibility.orekit_config import DEFAULT_OREKIT_CONFIG

        assert 'jvm' in DEFAULT_OREKIT_CONFIG
        jvm_config = DEFAULT_OREKIT_CONFIG['jvm']

        assert 'java_home' in jvm_config
        assert 'classpath' in jvm_config
        assert 'max_memory' in jvm_config

        assert isinstance(jvm_config['classpath'], list)
        assert jvm_config['max_memory'] in ['1g', '2g', '4g', '8g'] or jvm_config['max_memory'].endswith('g')

    def test_data_config_section(self):
        """测试数据配置部分"""
        from core.orbit.visibility.orekit_config import DEFAULT_OREKIT_CONFIG

        assert 'data' in DEFAULT_OREKIT_CONFIG
        data_config = DEFAULT_OREKIT_CONFIG['data']

        assert 'root_dir' in data_config
        assert 'iers_dir' in data_config
        assert 'gravity_dir' in data_config
        assert 'ephemeris_dir' in data_config

        assert isinstance(data_config['root_dir'], str)

    def test_propagator_config_section(self):
        """测试传播器配置部分"""
        from core.orbit.visibility.orekit_config import DEFAULT_OREKIT_CONFIG

        assert 'propagator' in DEFAULT_OREKIT_CONFIG
        prop_config = DEFAULT_OREKIT_CONFIG['propagator']

        assert 'integrator' in prop_config
        assert 'min_step' in prop_config
        assert 'max_step' in prop_config
        assert 'position_tolerance' in prop_config

        assert prop_config['integrator'] in ['DormandPrince853', 'DormandPrince54', 'GraggBulirschStoer']
        assert prop_config['min_step'] > 0
        assert prop_config['max_step'] > prop_config['min_step']
        assert prop_config['position_tolerance'] > 0

    def test_perturbations_config_section(self):
        """测试摄动力配置部分"""
        from core.orbit.visibility.orekit_config import DEFAULT_OREKIT_CONFIG

        assert 'perturbations' in DEFAULT_OREKIT_CONFIG
        pert_config = DEFAULT_OREKIT_CONFIG['perturbations']

        # 地球引力
        assert 'earth_gravity' in pert_config
        assert 'enabled' in pert_config['earth_gravity']
        assert 'model' in pert_config['earth_gravity']
        assert 'degree' in pert_config['earth_gravity']
        assert 'order' in pert_config['earth_gravity']

        # 大气阻力
        assert 'drag' in pert_config
        assert 'enabled' in pert_config['drag']
        assert 'model' in pert_config['drag']
        assert 'cd' in pert_config['drag']
        assert 'area' in pert_config['drag']

        # 太阳光压
        assert 'solar_radiation' in pert_config
        assert 'enabled' in pert_config['solar_radiation']
        assert 'cr' in pert_config['solar_radiation']
        assert 'area' in pert_config['solar_radiation']

        # 第三体引力
        assert 'third_body' in pert_config
        assert 'enabled' in pert_config['third_body']
        assert 'bodies' in pert_config['third_body']
        assert isinstance(pert_config['third_body']['bodies'], list)

        # 相对论效应
        assert 'relativity' in pert_config
        assert 'enabled' in pert_config['relativity']


class TestOrekitConfigValues:
    """测试配置值"""

    def test_default_values_reasonable(self):
        """测试默认值合理"""
        from core.orbit.visibility.orekit_config import DEFAULT_OREKIT_CONFIG

        # JVM内存应该在合理范围
        jvm_memory = DEFAULT_OREKIT_CONFIG['jvm']['max_memory']
        assert jvm_memory in ['512m', '1g', '2g', '4g', '8g'] or jvm_memory.endswith(('m', 'g'))

        # 步长应该在合理范围
        min_step = DEFAULT_OREKIT_CONFIG['propagator']['min_step']
        max_step = DEFAULT_OREKIT_CONFIG['propagator']['max_step']
        assert 0.0001 <= min_step <= 1.0
        assert 1.0 <= max_step <= 1000.0

        # 引力场阶数应该在合理范围
        degree = DEFAULT_OREKIT_CONFIG['perturbations']['earth_gravity']['degree']
        order = DEFAULT_OREKIT_CONFIG['perturbations']['earth_gravity']['order']
        assert 2 <= degree <= 360
        assert 2 <= order <= 360
        assert order <= degree

    def test_classpath_entries(self):
        """测试classpath条目"""
        from core.orbit.visibility.orekit_config import DEFAULT_OREKIT_CONFIG

        classpath = DEFAULT_OREKIT_CONFIG['jvm']['classpath']
        assert len(classpath) > 0

        # 应该包含orekit jar
        orekit_jars = [jar for jar in classpath if 'orekit' in jar.lower()]
        assert len(orekit_jars) > 0

        # 应该包含hipparchus jar
        hipparchus_jars = [jar for jar in classpath if 'hipparchus' in jar.lower()]
        assert len(hipparchus_jars) > 0

    def test_third_body_bodies(self):
        """测试第三体引力配置"""
        from core.orbit.visibility.orekit_config import DEFAULT_OREKIT_CONFIG

        bodies = DEFAULT_OREKIT_CONFIG['perturbations']['third_body']['bodies']
        assert len(bodies) > 0

        # 应该包含太阳和月球
        valid_bodies = ['SUN', 'MOON', 'MERCURY', 'VENUS', 'MARS', 'JUPITER', 'SATURN']
        for body in bodies:
            assert body.upper() in valid_bodies


class TestOrekitConfigHelperFunctions:
    """测试配置辅助函数"""

    def test_get_orekit_data_dir_function_exists(self):
        """测试获取Orekit数据目录函数存在"""
        try:
            from core.orbit.visibility.orekit_config import get_orekit_data_dir
            assert callable(get_orekit_data_dir)
        except ImportError:
            pytest.fail("get_orekit_data_dir函数不存在")

    def test_get_orekit_data_dir_returns_string(self):
        """测试获取Orekit数据目录返回字符串"""
        from core.orbit.visibility.orekit_config import get_orekit_data_dir

        result = get_orekit_data_dir()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_orekit_data_dir_respects_env_var(self):
        """测试环境变量可以覆盖数据目录"""
        from core.orbit.visibility.orekit_config import get_orekit_data_dir

        with patch.dict(os.environ, {'OREKIT_DATA_DIR': '/custom/path'}):
            result = get_orekit_data_dir()
            assert '/custom/path' in result or result == '/custom/path'

    def test_merge_config_function_exists(self):
        """测试配置合并函数存在"""
        try:
            from core.orbit.visibility.orekit_config import merge_config
            assert callable(merge_config)
        except ImportError:
            pytest.fail("merge_config函数不存在")

    def test_merge_config_updates_values(self):
        """测试配置合并可以更新值"""
        from core.orbit.visibility.orekit_config import merge_config, DEFAULT_OREKIT_CONFIG

        custom_config = {
            'jvm': {
                'max_memory': '4g'
            },
            'propagator': {
                'min_step': 0.01
            }
        }

        merged = merge_config(custom_config)

        assert merged['jvm']['max_memory'] == '4g'
        assert merged['propagator']['min_step'] == 0.01

    def test_merge_config_preserves_defaults(self):
        """测试配置合并保留未修改的默认值"""
        from core.orbit.visibility.orekit_config import merge_config, DEFAULT_OREKIT_CONFIG

        custom_config = {
            'jvm': {
                'max_memory': '4g'
            }
        }

        merged = merge_config(custom_config)

        # 未修改的值应该保留默认值
        assert merged['propagator']['integrator'] == DEFAULT_OREKIT_CONFIG['propagator']['integrator']
        assert merged['data']['root_dir'] == DEFAULT_OREKIT_CONFIG['data']['root_dir']


class TestOrekitConfigEdgeCases:
    """测试配置边界情况"""

    def test_merge_config_with_empty_dict(self):
        """测试合并空配置"""
        from core.orbit.visibility.orekit_config import merge_config, DEFAULT_OREKIT_CONFIG

        merged = merge_config({})
        assert merged == DEFAULT_OREKIT_CONFIG

    def test_merge_config_with_none(self):
        """测试合并None配置"""
        from core.orbit.visibility.orekit_config import merge_config, DEFAULT_OREKIT_CONFIG

        merged = merge_config(None)
        assert merged == DEFAULT_OREKIT_CONFIG

    def test_merge_config_deep_merge(self):
        """测试深度合并"""
        from core.orbit.visibility.orekit_config import merge_config

        custom_config = {
            'perturbations': {
                'earth_gravity': {
                    'degree': 72
                }
            }
        }

        merged = merge_config(custom_config)

        # 深度合并应该只更新指定字段
        assert merged['perturbations']['earth_gravity']['degree'] == 72
        # 其他字段应该保留
        assert 'order' in merged['perturbations']['earth_gravity']
        assert 'enabled' in merged['perturbations']['earth_gravity']


class TestOrekitConfigAdditionalFunctions:
    """测试额外的配置函数"""

    def test_get_jvm_classpath_function_exists(self):
        """测试get_jvm_classpath函数存在"""
        try:
            from core.orbit.visibility.orekit_config import get_jvm_classpath
            assert callable(get_jvm_classpath)
        except ImportError:
            pytest.fail("get_jvm_classpath函数不存在")

    def test_get_jvm_classpath_returns_string(self):
        """测试get_jvm_classpath返回字符串"""
        from core.orbit.visibility.orekit_config import get_jvm_classpath

        result = get_jvm_classpath()
        assert isinstance(result, str)

    def test_get_jvm_classpath_with_default_config(self):
        """测试get_jvm_classpath使用默认配置"""
        from core.orbit.visibility.orekit_config import get_jvm_classpath

        result = get_jvm_classpath()
        # 应该包含一些jar文件路径
        assert 'orekit' in result.lower() or 'hipparchus' in result.lower() or len(result) == 0

    def test_get_jvm_classpath_with_custom_config(self):
        """测试get_jvm_classpath使用自定义配置"""
        from core.orbit.visibility.orekit_config import get_jvm_classpath

        custom_config = {
            'jvm': {
                'classpath': ['/path/to/test1.jar', '/path/to/test2.jar']
            }
        }

        result = get_jvm_classpath(custom_config)
        assert 'test1.jar' in result
        assert 'test2.jar' in result

    def test_validate_config_function_exists(self):
        """测试validate_config函数存在"""
        try:
            from core.orbit.visibility.orekit_config import validate_config
            assert callable(validate_config)
        except ImportError:
            pytest.fail("validate_config函数不存在")

    def test_validate_config_valid(self):
        """测试验证有效配置"""
        from core.orbit.visibility.orekit_config import validate_config, DEFAULT_OREKIT_CONFIG

        is_valid, errors = validate_config(DEFAULT_OREKIT_CONFIG)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_config_missing_jvm(self):
        """测试验证缺少jvm配置"""
        from core.orbit.visibility.orekit_config import validate_config

        config = {'data': {}, 'propagator': {}, 'perturbations': {}}
        is_valid, errors = validate_config(config)
        assert is_valid is False
        assert any('jvm' in error.lower() for error in errors)

    def test_validate_config_missing_data(self):
        """测试验证缺少data配置"""
        from core.orbit.visibility.orekit_config import validate_config

        config = {
            'jvm': {'classpath': [], 'max_memory': '2g'},
            'propagator': {},
            'perturbations': {}
        }
        is_valid, errors = validate_config(config)
        assert is_valid is False
        assert any('data' in error.lower() for error in errors)

    def test_validate_config_missing_classpath(self):
        """测试验证缺少classpath配置"""
        from core.orbit.visibility.orekit_config import validate_config

        config = {
            'jvm': {'max_memory': '2g'},
            'data': {'root_dir': '/test', 'iers_dir': 'IERS', 'gravity_dir': 'EGM96', 'ephemeris_dir': 'DE440'},
            'propagator': {},
            'perturbations': {}
        }
        is_valid, errors = validate_config(config)
        assert is_valid is False
        assert any('classpath' in error.lower() for error in errors)

    def test_validate_config_invalid_propagator_steps(self):
        """测试验证传播器步长配置"""
        from core.orbit.visibility.orekit_config import validate_config

        config = {
            'jvm': {'classpath': [], 'max_memory': '2g'},
            'data': {'root_dir': '/test', 'iers_dir': 'IERS', 'gravity_dir': 'EGM96', 'ephemeris_dir': 'DE440'},
            'propagator': {'min_step': 100, 'max_step': 10},  # min > max
            'perturbations': {}
        }
        is_valid, errors = validate_config(config)
        assert is_valid is False
        assert any('min_step' in error.lower() and 'max_step' in error.lower() for error in errors)

    def test_validate_config_zero_steps(self):
        """测试验证零步长配置"""
        from core.orbit.visibility.orekit_config import validate_config

        config = {
            'jvm': {'classpath': [], 'max_memory': '2g'},
            'data': {'root_dir': '/test', 'iers_dir': 'IERS', 'gravity_dir': 'EGM96', 'ephemeris_dir': 'DE440'},
            'propagator': {'min_step': 0, 'max_step': 0},
            'perturbations': {}
        }
        is_valid, errors = validate_config(config)
        assert is_valid is False

    def test_validate_config_classpath_not_list(self):
        """测试验证classpath不是列表"""
        from core.orbit.visibility.orekit_config import validate_config

        config = {
            'jvm': {'classpath': '/not/a/list', 'max_memory': '2g'},
            'data': {'root_dir': '/test', 'iers_dir': 'IERS', 'gravity_dir': 'EGM96', 'ephemeris_dir': 'DE440'},
            'propagator': {},
            'perturbations': {}
        }
        is_valid, errors = validate_config(config)
        assert is_valid is False
        assert any('classpath' in error.lower() and '列表' in error for error in errors)

    def test_validate_config_missing_max_memory(self):
        """测试验证缺少max_memory配置"""
        from core.orbit.visibility.orekit_config import validate_config

        config = {
            'jvm': {'classpath': []},
            'data': {'root_dir': '/test', 'iers_dir': 'IERS', 'gravity_dir': 'EGM96', 'ephemeris_dir': 'DE440'},
            'propagator': {},
            'perturbations': {}
        }
        is_valid, errors = validate_config(config)
        assert is_valid is False
        assert any('max_memory' in error for error in errors)

    def test_validate_config_missing_perturbations(self):
        """测试验证缺少perturbations配置"""
        from core.orbit.visibility.orekit_config import validate_config

        config = {
            'jvm': {'classpath': [], 'max_memory': '2g'},
            'data': {'root_dir': '/test', 'iers_dir': 'IERS', 'gravity_dir': 'EGM96', 'ephemeris_dir': 'DE440'},
            'propagator': {}
        }
        is_valid, errors = validate_config(config)
        assert is_valid is False
        assert any('perturbations' in error.lower() for error in errors)

    def test_validate_config_missing_propagator(self):
        """测试验证缺少propagator配置"""
        from core.orbit.visibility.orekit_config import validate_config

        config = {
            'jvm': {'classpath': [], 'max_memory': '2g'},
            'data': {'root_dir': '/test', 'iers_dir': 'IERS', 'gravity_dir': 'EGM96', 'ephemeris_dir': 'DE440'},
            'perturbations': {}
        }
        is_valid, errors = validate_config(config)
        assert is_valid is False
        assert any('propagator' in error.lower() for error in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
