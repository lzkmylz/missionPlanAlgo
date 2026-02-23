"""
ConfigLoader模块的单元测试

TDD流程：
1. 先写测试（RED）
2. 运行测试 - 验证失败
3. 实现代码（GREEN）
4. 运行测试 - 验证通过
5. 重构（IMPROVE）
"""

import json
import os
import tempfile
from unittest import TestCase, mock

import pytest


class TestConfigLoader(TestCase):
    """ConfigLoader测试类"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_temp_file(self, content: str, suffix: str = ".json") -> str:
        """创建临时文件"""
        fd, path = tempfile.mkstemp(suffix=suffix, dir=self.temp_dir)
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        return path

    # ==================== 基础加载测试 ====================

    def test_load_json_file(self):
        """测试加载JSON配置文件"""
        from utils.config_loader import ConfigLoader

        config_data = {"name": "test", "value": 123}
        file_path = self._create_temp_file(json.dumps(config_data), ".json")

        loader = ConfigLoader()
        result = loader.load(file_path, format="json")

        self.assertEqual(result["name"], "test")
        self.assertEqual(result["value"], 123)

    def test_load_yaml_file(self):
        """测试加载YAML配置文件"""
        from utils.config_loader import ConfigLoader

        yaml_content = """
name: test
value: 123
nested:
  key: value
"""
        file_path = self._create_temp_file(yaml_content, ".yaml")

        loader = ConfigLoader()
        result = loader.load(file_path, format="yaml")

        self.assertEqual(result["name"], "test")
        self.assertEqual(result["value"], 123)
        self.assertEqual(result["nested"]["key"], "value")

    def test_load_ini_file(self):
        """测试加载INI配置文件"""
        from utils.config_loader import ConfigLoader

        ini_content = """[section1]
key1 = value1
key2 = 123

[section2]
enabled = true
"""
        file_path = self._create_temp_file(ini_content, ".ini")

        loader = ConfigLoader()
        result = loader.load(file_path, format="ini")

        self.assertEqual(result["section1"]["key1"], "value1")
        self.assertEqual(result["section1"]["key2"], "123")
        self.assertEqual(result["section2"]["enabled"], "true")

    def test_load_auto_detect_format(self):
        """测试自动检测文件格式"""
        from utils.config_loader import ConfigLoader

        config_data = {"name": "test", "value": 123}
        file_path = self._create_temp_file(json.dumps(config_data), ".json")

        loader = ConfigLoader()
        result = loader.load(file_path)  # format="auto"

        self.assertEqual(result["name"], "test")

    # ==================== 错误处理测试 ====================

    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        from utils.config_loader import ConfigLoader, ConfigLoadError

        loader = ConfigLoader()
        with self.assertRaises(ConfigLoadError):
            loader.load("/nonexistent/path/config.json")

    def test_load_invalid_json(self):
        """测试加载无效的JSON"""
        from utils.config_loader import ConfigLoader, ConfigLoadError

        file_path = self._create_temp_file("{invalid json", ".json")

        loader = ConfigLoader()
        with self.assertRaises(ConfigLoadError):
            loader.load(file_path, format="json")

    def test_load_invalid_yaml(self):
        """测试加载无效的YAML"""
        from utils.config_loader import ConfigLoader, ConfigLoadError

        yaml_content = "{invalid: yaml: content:"
        file_path = self._create_temp_file(yaml_content, ".yaml")

        loader = ConfigLoader()
        with self.assertRaises(ConfigLoadError):
            loader.load(file_path, format="yaml")

    def test_load_unsupported_format(self):
        """测试加载不支持的格式"""
        from utils.config_loader import ConfigLoader, ConfigLoadError

        file_path = self._create_temp_file("content", ".txt")

        loader = ConfigLoader()
        with self.assertRaises(ConfigLoadError):
            loader.load(file_path, format="txt")

    # ==================== 环境变量覆盖测试 ====================

    def test_load_from_env(self):
        """测试从环境变量加载配置"""
        from utils.config_loader import ConfigLoader

        with mock.patch.dict(os.environ, {
            "APP_NAME": "test_app",
            "APP_PORT": "8080",
            "APP_DEBUG": "true"
        }):
            loader = ConfigLoader()
            result = loader.load_from_env("APP_")

            self.assertEqual(result["name"], "test_app")
            self.assertEqual(result["port"], "8080")
            self.assertEqual(result["debug"], "true")

    def test_load_from_env_empty_prefix(self):
        """测试空前缀加载环境变量"""
        from utils.config_loader import ConfigLoader

        with mock.patch.dict(os.environ, {"TEST_VAR": "value"}):
            loader = ConfigLoader()
            result = loader.load_from_env("")
            # 环境变量名会被转换为小写
            self.assertIn("test_var", result)

    def test_load_from_env_no_matching_vars(self):
        """测试没有匹配的环境变量"""
        from utils.config_loader import ConfigLoader

        loader = ConfigLoader()
        result = loader.load_from_env("NONEXISTENT_")
        self.assertEqual(result, {})

    # ==================== 配置验证测试 ====================

    def test_validate_success(self):
        """测试配置验证成功"""
        from utils.config_loader import ConfigLoader

        config = {"name": "test", "port": 8080}
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "port": {"type": "integer"}
            },
            "required": ["name", "port"]
        }

        loader = ConfigLoader()
        is_valid, errors = loader.validate(config, schema)

        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    def test_validate_failure(self):
        """测试配置验证失败"""
        from utils.config_loader import ConfigLoader

        config = {"name": "test", "port": "not_a_number"}
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "port": {"type": "integer"}
            },
            "required": ["name", "port"]
        }

        loader = ConfigLoader()
        is_valid, errors = loader.validate(config, schema)

        self.assertFalse(is_valid)
        self.assertTrue(len(errors) > 0)

    def test_validate_missing_required_field(self):
        """测试缺少必需字段"""
        from utils.config_loader import ConfigLoader

        config = {"name": "test"}  # 缺少port
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "port": {"type": "integer"}
            },
            "required": ["name", "port"]
        }

        loader = ConfigLoader()
        is_valid, errors = loader.validate(config, schema)

        self.assertFalse(is_valid)

    def test_validate_empty_config(self):
        """测试验证空配置"""
        from utils.config_loader import ConfigLoader

        config = {}
        schema = {"type": "object"}

        loader = ConfigLoader()
        is_valid, errors = loader.validate(config, schema)

        self.assertTrue(is_valid)

    def test_validate_empty_schema(self):
        """测试验证空schema"""
        from utils.config_loader import ConfigLoader

        config = {"any": "value"}
        schema = {}

        loader = ConfigLoader()
        is_valid, errors = loader.validate(config, schema)

        self.assertTrue(is_valid)

    # ==================== 边缘情况测试 ====================

    def test_load_empty_json_file(self):
        """测试加载空JSON文件"""
        from utils.config_loader import ConfigLoader, ConfigLoadError

        file_path = self._create_temp_file("", ".json")

        loader = ConfigLoader()
        with self.assertRaises(ConfigLoadError):
            loader.load(file_path, format="json")

    def test_load_empty_yaml_file(self):
        """测试加载空YAML文件"""
        from utils.config_loader import ConfigLoader

        file_path = self._create_temp_file("", ".yaml")

        loader = ConfigLoader()
        result = loader.load(file_path, format="yaml")
        self.assertIsNone(result)

    def test_load_nested_config(self):
        """测试加载嵌套配置"""
        from utils.config_loader import ConfigLoader

        config_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "deep"
                    }
                }
            }
        }
        file_path = self._create_temp_file(json.dumps(config_data), ".json")

        loader = ConfigLoader()
        result = loader.load(file_path)

        self.assertEqual(result["level1"]["level2"]["level3"]["value"], "deep")

    def test_load_unicode_content(self):
        """测试加载包含Unicode的配置"""
        from utils.config_loader import ConfigLoader

        config_data = {"name": "测试中文", "emoji": "🚀"}
        file_path = self._create_temp_file(json.dumps(config_data, ensure_ascii=False), ".json")

        loader = ConfigLoader()
        result = loader.load(file_path)

        self.assertEqual(result["name"], "测试中文")
        self.assertEqual(result["emoji"], "🚀")

    def test_load_large_config(self):
        """测试加载大配置文件"""
        from utils.config_loader import ConfigLoader

        config_data = {"items": [{"id": i, "data": "x" * 100} for i in range(1000)]}
        file_path = self._create_temp_file(json.dumps(config_data), ".json")

        loader = ConfigLoader()
        result = loader.load(file_path)

        self.assertEqual(len(result["items"]), 1000)


class TestConfigLoaderIntegration(TestCase):
    """ConfigLoader集成测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_temp_file(self, content: str, suffix: str = ".json") -> str:
        fd, path = tempfile.mkstemp(suffix=suffix, dir=self.temp_dir)
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        return path

    def test_full_workflow(self):
        """测试完整工作流程"""
        from utils.config_loader import ConfigLoader

        # 创建配置文件
        config_data = {
            "app": {
                "name": "myapp",
                "version": "1.0.0"
            },
            "database": {
                "host": "localhost",
                "port": 5432
            }
        }
        file_path = self._create_temp_file(json.dumps(config_data), ".json")

        # 加载配置
        loader = ConfigLoader()
        config = loader.load(file_path)

        # 验证配置
        schema = {
            "type": "object",
            "properties": {
                "app": {"type": "object"},
                "database": {"type": "object"}
            },
            "required": ["app", "database"]
        }
        is_valid, errors = loader.validate(config, schema)

        self.assertTrue(is_valid)
        self.assertEqual(config["app"]["name"], "myapp")


if __name__ == "__main__":
    import unittest
    unittest.main()
