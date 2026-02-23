"""
通用配置加载器

功能：
- 支持加载JSON配置文件
- 支持加载YAML配置文件
- 支持加载INI配置文件
- 支持环境变量覆盖
- 支持配置验证（schema验证）
"""

import json
import os
import re
from configparser import ConfigParser
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

import yaml

from .json_utils import load_json


class ConfigLoadError(Exception):
    """配置加载错误"""
    pass


class ConfigValidationError(Exception):
    """配置验证错误"""
    pass


class ConfigLoader:
    """
    通用配置加载器

    支持多种格式的配置文件加载和验证
    """

    def __init__(self):
        """初始化加载器"""
        self._loaded_config: Optional[Dict[str, Any]] = None
        self._file_path: Optional[str] = None

    def load(self, path: str, format: str = "auto") -> Dict[str, Any]:
        """
        加载配置文件

        Args:
            path: 配置文件路径
            format: 文件格式 ("auto", "json", "yaml", "ini")

        Returns:
            Dict[str, Any]: 配置字典

        Raises:
            ConfigLoadError: 加载失败时抛出
        """
        if not os.path.exists(path):
            raise ConfigLoadError(f"配置文件不存在: {path}")

        # 自动检测格式
        if format == "auto":
            format = self._detect_format(path)

        try:
            if format == "json":
                config = self._load_json(path)
            elif format == "yaml":
                config = self._load_yaml(path)
            elif format == "ini":
                config = self._load_ini(path)
            else:
                raise ConfigLoadError(f"不支持的配置格式: {format}")
        except ConfigLoadError:
            raise
        except Exception as e:
            raise ConfigLoadError(f"加载配置文件失败: {e}")

        self._loaded_config = config
        self._file_path = path

        return config

    def _detect_format(self, path: str) -> str:
        """
        根据文件扩展名检测格式

        Args:
            path: 文件路径

        Returns:
            str: 检测到的格式
        """
        ext = Path(path).suffix.lower()
        format_map = {
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".ini": "ini",
            ".conf": "ini",
            ".cfg": "ini"
        }
        if ext in format_map:
            return format_map[ext]
        raise ConfigLoadError(f"无法自动检测文件格式: {ext}")

    def _load_json(self, path: str) -> Dict[str, Any]:
        """加载JSON文件"""
        try:
            return load_json(path)
        except FileNotFoundError as e:
            raise ConfigLoadError(f"配置文件不存在: {e}")
        except json.JSONDecodeError as e:
            raise ConfigLoadError(f"JSON解析错误: {e}")

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """加载YAML文件"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigLoadError(f"YAML解析错误: {e}")

    def _load_ini(self, path: str) -> Dict[str, Any]:
        """加载INI文件"""
        try:
            parser = ConfigParser()
            parser.read(path, encoding='utf-8')
            result = {}
            for section in parser.sections():
                result[section] = dict(parser.items(section))
            return result
        except Exception as e:
            raise ConfigLoadError(f"INI解析错误: {e}")

    def load_from_env(self, prefix: str) -> Dict[str, Any]:
        """
        从环境变量加载配置

        Args:
            prefix: 环境变量前缀

        Returns:
            Dict[str, Any]: 配置字典
        """
        result = {}
        prefix_lower = prefix.lower()

        for key, value in os.environ.items():
            key_lower = key.lower()
            if key_lower.startswith(prefix_lower):
                # 移除前缀并转换为小写
                config_key = key_lower[len(prefix_lower):]
                result[config_key] = value

        return result

    def validate(self, config: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证配置

        Args:
            config: 配置字典
            schema: 验证schema

        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误列表)
        """
        errors = []

        if not schema:
            return True, errors

        # 验证类型
        if "type" in schema:
            type_valid, type_errors = self._validate_type(config, schema["type"], "root")
            if not type_valid:
                errors.extend(type_errors)

        # 验证必需字段
        if "required" in schema and isinstance(config, dict):
            for field in schema["required"]:
                if field not in config:
                    errors.append(f"缺少必需字段: {field}")

        # 验证属性
        if "properties" in schema and isinstance(config, dict):
            for prop, prop_schema in schema["properties"].items():
                if prop in config:
                    prop_valid, prop_errors = self._validate_property(
                        config[prop], prop_schema, prop
                    )
                    if not prop_valid:
                        errors.extend(prop_errors)

        return len(errors) == 0, errors

    def _validate_type(self, value: Any, expected_type: str, path: str) -> Tuple[bool, List[str]]:
        """验证类型"""
        errors = []

        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }

        if expected_type not in type_map:
            return True, errors

        expected = type_map[expected_type]
        if not isinstance(value, expected):
            actual_type = type(value).__name__
            errors.append(f"字段 '{path}' 类型错误: 期望 {expected_type}, 实际 {actual_type}")
            return False, errors

        return True, errors

    def _validate_property(self, value: Any, schema: Dict[str, Any], path: str) -> Tuple[bool, List[str]]:
        """验证属性"""
        errors = []

        if "type" in schema:
            type_valid, type_errors = self._validate_type(value, schema["type"], path)
            if not type_valid:
                errors.extend(type_errors)

        # 递归验证嵌套对象
        if isinstance(value, dict) and "properties" in schema:
            for prop, prop_schema in schema["properties"].items():
                if prop in value:
                    prop_valid, prop_errors = self._validate_property(
                        value[prop], prop_schema, f"{path}.{prop}"
                    )
                    if not prop_valid:
                        errors.extend(prop_errors)

        # 验证数组项
        if isinstance(value, list) and "items" in schema:
            for i, item in enumerate(value):
                item_valid, item_errors = self._validate_property(
                    item, schema["items"], f"{path}[{i}]"
                )
                if not item_valid:
                    errors.extend(item_errors)

        return len(errors) == 0, errors

    def get_loaded_config(self) -> Optional[Dict[str, Any]]:
        """获取最后加载的配置"""
        return self._loaded_config

    def get_file_path(self) -> Optional[str]:
        """获取最后加载的文件路径"""
        return self._file_path
