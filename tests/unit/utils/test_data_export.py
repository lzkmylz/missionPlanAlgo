"""
DataExporter模块的单元测试

TDD流程：
1. 先写测试（RED）
2. 运行测试 - 验证失败
3. 实现代码（GREEN）
4. 运行测试 - 验证通过
5. 重构（IMPROVE）
"""

import csv
import json
import os
import tempfile
from unittest import TestCase

import pytest


class TestDataExporter(TestCase):
    """DataExporter测试类"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = [
            {"id": 1, "name": "Alice", "age": 30, "city": "Beijing"},
            {"id": 2, "name": "Bob", "age": 25, "city": "Shanghai"},
            {"id": 3, "name": "Charlie", "age": 35, "city": "Guangzhou"}
        ]

    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _get_temp_path(self, filename: str) -> str:
        """获取临时文件路径"""
        return os.path.join(self.temp_dir, filename)

    # ==================== CSV导出测试 ====================

    def test_to_csv_basic(self):
        """测试基本CSV导出"""
        from utils.data_export import DataExporter

        output_path = self._get_temp_path("output.csv")
        exporter = DataExporter()
        exporter.to_csv(self.sample_data, output_path)

        # 验证文件存在
        self.assertTrue(os.path.exists(output_path))

        # 验证内容
        with open(output_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["name"], "Alice")
            self.assertEqual(rows[1]["age"], "25")

    def test_to_csv_empty_data(self):
        """测试导出空数据到CSV"""
        from utils.data_export import DataExporter, DataExportError

        output_path = self._get_temp_path("output.csv")
        exporter = DataExporter()

        with self.assertRaises(DataExportError):
            exporter.to_csv([], output_path)

    def test_to_csv_missing_fields(self):
        """测试处理缺失字段的数据"""
        from utils.data_export import DataExporter

        data = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob", "extra": "field"}
        ]
        output_path = self._get_temp_path("output.csv")
        exporter = DataExporter()
        exporter.to_csv(data, output_path)

        with open(output_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(rows[0].get("extra"), "")
            self.assertEqual(rows[1]["extra"], "field")

    def test_to_csv_unicode_content(self):
        """测试导出Unicode内容到CSV"""
        from utils.data_export import DataExporter

        data = [
            {"id": 1, "name": "张三", "city": "北京"},
            {"id": 2, "name": "🚀", "city": "上海"}
        ]
        output_path = self._get_temp_path("output.csv")
        exporter = DataExporter()
        exporter.to_csv(data, output_path)

        with open(output_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(rows[0]["name"], "张三")
            self.assertEqual(rows[1]["name"], "🚀")

    def test_to_csv_special_characters(self):
        """测试导出包含特殊字符的数据"""
        from utils.data_export import DataExporter

        data = [
            {"id": 1, "description": "Line1\nLine2", "value": "a,b,c"},
            {"id": 2, "description": "Quote\"Test", "value": "normal"}
        ]
        output_path = self._get_temp_path("output.csv")
        exporter = DataExporter()
        exporter.to_csv(data, output_path)

        with open(output_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 2)

    # ==================== JSON导出测试 ====================

    def test_to_json_basic(self):
        """测试基本JSON导出"""
        from utils.data_export import DataExporter

        output_path = self._get_temp_path("output.json")
        exporter = DataExporter()
        exporter.to_json(self.sample_data, output_path)

        # 验证文件存在
        self.assertTrue(os.path.exists(output_path))

        # 验证内容
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.assertEqual(len(data), 3)
            self.assertEqual(data[0]["name"], "Alice")

    def test_to_json_empty_data(self):
        """测试导出空数据到JSON"""
        from utils.data_export import DataExporter, DataExportError

        output_path = self._get_temp_path("output.json")
        exporter = DataExporter()

        with self.assertRaises(DataExportError):
            exporter.to_json([], output_path)

    def test_to_json_pretty_print(self):
        """测试JSON美化输出"""
        from utils.data_export import DataExporter

        output_path = self._get_temp_path("output.json")
        exporter = DataExporter()
        exporter.to_json(self.sample_data, output_path, indent=2)

        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("\n", content)  # 应该有换行
            self.assertIn("  ", content)  # 应该有缩进

    def test_to_json_unicode(self):
        """测试导出Unicode到JSON"""
        from utils.data_export import DataExporter

        data = [{"name": "中文测试", "emoji": "🚀"}]
        output_path = self._get_temp_path("output.json")
        exporter = DataExporter()
        exporter.to_json(data, output_path)

        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("中文测试", content)  # 不应该被转义
            self.assertIn("🚀", content)

    def test_to_json_nested_data(self):
        """测试导出嵌套数据到JSON"""
        from utils.data_export import DataExporter

        data = [
            {
                "id": 1,
                "nested": {
                    "level1": {
                        "level2": "deep value"
                    }
                },
                "list": [1, 2, 3]
            }
        ]
        output_path = self._get_temp_path("output.json")
        exporter = DataExporter()
        exporter.to_json(data, output_path)

        with open(output_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
            self.assertEqual(result[0]["nested"]["level1"]["level2"], "deep value")

    # ==================== Excel导出测试 ====================

    def test_to_excel_basic(self):
        """测试基本Excel导出"""
        from utils.data_export import DataExporter

        output_path = self._get_temp_path("output.xlsx")
        exporter = DataExporter()
        exporter.to_excel(self.sample_data, output_path, sheet_name="Sheet1")

        # 验证文件存在
        self.assertTrue(os.path.exists(output_path))

    def test_to_excel_empty_data(self):
        """测试导出空数据到Excel"""
        from utils.data_export import DataExporter, DataExportError

        output_path = self._get_temp_path("output.xlsx")
        exporter = DataExporter()

        with self.assertRaises(DataExportError):
            exporter.to_excel([], output_path, sheet_name="Sheet1")

    def test_to_excel_default_sheet_name(self):
        """测试默认sheet名称"""
        from utils.data_export import DataExporter

        output_path = self._get_temp_path("output.xlsx")
        exporter = DataExporter()
        exporter.to_excel(self.sample_data, output_path)

        self.assertTrue(os.path.exists(output_path))

    def test_to_excel_unicode(self):
        """测试导出Unicode到Excel"""
        from utils.data_export import DataExporter

        data = [
            {"id": 1, "name": "张三", "description": "🚀火箭"},
            {"id": 2, "name": "李四", "description": "🛰️卫星"}
        ]
        output_path = self._get_temp_path("output.xlsx")
        exporter = DataExporter()
        exporter.to_excel(data, output_path, sheet_name="数据")

        self.assertTrue(os.path.exists(output_path))

    # ==================== 数据格式化测试 ====================

    def test_format_data_date(self):
        """测试日期格式化"""
        from utils.data_export import DataExporter
        from datetime import datetime

        data = [
            {"id": 1, "created_at": datetime(2024, 1, 15, 10, 30, 0)}
        ]
        output_path = self._get_temp_path("output.json")
        exporter = DataExporter()
        exporter.to_json(data, output_path)

        with open(output_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
            self.assertIn("2024-01-15", result[0]["created_at"])

    def test_format_data_number(self):
        """测试数字格式化"""
        from utils.data_export import DataExporter

        data = [
            {"id": 1, "value": 1234.56789, "int_val": 42}
        ]
        output_path = self._get_temp_path("output.csv")
        exporter = DataExporter()
        exporter.to_csv(data, output_path)

        with open(output_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(rows[0]["int_val"], "42")

    # ==================== 错误处理测试 ====================

    def test_export_to_invalid_path(self):
        """测试导出到无效路径"""
        from utils.data_export import DataExporter, DataExportError

        exporter = DataExporter()

        with self.assertRaises(DataExportError):
            exporter.to_csv(self.sample_data, "/nonexistent/directory/file.csv")

    def test_export_none_data(self):
        """测试导出None数据"""
        from utils.data_export import DataExporter, DataExportError

        output_path = self._get_temp_path("output.csv")
        exporter = DataExporter()

        with self.assertRaises(DataExportError):
            exporter.to_csv(None, output_path)

    def test_export_invalid_data_type(self):
        """测试导出无效数据类型"""
        from utils.data_export import DataExporter, DataExportError

        output_path = self._get_temp_path("output.csv")
        exporter = DataExporter()

        with self.assertRaises(DataExportError):
            exporter.to_csv("not a list", output_path)

    # ==================== 边缘情况测试 ====================

    def test_export_single_row(self):
        """测试导出单行数据"""
        from utils.data_export import DataExporter

        data = [{"id": 1, "name": "Only"}]
        output_path = self._get_temp_path("output.csv")
        exporter = DataExporter()
        exporter.to_csv(data, output_path)

        with open(output_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 1)

    def test_export_many_columns(self):
        """测试导出多列数据"""
        from utils.data_export import DataExporter

        data = [{f"col_{i}": f"value_{i}" for i in range(50)}]
        output_path = self._get_temp_path("output.csv")
        exporter = DataExporter()
        exporter.to_csv(data, output_path)

        with open(output_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows[0]), 50)

    def test_csv_no_valid_fields(self):
        """测试CSV导出无有效字段"""
        from utils.data_export import DataExporter, DataExportError

        data = ["not a dict", "also not a dict"]
        output_path = self._get_temp_path("output.csv")
        exporter = DataExporter()

        with self.assertRaises(DataExportError) as context:
            exporter.to_csv(data, output_path)
        self.assertIn("没有有效的字段", str(context.exception))

    def test_csv_with_nested_list_value(self):
        """测试CSV导出嵌套列表值"""
        from utils.data_export import DataExporter

        data = [{"id": 1, "items": [1, 2, 3], "nested": {"a": 1}}]
        output_path = self._get_temp_path("output.csv")
        exporter = DataExporter()
        exporter.to_csv(data, output_path)

        with open(output_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertIn('[1, 2, 3]', rows[0]["items"])
            self.assertIn('{"a": 1}', rows[0]["nested"])

    def test_csv_with_date_value(self):
        """测试CSV导出日期值"""
        from utils.data_export import DataExporter
        from datetime import datetime, date

        data = [{"id": 1, "created": datetime(2024, 1, 15, 10, 30), "birth": date(1990, 5, 20)}]
        output_path = self._get_temp_path("output.csv")
        exporter = DataExporter()
        exporter.to_csv(data, output_path)

        with open(output_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertIn("2024-01-15", rows[0]["created"])
            self.assertIn("1990-05-20", rows[0]["birth"])

    def test_export_large_data(self):
        """测试导出大量数据"""
        from utils.data_export import DataExporter

        data = [{"id": i, "value": f"data_{i}"} for i in range(10000)]
        output_path = self._get_temp_path("output.csv")
        exporter = DataExporter()
        exporter.to_csv(data, output_path)

        with open(output_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 10000)

    def test_export_auto_format_csv(self):
        """测试自动格式检测CSV"""
        from utils.data_export import DataExporter

        output_path = self._get_temp_path("output.csv")
        exporter = DataExporter()
        exporter.export(self.sample_data, output_path)  # 自动检测

        self.assertTrue(os.path.exists(output_path))
        with open(output_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 3)

    def test_export_auto_format_json(self):
        """测试自动格式检测JSON"""
        from utils.data_export import DataExporter

        output_path = self._get_temp_path("output.json")
        exporter = DataExporter()
        exporter.export(self.sample_data, output_path)  # 自动检测

        self.assertTrue(os.path.exists(output_path))
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.assertEqual(len(data), 3)

    def test_export_auto_format_excel(self):
        """测试自动格式检测Excel"""
        from utils.data_export import DataExporter

        output_path = self._get_temp_path("output.xlsx")
        exporter = DataExporter()
        exporter.export(self.sample_data, output_path)  # 自动检测

        self.assertTrue(os.path.exists(output_path))

    def test_export_unsupported_format(self):
        """测试不支持的格式"""
        from utils.data_export import DataExporter, DataExportError

        output_path = self._get_temp_path("output.txt")
        exporter = DataExporter()

        with self.assertRaises(DataExportError) as context:
            exporter.export(self.sample_data, output_path)
        self.assertIn("无法自动检测", str(context.exception))

    def test_export_explicit_format(self):
        """测试显式指定格式"""
        from utils.data_export import DataExporter

        output_path = self._get_temp_path("data.txt")
        exporter = DataExporter()
        exporter.export(self.sample_data, output_path, format='csv')

        self.assertTrue(os.path.exists(output_path))

    def test_json_serializer_object(self):
        """测试JSON序列化对象"""
        from utils.data_export import DataExporter

        class TestObj:
            def __init__(self):
                self.name = "test"
                self.value = 123

        data = [{"id": 1, "obj": TestObj()}]
        output_path = self._get_temp_path("output.json")
        exporter = DataExporter()
        exporter.to_json(data, output_path)

        with open(output_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
            self.assertEqual(result[0]["obj"]["name"], "test")
            self.assertEqual(result[0]["obj"]["value"], 123)

    def test_excel_no_valid_fields(self):
        """测试Excel导出无有效字段"""
        from utils.data_export import DataExporter, DataExportError

        data = ["not a dict", "also not a dict"]
        output_path = self._get_temp_path("output.xlsx")
        exporter = DataExporter()

        with self.assertRaises(DataExportError) as context:
            exporter.to_excel(data, output_path)
        self.assertIn("没有有效的字段", str(context.exception))

    def test_excel_with_bool_value(self):
        """测试Excel导出布尔值"""
        from utils.data_export import DataExporter

        data = [{"id": 1, "active": True, "deleted": False}]
        output_path = self._get_temp_path("output.xlsx")
        exporter = DataExporter()
        exporter.to_excel(data, output_path)

        self.assertTrue(os.path.exists(output_path))

    def test_excel_with_numeric_value(self):
        """测试Excel导出数值"""
        from utils.data_export import DataExporter

        data = [{"id": 1, "count": 100, "price": 99.99}]
        output_path = self._get_temp_path("output.xlsx")
        exporter = DataExporter()
        exporter.to_excel(data, output_path)

        self.assertTrue(os.path.exists(output_path))


class TestDataExporterIntegration(TestCase):
    """DataExporter集成测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _get_temp_path(self, filename: str) -> str:
        return os.path.join(self.temp_dir, filename)

    def test_full_export_workflow(self):
        """测试完整导出工作流程"""
        from utils.data_export import DataExporter

        data = [
            {"id": 1, "name": "Task1", "status": "completed", "duration": 120},
            {"id": 2, "name": "Task2", "status": "pending", "duration": 60},
            {"id": 3, "name": "Task3", "status": "failed", "duration": 0}
        ]

        exporter = DataExporter()

        # 导出为CSV
        csv_path = self._get_temp_path("tasks.csv")
        exporter.to_csv(data, csv_path)

        # 导出为JSON
        json_path = self._get_temp_path("tasks.json")
        exporter.to_json(data, json_path)

        # 导出为Excel
        excel_path = self._get_temp_path("tasks.xlsx")
        exporter.to_excel(data, excel_path, sheet_name="Tasks")

        # 验证所有文件存在
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists(excel_path))

        # 验证JSON内容
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            self.assertEqual(len(json_data), 3)
            self.assertEqual(json_data[0]["name"], "Task1")


if __name__ == "__main__":
    import unittest
    unittest.main()
