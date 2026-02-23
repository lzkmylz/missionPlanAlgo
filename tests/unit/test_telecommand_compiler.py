"""
指令编译器测试

测试第19章设计的TelecommandCompiler
遵循TDD原则：先写测试，再实现代码
"""

import pytest
import json
import tempfile
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
from xml.etree import ElementTree as ET


class TestTelecommandCompilerInitialization:
    """测试TelecommandCompiler初始化"""

    def test_compiler_creation(self):
        """测试编译器基本创建"""
        from core.telecommand.telecommand_compiler import TelecommandCompiler

        compiler = TelecommandCompiler()

        assert compiler is not None
        assert hasattr(compiler, 'version')
        assert hasattr(compiler, 'supported_formats')

    def test_compiler_with_config(self):
        """测试带配置的编译器创建"""
        from core.telecommand.telecommand_compiler import TelecommandCompiler

        config = {
            'version': '2.0',
            'mission_id': 'TEST_MISSION',
            'default_output_format': 'json'
        }

        compiler = TelecommandCompiler(config=config)

        assert compiler.version == '2.0'
        assert compiler.mission_id == 'TEST_MISSION'


class TestCompileSOE:
    """测试SOE编译功能"""

    @pytest.fixture
    def sample_soe(self):
        """创建测试用的SOE数据"""
        return {
            'satellite_id': 'sat_001',
            'execution_time': datetime(2024, 1, 1, 12, 0, 0),
            'entries': [
                {
                    'id': 'entry_001',
                    'action_type': 'IMAGING',
                    'start_time': datetime(2024, 1, 1, 12, 0, 0),
                    'end_time': datetime(2024, 1, 1, 12, 5, 0),
                    'target_id': 'target_001',
                    'parameters': {
                        'imaging_mode': 'stripmap',
                        'off_nadir_angle': 15.0
                    }
                },
                {
                    'id': 'entry_002',
                    'action_type': 'SLEW',
                    'start_time': datetime(2024, 1, 1, 12, 5, 0),
                    'end_time': datetime(2024, 1, 1, 12, 6, 0),
                    'parameters': {
                        'slew_angle': 30.0,
                        'target_angle': 45.0
                    }
                },
                {
                    'id': 'entry_003',
                    'action_type': 'DOWNLINK',
                    'start_time': datetime(2024, 1, 1, 12, 30, 0),
                    'end_time': datetime(2024, 1, 1, 12, 35, 0),
                    'ground_station_id': 'gs_001',
                    'parameters': {
                        'data_volume_mb': 500.0
                    }
                }
            ]
        }

    @pytest.fixture
    def compiler(self):
        """创建测试用的编译器"""
        from core.telecommand.telecommand_compiler import TelecommandCompiler

        return TelecommandCompiler()

    def test_compile_soe_basic(self, compiler, sample_soe):
        """测试基本SOE编译"""
        result = compiler.compile_soe(sample_soe)

        assert result is not None
        assert isinstance(result, dict)
        assert 'commands' in result
        assert len(result['commands']) == len(sample_soe['entries'])

    def test_compile_soe_structure(self, compiler, sample_soe):
        """测试编译后的SOE结构"""
        result = compiler.compile_soe(sample_soe)

        for cmd in result['commands']:
            assert 'command_id' in cmd
            assert 'execution_time' in cmd
            assert 'command_type' in cmd
            assert 'parameters' in cmd
            assert 'checksum' in cmd

    def test_compile_imaging_command(self, compiler, sample_soe):
        """测试成像指令编译"""
        result = compiler.compile_soe(sample_soe)

        imaging_cmd = result['commands'][0]
        assert imaging_cmd['command_type'] == 'IMAGING'
        assert 'target_id' in imaging_cmd['parameters']
        assert 'imaging_mode' in imaging_cmd['parameters']

    def test_compile_slew_command(self, compiler, sample_soe):
        """测试机动指令编译"""
        result = compiler.compile_soe(sample_soe)

        slew_cmd = result['commands'][1]
        assert slew_cmd['command_type'] == 'SLEW'
        assert 'slew_angle' in slew_cmd['parameters']

    def test_compile_downlink_command(self, compiler, sample_soe):
        """测试数传指令编译"""
        result = compiler.compile_soe(sample_soe)

        downlink_cmd = result['commands'][2]
        assert downlink_cmd['command_type'] == 'DOWNLINK'
        assert 'ground_station_id' in downlink_cmd['parameters']

    def test_compile_with_metadata(self, compiler, sample_soe):
        """测试带元数据的编译"""
        metadata = {
            'mission_name': 'Test Mission',
            'operator': 'Test Operator',
            'generation_time': datetime(2024, 1, 1, 10, 0, 0)
        }

        result = compiler.compile_soe(sample_soe, metadata=metadata)

        assert 'metadata' in result
        assert result['metadata']['mission_name'] == 'Test Mission'
        assert result['metadata']['operator'] == 'Test Operator'

    def test_compile_empty_soe(self, compiler):
        """测试空SOE编译"""
        empty_soe = {
            'satellite_id': 'sat_001',
            'execution_time': datetime(2024, 1, 1, 12, 0, 0),
            'entries': []
        }

        result = compiler.compile_soe(empty_soe)

        assert result is not None
        assert len(result['commands']) == 0


class TestExportToJSON:
    """测试JSON格式导出"""

    @pytest.fixture
    def compiled_commands(self):
        """创建测试用的编译后指令"""
        return {
            'metadata': {
                'mission_name': 'Test Mission',
                'version': '1.0',
                'generation_time': '2024-01-01T10:00:00'
            },
            'commands': [
                {
                    'command_id': 'cmd_001',
                    'execution_time': '2024-01-01T12:00:00',
                    'command_type': 'IMAGING',
                    'parameters': {'target_id': 'target_001'},
                    'checksum': 'abc123'
                }
            ]
        }

    @pytest.fixture
    def compiler(self):
        """创建测试用的编译器"""
        from core.telecommand.telecommand_compiler import TelecommandCompiler

        return TelecommandCompiler()

    def test_export_to_json_file(self, compiler, compiled_commands):
        """测试导出到JSON文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            compiler.export_to_file(compiled_commands, temp_path, format='json')

            assert os.path.exists(temp_path)

            with open(temp_path, 'r') as f:
                loaded = json.load(f)

            assert loaded['metadata']['mission_name'] == 'Test Mission'
            assert len(loaded['commands']) == 1
        finally:
            os.unlink(temp_path)

    def test_export_to_json_string(self, compiler, compiled_commands):
        """测试导出到JSON字符串"""
        json_str = compiler.export_to_string(compiled_commands, format='json')

        assert isinstance(json_str, str)

        loaded = json.loads(json_str)
        assert loaded['metadata']['mission_name'] == 'Test Mission'

    def test_json_format_validation(self, compiler, compiled_commands):
        """测试JSON格式验证"""
        json_str = compiler.export_to_string(compiled_commands, format='json')

        # 验证是有效的JSON
        parsed = json.loads(json_str)
        assert 'commands' in parsed
        assert 'metadata' in parsed


class TestExportToXML:
    """测试XML格式导出"""

    @pytest.fixture
    def compiled_commands(self):
        """创建测试用的编译后指令"""
        return {
            'metadata': {
                'mission_name': 'Test Mission',
                'version': '1.0',
                'generation_time': '2024-01-01T10:00:00'
            },
            'commands': [
                {
                    'command_id': 'cmd_001',
                    'execution_time': '2024-01-01T12:00:00',
                    'command_type': 'IMAGING',
                    'parameters': {'target_id': 'target_001', 'mode': 'stripmap'},
                    'checksum': 'abc123'
                },
                {
                    'command_id': 'cmd_002',
                    'execution_time': '2024-01-01T12:05:00',
                    'command_type': 'SLEW',
                    'parameters': {'angle': 30.0},
                    'checksum': 'def456'
                }
            ]
        }

    @pytest.fixture
    def compiler(self):
        """创建测试用的编译器"""
        from core.telecommand.telecommand_compiler import TelecommandCompiler

        return TelecommandCompiler()

    def test_export_to_xml_file(self, compiler, compiled_commands):
        """测试导出到XML文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            temp_path = f.name

        try:
            compiler.export_to_file(compiled_commands, temp_path, format='xml')

            assert os.path.exists(temp_path)

            # 解析XML验证结构
            tree = ET.parse(temp_path)
            root = tree.getroot()

            assert root.tag == 'telecommand_sequence'
            assert root.find('metadata') is not None
            commands_elem = root.find('commands')
            assert commands_elem is not None
            assert len(commands_elem.findall('command')) == 2
        finally:
            os.unlink(temp_path)

    def test_export_to_xml_string(self, compiler, compiled_commands):
        """测试导出到XML字符串"""
        xml_str = compiler.export_to_string(compiled_commands, format='xml')

        assert isinstance(xml_str, str)
        assert '<telecommand_sequence>' in xml_str
        assert '<command' in xml_str
        assert '</telecommand_sequence>' in xml_str

    def test_xml_command_structure(self, compiler, compiled_commands):
        """测试XML指令结构"""
        xml_str = compiler.export_to_string(compiled_commands, format='xml')

        root = ET.fromstring(xml_str)

        commands_elem = root.find('commands')
        assert commands_elem is not None
        commands = commands_elem.findall('command')
        assert len(commands) == 2

        cmd = commands[0]
        assert cmd.get('id') == 'cmd_001'
        assert cmd.find('type').text == 'IMAGING'
        assert cmd.find('execution_time').text == '2024-01-01T12:00:00'

    def test_xml_metadata_structure(self, compiler, compiled_commands):
        """测试XML元数据结构"""
        xml_str = compiler.export_to_string(compiled_commands, format='xml')

        root = ET.fromstring(xml_str)

        metadata = root.find('metadata')
        assert metadata is not None
        assert metadata.find('mission_name').text == 'Test Mission'
        assert metadata.find('version').text == '1.0'


class TestChecksumGeneration:
    """测试校验和生成"""

    def test_checksum_generation(self):
        """测试校验和生成"""
        from core.telecommand.telecommand_compiler import TelecommandCompiler

        compiler = TelecommandCompiler()

        command = {
            'command_id': 'cmd_001',
            'command_type': 'IMAGING',
            'parameters': {'target_id': 'target_001'}
        }

        checksum = compiler._generate_checksum(command)

        assert isinstance(checksum, str)
        assert len(checksum) > 0

    def test_checksum_consistency(self):
        """测试校验和一致性"""
        from core.telecommand.telecommand_compiler import TelecommandCompiler

        compiler = TelecommandCompiler()

        command = {
            'command_id': 'cmd_001',
            'command_type': 'IMAGING',
            'parameters': {'target_id': 'target_001'}
        }

        checksum1 = compiler._generate_checksum(command)
        checksum2 = compiler._generate_checksum(command)

        assert checksum1 == checksum2

    def test_checksum_uniqueness(self):
        """测试校验和唯一性"""
        from core.telecommand.telecommand_compiler import TelecommandCompiler

        compiler = TelecommandCompiler()

        command1 = {'command_id': 'cmd_001', 'command_type': 'IMAGING'}
        command2 = {'command_id': 'cmd_002', 'command_type': 'SLEW'}

        checksum1 = compiler._generate_checksum(command1)
        checksum2 = compiler._generate_checksum(command2)

        assert checksum1 != checksum2


class TestCommandValidation:
    """测试指令验证"""

    @pytest.fixture
    def compiler(self):
        """创建测试用的编译器"""
        from core.telecommand.telecommand_compiler import TelecommandCompiler

        return TelecommandCompiler()

    def test_valid_command(self, compiler):
        """测试有效指令"""
        command = {
            'command_id': 'cmd_001',
            'command_type': 'IMAGING',
            'execution_time': datetime(2024, 1, 1, 12, 0, 0),
            'parameters': {'target_id': 'target_001'}
        }

        is_valid, errors = compiler.validate_command(command)

        assert is_valid is True
        assert len(errors) == 0

    def test_missing_command_id(self, compiler):
        """测试缺少指令ID"""
        command = {
            'command_type': 'IMAGING',
            'execution_time': datetime(2024, 1, 1, 12, 0, 0),
            'parameters': {}
        }

        is_valid, errors = compiler.validate_command(command)

        assert is_valid is False
        assert any('id' in error.lower() for error in errors)

    def test_missing_command_type(self, compiler):
        """测试缺少指令类型"""
        command = {
            'command_id': 'cmd_001',
            'execution_time': datetime(2024, 1, 1, 12, 0, 0),
            'parameters': {}
        }

        is_valid, errors = compiler.validate_command(command)

        assert is_valid is False
        assert any('type' in error.lower() for error in errors)

    def test_invalid_command_type(self, compiler):
        """测试无效指令类型"""
        command = {
            'command_id': 'cmd_001',
            'command_type': 'INVALID_TYPE',
            'execution_time': datetime(2024, 1, 1, 12, 0, 0),
            'parameters': {}
        }

        is_valid, errors = compiler.validate_command(command)

        assert is_valid is False
        assert any('invalid' in error.lower() for error in errors)

    def test_missing_execution_time(self, compiler):
        """测试缺少执行时间"""
        command = {
            'command_id': 'cmd_001',
            'command_type': 'IMAGING',
            'parameters': {}
        }

        is_valid, errors = compiler.validate_command(command)

        assert is_valid is False
        assert any('time' in error.lower() for error in errors)


class TestSupportedFormats:
    """测试支持的格式"""

    def test_default_formats(self):
        """测试默认支持格式"""
        from core.telecommand.telecommand_compiler import TelecommandCompiler

        compiler = TelecommandCompiler()

        assert 'json' in compiler.supported_formats
        assert 'xml' in compiler.supported_formats

    def test_is_format_supported(self):
        """测试格式支持检查"""
        from core.telecommand.telecommand_compiler import TelecommandCompiler

        compiler = TelecommandCompiler()

        assert compiler.is_format_supported('json') is True
        assert compiler.is_format_supported('xml') is True
        assert compiler.is_format_supported('yaml') is False
        assert compiler.is_format_supported('binary') is False


class TestEdgeCases:
    """测试边界情况"""

    @pytest.fixture
    def compiler(self):
        """创建测试用的编译器"""
        from core.telecommand.telecommand_compiler import TelecommandCompiler

        return TelecommandCompiler()

    def test_empty_command_list(self, compiler):
        """测试空指令列表"""
        soe = {
            'satellite_id': 'sat_001',
            'execution_time': datetime(2024, 1, 1, 12, 0, 0),
            'entries': []
        }

        result = compiler.compile_soe(soe)

        assert result is not None
        assert len(result['commands']) == 0

    def test_very_long_command_list(self, compiler):
        """测试超长指令列表"""
        entries = [
            {
                'id': f'entry_{i:04d}',
                'action_type': 'IMAGING',
                'start_time': datetime(2024, 1, 1, 12, 0, 0) + timedelta(minutes=i),
                'end_time': datetime(2024, 1, 1, 12, 0, 0) + timedelta(minutes=i+1),
                'target_id': f'target_{i:04d}',
                'parameters': {}
            }
            for i in range(1000)
        ]

        soe = {
            'satellite_id': 'sat_001',
            'execution_time': datetime(2024, 1, 1, 12, 0, 0),
            'entries': entries
        }

        result = compiler.compile_soe(soe)

        assert len(result['commands']) == 1000

    def test_special_characters_in_parameters(self, compiler):
        """测试特殊字符参数"""
        soe = {
            'satellite_id': 'sat_001',
            'execution_time': datetime(2024, 1, 1, 12, 0, 0),
            'entries': [
                {
                    'id': 'entry_001',
                    'action_type': 'IMAGING',
                    'start_time': datetime(2024, 1, 1, 12, 0, 0),
                    'end_time': datetime(2024, 1, 1, 12, 5, 0),
                    'target_id': 'target_001',
                    'parameters': {
                        'description': 'Test with special chars: <>&"\'',
                        'unicode': '中文测试'
                    }
                }
            ]
        }

        result = compiler.compile_soe(soe)

        # JSON应正确处理特殊字符
        json_str = compiler.export_to_string(result, format='json')
        loaded = json.loads(json_str)
        assert loaded['commands'][0]['parameters']['unicode'] == '中文测试'

    def test_nested_parameters(self, compiler):
        """测试嵌套参数"""
        soe = {
            'satellite_id': 'sat_001',
            'execution_time': datetime(2024, 1, 1, 12, 0, 0),
            'entries': [
                {
                    'id': 'entry_001',
                    'action_type': 'IMAGING',
                    'start_time': datetime(2024, 1, 1, 12, 0, 0),
                    'end_time': datetime(2024, 1, 1, 12, 5, 0),
                    'target_id': 'target_001',
                    'parameters': {
                        'camera_settings': {
                            'exposure': 0.01,
                            'gain': 2.0,
                            'mode': 'high_resolution'
                        },
                        'attitude': {
                            'roll': 0.0,
                            'pitch': 15.0,
                            'yaw': 0.0
                        }
                    }
                }
            ]
        }

        result = compiler.compile_soe(soe)

        params = result['commands'][0]['parameters']
        assert 'camera_settings' in params
        assert params['camera_settings']['exposure'] == 0.01

    def test_export_unsupported_format(self, compiler):
        """测试导出到不支持的格式"""
        compiled_commands = {
            'metadata': {},
            'commands': []
        }

        with pytest.raises(ValueError):
            compiler.export_to_string(compiled_commands, format='unsupported')
