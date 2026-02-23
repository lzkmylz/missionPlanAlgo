"""
指令编译器

实现第19章设计：
- TelecommandCompiler类
- compile_soe方法
- export_to_file方法（支持json/xml格式）
"""

import json
import hashlib
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from xml.etree import ElementTree as ET
from xml.dom import minidom


class CommandType(Enum):
    """指令类型枚举"""
    IMAGING = "IMAGING"
    SLEW = "SLEW"
    DOWNLINK = "DOWNLINK"
    IDLE = "IDLE"
    MAINTENANCE = "MAINTENANCE"


@dataclass
class CompiledCommand:
    """编译后的指令"""
    command_id: str
    command_type: str
    execution_time: str
    parameters: Dict[str, Any]
    checksum: str
    priority: int = 5
    status: str = "pending"


class TelecommandCompiler:
    """
    指令编译器

    将SOE（事件序列）编译为可执行的卫星指令
    支持导出为JSON和XML格式
    """

    # 支持的导出格式
    SUPPORTED_FORMATS = ['json', 'xml']

    # 有效的指令类型
    VALID_COMMAND_TYPES = [t.value for t in CommandType]

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化指令编译器

        Args:
            config: 配置参数
                - version: 编译器版本
                - mission_id: 任务ID
                - default_output_format: 默认输出格式
        """
        self.config = config or {}
        self.version = self.config.get('version', '1.0')
        self.mission_id = self.config.get('mission_id', 'DEFAULT_MISSION')
        self.default_format = self.config.get('default_output_format', 'json')

    def compile_soe(
        self,
        soe: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        编译SOE为可执行指令序列

        Args:
            soe: 事件序列
                - satellite_id: 卫星ID
                - execution_time: 执行时间
                - entries: 事件条目列表
            metadata: 可选的元数据

        Returns:
            编译后的指令序列
        """
        compiled_commands = []

        for entry in soe.get('entries', []):
            command = self._compile_entry(entry)
            if command:
                compiled_commands.append(command)

        result = {
            'metadata': self._build_metadata(soe, metadata),
            'commands': compiled_commands,
            'satellite_id': soe.get('satellite_id'),
            'mission_id': self.mission_id
        }

        return result

    def export_to_file(
        self,
        compiled_commands: Dict[str, Any],
        filepath: str,
        format: Optional[str] = None
    ) -> None:
        """
        导出编译后的指令到文件

        Args:
            compiled_commands: 编译后的指令
            filepath: 文件路径
            format: 导出格式 ('json' 或 'xml')
        """
        fmt = format or self.default_format

        if fmt not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {fmt}. Supported: {self.SUPPORTED_FORMATS}")

        content = self.export_to_string(compiled_commands, fmt)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    def export_to_string(
        self,
        compiled_commands: Dict[str, Any],
        format: str
    ) -> str:
        """
        导出编译后的指令为字符串

        Args:
            compiled_commands: 编译后的指令
            format: 导出格式 ('json' 或 'xml')

        Returns:
            格式化字符串
        """
        if format == 'json':
            return self._export_to_json(compiled_commands)
        elif format == 'xml':
            return self._export_to_xml(compiled_commands)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def validate_command(self, command: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证指令有效性

        Args:
            command: 指令字典

        Returns:
            (是否有效, 错误列表)
        """
        errors = []

        # 检查必需字段
        if 'command_id' not in command:
            errors.append("Missing required field: command_id")

        if 'command_type' not in command:
            errors.append("Missing required field: command_type")
        elif command['command_type'] not in self.VALID_COMMAND_TYPES:
            errors.append(f"Invalid command type: {command['command_type']}")

        if 'execution_time' not in command:
            errors.append("Missing required field: execution_time")

        return len(errors) == 0, errors

    def is_format_supported(self, format: str) -> bool:
        """检查格式是否支持"""
        return format in self.SUPPORTED_FORMATS

    @property
    def supported_formats(self) -> List[str]:
        """获取支持的格式列表"""
        return self.SUPPORTED_FORMATS.copy()

    def _compile_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """编译单个SOE条目"""
        action_type = entry.get('action_type', 'IDLE')

        # 映射动作类型到指令类型
        command_type = self._map_action_type(action_type)

        # 构建参数
        parameters = self._build_parameters(entry)

        # 构建指令
        command = {
            'command_id': entry.get('id', self._generate_command_id()),
            'command_type': command_type,
            'execution_time': self._format_datetime(entry.get('start_time')),
            'end_time': self._format_datetime(entry.get('end_time')),
            'parameters': parameters,
            'checksum': ''  # 稍后计算
        }

        # 计算校验和
        command['checksum'] = self._generate_checksum(command)

        return command

    def _map_action_type(self, action_type: str) -> str:
        """映射动作类型到指令类型"""
        mapping = {
            'IMAGING': 'IMAGING',
            'SLEW': 'SLEW',
            'DOWNLINK': 'DOWNLINK',
            'IDLE': 'IDLE',
            'MAINTENANCE': 'MAINTENANCE'
        }
        return mapping.get(action_type.upper(), 'IDLE')

    def _build_parameters(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """构建指令参数"""
        params = entry.get('parameters', {}).copy()

        # 添加通用参数
        if 'target_id' in entry:
            params['target_id'] = entry['target_id']
        if 'ground_station_id' in entry:
            params['ground_station_id'] = entry['ground_station_id']

        return params

    def _build_metadata(
        self,
        soe: Dict[str, Any],
        extra_metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """构建元数据"""
        metadata = {
            'version': self.version,
            'mission_id': self.mission_id,
            'satellite_id': soe.get('satellite_id'),
            'generation_time': datetime.now().isoformat(),
            'command_count': len(soe.get('entries', []))
        }

        if extra_metadata:
            metadata.update(extra_metadata)

        return metadata

    def _generate_checksum(self, command: Dict[str, Any]) -> str:
        """生成指令校验和"""
        # 创建可序列化的副本（排除校验和字段本身）
        data = {k: v for k, v in command.items() if k != 'checksum'}

        # 序列化为JSON字符串
        json_str = json.dumps(data, sort_keys=True, default=str)

        # 计算MD5校验和
        return hashlib.md5(json_str.encode('utf-8')).hexdigest()[:8]

    def _generate_command_id(self) -> str:
        """生成唯一指令ID"""
        import uuid
        return f"cmd_{uuid.uuid4().hex[:8]}"

    def _format_datetime(self, dt: Any) -> Optional[str]:
        """格式化日期时间"""
        if dt is None:
            return None
        if isinstance(dt, datetime):
            return dt.isoformat()
        return str(dt)

    def _export_to_json(self, compiled_commands: Dict[str, Any]) -> str:
        """导出为JSON格式"""
        return json.dumps(compiled_commands, indent=2, ensure_ascii=False, default=str)

    def _export_to_xml(self, compiled_commands: Dict[str, Any]) -> str:
        """导出为XML格式"""
        root = ET.Element('telecommand_sequence')

        # 添加元数据
        metadata_elem = ET.SubElement(root, 'metadata')
        metadata = compiled_commands.get('metadata', {})
        for key, value in metadata.items():
            child = ET.SubElement(metadata_elem, key)
            child.text = str(value) if value is not None else ''

        # 添加任务和卫星ID
        if compiled_commands.get('mission_id'):
            mission_elem = ET.SubElement(root, 'mission_id')
            mission_elem.text = compiled_commands['mission_id']

        if compiled_commands.get('satellite_id'):
            sat_elem = ET.SubElement(root, 'satellite_id')
            sat_elem.text = compiled_commands['satellite_id']

        # 添加指令
        commands_elem = ET.SubElement(root, 'commands')
        for cmd in compiled_commands.get('commands', []):
            cmd_elem = ET.SubElement(commands_elem, 'command')
            cmd_elem.set('id', cmd.get('command_id', ''))

            # 指令类型
            type_elem = ET.SubElement(cmd_elem, 'type')
            type_elem.text = cmd.get('command_type', '')

            # 执行时间
            exec_time_elem = ET.SubElement(cmd_elem, 'execution_time')
            exec_time_elem.text = cmd.get('execution_time', '')

            if cmd.get('end_time'):
                end_time_elem = ET.SubElement(cmd_elem, 'end_time')
                end_time_elem.text = cmd['end_time']

            # 参数
            params_elem = ET.SubElement(cmd_elem, 'parameters')
            for key, value in cmd.get('parameters', {}).items():
                param_elem = ET.SubElement(params_elem, key)
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        sub_elem = ET.SubElement(param_elem, sub_key)
                        sub_elem.text = str(sub_value) if sub_value is not None else ''
                else:
                    param_elem.text = str(value) if value is not None else ''

            # 校验和
            checksum_elem = ET.SubElement(cmd_elem, 'checksum')
            checksum_elem.text = cmd.get('checksum', '')

        # 格式化XML
        rough_string = ET.tostring(root, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
