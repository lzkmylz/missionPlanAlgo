"""
YAML场景配置解析器

实现第3.4章设计：
- 从YAML文件加载场景配置
- 解析卫星、目标、地面站配置
- 转换为Mission对象
"""

import yaml
import os
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from pathlib import Path


class YamlLoader:
    """
    YAML场景配置加载器

    支持从YAML文件加载场景配置，包括：
    - 场景基本信息（名称、时间范围）
    - 卫星配置
    - 目标配置
    - 地面站配置
    """

    def __init__(self):
        """初始化加载器"""
        self._loaded_config: Optional[Dict[str, Any]] = None
        self._file_path: Optional[str] = None

    def load(self, file_path: str) -> Dict[str, Any]:
        """
        加载YAML文件

        Args:
            file_path: YAML文件路径

        Returns:
            Dict[str, Any]: 解析后的配置字典

        Raises:
            FileNotFoundError: 文件不存在
            yaml.YAMLError: YAML解析错误
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"YAML文件不存在: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        self._loaded_config = config
        self._file_path = file_path

        return config

    def parse_scenario_basic_info(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析场景基本信息

        Args:
            config: 配置字典

        Returns:
            Dict[str, Any]: 场景基本信息
        """
        if 'scenario' not in config:
            return {}

        scenario = config['scenario']
        result = {
            'name': scenario.get('name', 'Unnamed Scenario'),
        }

        # 解析时间范围
        if 'duration' in scenario:
            duration = scenario['duration']
            if 'start' in duration:
                start_str = duration['start']
                # 处理ISO格式时间字符串
                start_str = start_str.replace('Z', '+00:00')
                result['start_time'] = datetime.fromisoformat(start_str)
            if 'end' in duration:
                end_str = duration['end']
                end_str = end_str.replace('Z', '+00:00')
                result['end_time'] = datetime.fromisoformat(end_str)

        return result

    def parse_satellites(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        解析卫星配置

        Args:
            config: 配置字典

        Returns:
            List[Dict[str, Any]]: 卫星配置列表
        """
        if 'scenario' not in config:
            return []

        scenario = config['scenario']
        satellites = scenario.get('satellites', [])

        if satellites is None:
            return []

        return [self._normalize_satellite_config(sat) for sat in satellites]

    def _normalize_satellite_config(self, sat_config: Dict[str, Any]) -> Dict[str, Any]:
        """规范化卫星配置"""
        return {
            'id': sat_config.get('id', 'unknown'),
            'type': sat_config.get('type', 'optical'),
            'orbit': sat_config.get('orbit', {}),
            'capabilities': sat_config.get('capabilities', {})
        }

    def parse_targets(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析目标配置

        Args:
            config: 配置字典

        Returns:
            Dict[str, Any]: 目标配置
        """
        if 'scenario' not in config:
            return {}

        scenario = config['scenario']
        targets = scenario.get('targets', {})

        if targets is None:
            return {}

        return targets

    def parse_ground_stations(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        解析地面站配置

        Args:
            config: 配置字典

        Returns:
            List[Dict[str, Any]]: 地面站配置列表
        """
        if 'scenario' not in config:
            return []

        scenario = config['scenario']
        ground_stations = scenario.get('ground_stations', [])

        if ground_stations is None:
            return []

        return [self._normalize_ground_station_config(gs) for gs in ground_stations]

    def _normalize_ground_station_config(self, gs_config: Dict[str, Any]) -> Dict[str, Any]:
        """规范化地面站配置"""
        return {
            'id': gs_config.get('id', 'unknown'),
            'location': gs_config.get('location', [0.0, 0.0, 0.0]),
            'antennas': gs_config.get('antennas', [])
        }

    def validate_schema(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证配置模式

        Args:
            config: 配置字典

        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误列表)
        """
        errors = []

        if 'scenario' not in config:
            errors.append("缺少必需的 'scenario' 根节点")
            return False, errors

        scenario = config['scenario']

        # 检查必需字段
        if 'name' not in scenario:
            errors.append("scenario 缺少 'name' 字段")

        if 'duration' not in scenario:
            errors.append("scenario 缺少 'duration' 字段")
        else:
            duration = scenario['duration']
            if 'start' not in duration:
                errors.append("duration 缺少 'start' 字段")
            if 'end' not in duration:
                errors.append("duration 缺少 'end' 字段")

        return len(errors) == 0, errors

    def load_to_mission(self, file_path: str) -> Any:
        """
        加载YAML文件并转换为Mission对象

        Args:
            file_path: YAML文件路径

        Returns:
            Mission: 任务场景对象

        Raises:
            FileNotFoundError: 文件不存在
            ImportError: 无法导入Mission类
        """
        try:
            from core.models.mission import Mission
            from core.models.satellite import Satellite, SatelliteType, SatelliteCapabilities
            from core.models.target import Target, TargetType
            from core.models.ground_station import GroundStation
        except ImportError as e:
            raise ImportError(f"无法导入必要的模型类: {e}")

        config = self.load(file_path)

        # 解析基本信息
        basic_info = self.parse_scenario_basic_info(config)

        # 创建Mission对象
        mission = Mission(
            name=basic_info.get('name', 'Unnamed'),
            start_time=basic_info.get('start_time', datetime.now()),
            end_time=basic_info.get('end_time', datetime.now()),
            description=f"Loaded from {file_path}"
        )

        # 添加卫星
        for sat_config in self.parse_satellites(config):
            try:
                sat_type = SatelliteType(sat_config['type'])
            except ValueError:
                sat_type = SatelliteType.OPTICAL

            capabilities = sat_config.get('capabilities', {})
            caps = SatelliteCapabilities(
                imaging_modes=capabilities.get('imaging_modes', ['push_broom']),
                max_off_nadir=capabilities.get('max_off_nadir', 30.0),
                storage_capacity=capabilities.get('storage_capacity', 500.0),
                power_capacity=capabilities.get('power_capacity', 2000.0)
            )

            satellite = Satellite(
                id=sat_config['id'],
                name=sat_config.get('name', sat_config['id']),
                sat_type=sat_type,
                capabilities=caps
            )
            mission.add_satellite(satellite)

        # 添加目标
        targets_config = self.parse_targets(config)

        # 解析点目标群
        if 'point_group' in targets_config:
            pg_config = targets_config['point_group']
            regions = pg_config.get('regions', [])
            for region in regions:
                bounds = region.get('bounds', [0, 0, 1, 1])
                # 创建代表性点目标
                target = Target(
                    id=f"POINT_{region.get('name', 'unknown')}",
                    name=region.get('name', 'Unknown Region'),
                    target_type=TargetType.POINT,
                    longitude=(bounds[0] + bounds[2]) / 2,
                    latitude=(bounds[1] + bounds[3]) / 2,
                    priority=1
                )
                mission.add_target(target)

        # 解析区域目标
        if 'large_area' in targets_config:
            la_config = targets_config['large_area']
            areas = la_config.get('areas', [])
            for area in areas:
                vertices = area.get('vertices', [])
                if len(vertices) >= 3:
                    target = Target(
                        id=area.get('id', f"AREA_{len(mission.targets)}"),
                        name=area.get('id', f"Area {len(mission.targets)}"),
                        target_type=TargetType.AREA,
                        area_vertices=[(v[0], v[1]) for v in vertices],
                        priority=area.get('priority', 1),
                        resolution_required=area.get('resolution_required', 10.0)
                    )
                    mission.add_target(target)

        # 添加地面站
        for gs_config in self.parse_ground_stations(config):
            location = gs_config.get('location', [0.0, 0.0, 0.0])
            gs = GroundStation(
                id=gs_config['id'],
                longitude=location[0],
                latitude=location[1],
                altitude=location[2] if len(location) > 2 else 0.0
            )
            mission.add_ground_station(gs)

        return mission

    def get_loaded_config(self) -> Optional[Dict[str, Any]]:
        """获取最后加载的配置"""
        return self._loaded_config

    def get_file_path(self) -> Optional[str]:
        """获取最后加载的文件路径"""
        return self._file_path
