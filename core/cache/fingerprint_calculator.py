"""
场景指纹计算器

提供场景指纹计算和比较功能。
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Set, Any, Optional, List
from datetime import datetime

from .fingerprint import ScenarioFingerprint, ComponentHash, HASH_TRUNCATE_LENGTH


class FingerprintCalculator:
    """场景指纹计算器"""

    def __init__(self, hash_algorithm: str = 'sha256'):
        self.hash_algorithm = hash_algorithm

    def calculate(self, scenario_path: str) -> ScenarioFingerprint:
        """
        计算场景文件的完整指纹

        Args:
            scenario_path: 场景文件路径

        Returns:
            ScenarioFingerprint: 场景指纹对象
        """
        scenario_data = self._load_scenario(scenario_path)
        return self.calculate_from_data(scenario_data)

    def calculate_from_data(self, scenario_data: Dict[str, Any]) -> ScenarioFingerprint:
        """
        从场景数据字典计算指纹

        Args:
            scenario_data: 场景数据字典

        Returns:
            ScenarioFingerprint: 场景指纹对象
        """
        # 计算各组件哈希
        satellites_hash = self._calculate_satellites_hash(
            scenario_data.get('satellites', [])
        )
        ground_stations_hash = self._calculate_ground_stations_hash(
            scenario_data.get('ground_stations', [])
        )
        targets_hash = self._calculate_targets_hash(
            scenario_data.get('targets', [])
        )
        time_range_hash = self._calculate_time_range_hash(
            scenario_data.get('duration', {})
        )

        # 计算完整哈希
        full_content = {
            'satellites': satellites_hash.hash_value,
            'ground_stations': ground_stations_hash.hash_value,
            'targets': targets_hash.hash_value,
            'time_range': time_range_hash.hash_value
        }
        full_hash = self._hash_dict(full_content)

        return ScenarioFingerprint(
            full_hash=full_hash,
            satellites=satellites_hash,
            ground_stations=ground_stations_hash,
            targets=targets_hash,
            time_range=time_range_hash,
            scenario_name=scenario_data.get('name', 'Unknown'),
            created_at=datetime.now()
        )

    def _load_scenario(self, path: str) -> Dict[str, Any]:
        """加载场景文件并进行基本验证"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 验证必要字段
        if not isinstance(data, dict):
            raise ValueError(f"场景文件格式错误: 根对象必须是字典")

        required_fields = ['satellites', 'duration']
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(f"场景文件缺少必要字段: {missing}")

        # 验证duration字段
        duration = data.get('duration', {})
        if not isinstance(duration, dict):
            raise ValueError("duration字段必须是字典")
        if 'start' not in duration or 'end' not in duration:
            raise ValueError("duration字段必须包含'start'和'end'")

        return data

    def _calculate_satellites_hash(self, satellites: list) -> ComponentHash:
        """计算卫星配置哈希"""
        # 规范化并排序
        normalized = []
        sat_ids = set()

        for sat in satellites:
            # 提取关键字段
            orbit = sat.get('orbit', {})
            capabilities = sat.get('capabilities', {})

            norm_sat = {
                'id': sat.get('id'),
                'orbit': self._normalize_orbit(orbit),
                'capabilities': self._normalize_capabilities(capabilities)
            }
            normalized.append(norm_sat)
            sat_id = sat.get('id')
            if sat_id:
                sat_ids.add(sat_id)

        # 按ID排序确保一致性
        normalized.sort(key=lambda x: x['id'])

        hash_value = self._hash_dict(normalized)

        return ComponentHash(
            hash_value=hash_value[:HASH_TRUNCATE_LENGTH],
            component_type='satellites',
            item_count=len(satellites),
            item_ids=sat_ids
        )

    def _normalize_orbit(self, orbit: Dict[str, Any]) -> Dict[str, Any]:
        """规范化轨道参数"""
        # 提取影响计算的轨道要素
        return {
            'semi_major_axis': orbit.get('semi_major_axis'),
            'eccentricity': orbit.get('eccentricity'),
            'inclination': orbit.get('inclination'),
            'raan': orbit.get('raan'),
            'arg_of_perigee': orbit.get('arg_of_perigee'),
            'mean_anomaly': orbit.get('mean_anomaly'),
            'epoch': orbit.get('epoch')
        }

    def _normalize_capabilities(self, caps: Dict[str, Any]) -> Dict[str, Any]:
        """规范化卫星能力参数"""
        # 提取影响姿态和成像的能力参数
        agility = caps.get('agility', {})
        return {
            'max_roll_angle': caps.get('max_roll_angle'),
            'max_pitch_angle': caps.get('max_pitch_angle'),
            'max_roll_rate': agility.get('max_roll_rate'),
            'max_pitch_rate': agility.get('max_pitch_rate'),
            'max_roll_acceleration': agility.get('max_roll_acceleration'),
            'max_pitch_acceleration': agility.get('max_pitch_acceleration'),
            'resolution': caps.get('resolution'),
            'swath_width': caps.get('swath_width'),
            'storage_capacity': caps.get('storage_capacity'),
            'power_capacity': caps.get('power_capacity')
        }

    def _calculate_ground_stations_hash(self, stations: list) -> ComponentHash:
        """计算地面站配置哈希"""
        normalized = []
        gs_ids = set()

        for gs in stations:
            norm_gs = {
                'id': gs.get('id'),
                'latitude': gs.get('latitude'),
                'longitude': gs.get('longitude'),
                'altitude': gs.get('altitude', 0),
                'min_elevation': gs.get('min_elevation', 5.0)
            }
            normalized.append(norm_gs)
            gs_id = gs.get('id')
            if gs_id:
                gs_ids.add(gs_id)

        normalized.sort(key=lambda x: x['id'])
        hash_value = self._hash_dict(normalized)

        return ComponentHash(
            hash_value=hash_value[:16],
            component_type='ground_stations',
            item_count=len(stations),
            item_ids=gs_ids
        )

    def _calculate_targets_hash(self, targets: list) -> ComponentHash:
        """计算目标配置哈希"""
        normalized = []
        target_ids = set()

        for tgt in targets:
            # 支持点目标和区域目标
            norm_tgt = {
                'id': tgt.get('id'),
                'type': tgt.get('type', 'point')
            }

            if norm_tgt['type'] == 'point':
                norm_tgt['latitude'] = tgt.get('latitude')
                norm_tgt['longitude'] = tgt.get('longitude')
            elif norm_tgt['type'] == 'area':
                # 区域目标使用边界点哈希
                boundary = tgt.get('boundary', [])
                norm_tgt['boundary_hash'] = self._hash_dict(boundary)[:8]

            # 添加观测需求到哈希计算
            if 'observation_requirements' in tgt:
                reqs = tgt['observation_requirements']
                norm_tgt['requirements'] = {
                    'required_observations': reqs.get('required_observations'),
                    'min_interval_hours': reqs.get('min_interval_hours'),
                    'priority': reqs.get('priority')
                }

            # 精准需求字段纳入哈希（排序保证顺序无关）
            # 注意：key 名称与 Target.from_dict() 和场景 JSON 顶层字段保持一致，
            # 若将来修改字段名，此处和 Target.from_dict() 必须同步更新。
            # 旧字段 required_satellite_type / required_imaging_mode 也必须纳入，
            # 否则只改旧字段的两个场景会命中同一缓存条目。
            allowed_types = sorted(tgt.get('allowed_satellite_types', []))
            allowed_ids = sorted(tgt.get('allowed_satellite_ids', []))
            req_modes = sorted(tgt.get('required_imaging_modes', []))
            legacy_sat_type = tgt.get('required_satellite_type')
            legacy_mode = tgt.get('required_imaging_mode')
            if allowed_types or allowed_ids or req_modes or legacy_sat_type or legacy_mode:
                norm_tgt['precise_requirements'] = {
                    'allowed_satellite_types': allowed_types,
                    'allowed_satellite_ids': allowed_ids,
                    'required_imaging_modes': req_modes,
                    'required_satellite_type': legacy_sat_type,
                    'required_imaging_mode': legacy_mode,
                }

            normalized.append(norm_tgt)
            tgt_id = tgt.get('id')
            if tgt_id:
                target_ids.add(tgt_id)

        normalized.sort(key=lambda x: x['id'])
        hash_value = self._hash_dict(normalized)

        return ComponentHash(
            hash_value=hash_value[:16],
            component_type='targets',
            item_count=len(targets),
            item_ids=target_ids
        )

    def _calculate_time_range_hash(self, duration: Dict[str, Any]) -> ComponentHash:
        """计算时间范围哈希"""
        normalized = {
            'start': duration.get('start'),
            'end': duration.get('end')
        }
        hash_value = self._hash_dict(normalized)

        return ComponentHash(
            hash_value=hash_value[:16],
            component_type='time_range',
            item_count=1,
            item_ids=set()
        )

    def _hash_dict(self, data: Any) -> str:
        """计算字典的哈希值"""
        # 使用JSON序列化确保一致性
        content = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        hasher = hashlib.new(self.hash_algorithm)
        hasher.update(content.encode('utf-8'))
        return hasher.hexdigest()


class FingerprintComparator:
    """指纹比较器 - 分析场景差异"""

    def compare(
        self,
        fp1: ScenarioFingerprint,
        fp2: ScenarioFingerprint
    ) -> Dict[str, Any]:
        """
        比较两个场景指纹，返回差异分析

        Returns:
            {
                'identical': bool,      # 是否完全相同
                'same_satellites': bool,
                'same_ground_stations': bool,
                'same_targets': bool,
                'same_time_range': bool,
                'common_satellites': set,
                'common_targets': set,
                'recommendation': str    # 建议操作
            }
        """
        result = {
            'identical': fp1.full_hash == fp2.full_hash,
            'same_satellites': fp1.satellites.hash_value == fp2.satellites.hash_value,
            'same_ground_stations': fp1.ground_stations.hash_value == fp2.ground_stations.hash_value,
            'same_targets': fp1.targets.hash_value == fp2.targets.hash_value,
            'same_time_range': fp1.time_range.hash_value == fp2.time_range.hash_value,
            'common_satellites': fp1.satellites.item_ids & fp2.satellites.item_ids,
            'common_targets': fp1.targets.item_ids & fp2.targets.item_ids,
        }

        # 生成建议
        if result['identical']:
            result['recommendation'] = 'Scenes are identical, reuse full cache'
            result['reusable_components'] = ['all']
        elif result['same_satellites'] and result['same_time_range']:
            result['recommendation'] = 'Can reuse orbit cache, recompute visibility'
            result['reusable_components'] = ['orbit']
        elif result['same_satellites']:
            result['recommendation'] = 'Satellite configuration identical but time range differs'
            result['reusable_components'] = ['satellite_config']
        elif len(result['common_satellites']) > 0:
            result['recommendation'] = f'Can reuse {len(result["common_satellites"])} common satellite orbits'
            result['reusable_components'] = ['partial_orbit']
        else:
            result['recommendation'] = 'No cache reuse possible'
            result['reusable_components'] = []

        return result
