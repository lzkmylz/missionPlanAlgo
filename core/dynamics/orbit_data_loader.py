"""
轨道数据加载器

从JSON+GZIP文件加载Java预计算的轨道数据，供调度器使用。
"""

import gzip
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

from .orbit_batch_propagator import SatelliteOrbitCache

logger = logging.getLogger(__name__)


class OrbitDataLoader:
    """轨道数据加载器 - 从JSON+GZIP文件加载预计算轨道"""

    def load_from_json(self, filepath: str) -> Dict[str, SatelliteOrbitCache]:
        """
        从JSON+GZIP文件加载所有卫星轨道数据

        Args:
            filepath: JSON+GZIP文件路径

        Returns:
            Dict[satellite_id, SatelliteOrbitCache]: 卫星轨道缓存字典
        """
        logger.info(f"从JSON+GZIP文件加载轨道数据: {filepath}")

        try:
            # 打开并读取GZIP压缩的JSON文件
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                data = json.load(f)

            metadata = data.get('metadata', {})
            orbits = data.get('orbits', [])

            logger.info(f"  总记录数: {metadata.get('total_records', len(orbits))}")
            logger.info(f"  卫星数量: {metadata.get('satellite_count', 'unknown')}")

            # 按卫星分组
            sat_groups: Dict[str, List[dict]] = {}
            for record in orbits:
                sat_id = record['satellite_id']
                if sat_id not in sat_groups:
                    sat_groups[sat_id] = []
                sat_groups[sat_id].append(record)

            # 创建SatelliteOrbitCache
            caches = {}
            for sat_id, records in sat_groups.items():
                # 按时间排序
                records.sort(key=lambda x: x['timestamp'])

                # 提取数据
                timestamps = [timedelta(seconds=int(r['timestamp'])) for r in records]
                positions = [(r['pos_x'], r['pos_y'], r['pos_z']) for r in records]
                velocities = [(r['vel_x'], r['vel_y'], r['vel_z']) for r in records]

                # 创建缓存
                cache = SatelliteOrbitCache(
                    satellite_id=sat_id,
                    timestamps=timestamps,
                    positions=positions,
                    velocities=velocities
                )

                caches[sat_id] = cache

            logger.info(f"成功加载 {len(caches)} 颗卫星的轨道数据")
            return caches

        except Exception as e:
            logger.error(f"加载JSON+GZIP文件失败: {e}")
            raise

    def load_from_json_with_start_time(
        self,
        filepath: str,
        start_time: datetime
    ) -> Dict[str, SatelliteOrbitCache]:
        """
        从JSON+GZIP文件加载轨道数据，并设置正确的场景开始时间

        Args:
            filepath: JSON+GZIP文件路径
            start_time: 场景开始时间

        Returns:
            Dict[satellite_id, SatelliteOrbitCache]: 卫星轨道缓存字典
        """
        logger.info(f"加载轨道数据并设置开始时间: {start_time}")

        caches = self.load_from_json(filepath)

        # 更新每个缓存的时间戳为实际datetime
        for sat_id, cache in caches.items():
            # 将timedelta转换为实际datetime
            cache.timestamps = [
                start_time + t for t in cache.timestamps
            ]
            cache.start_time = cache.timestamps[0]
            cache.end_time = cache.timestamps[-1]

        return caches

    def get_file_info(self, filepath: str) -> Dict:
        """
        获取JSON+GZIP文件信息

        Args:
            filepath: JSON+GZIP文件路径

        Returns:
            文件信息字典
        """
        try:
            import os

            file_size = os.path.getsize(filepath)

            # 读取元数据（只读取前1000行）
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                data = json.load(f)

            metadata = data.get('metadata', {})

            return {
                'file_size_mb': file_size / (1024 * 1024),
                'satellite_count': metadata.get('satellite_count', 0),
                'total_records': metadata.get('total_records', 0),
            }

        except Exception as e:
            logger.error(f"获取文件信息失败: {e}")
            return {}
