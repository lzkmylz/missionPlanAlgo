"""
缓存索引管理器

提供缓存索引的加载、保存、查询和管理功能。
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime, timedelta
from threading import Lock

from .index import CacheIndexEntry, CacheStatus
from .fingerprint import ScenarioFingerprint, HASH_TRUNCATE_LENGTH

logger = logging.getLogger(__name__)


class CacheIndexManager:
    """
    缓存索引管理器

    职责：
    1. 维护缓存索引文件
    2. 提供缓存查找功能
    3. 支持缓存生命周期管理
    4. 线程安全
    """

    DEFAULT_INDEX_PATH = Path("cache/index.json")
    DEFAULT_CACHE_DIR = Path("cache/windows")
    DEFAULT_ORBIT_DIR = Path("cache/orbits")

    def __init__(
        self,
        index_path: Optional[str] = None,
        auto_save: bool = True
    ):
        self.index_path = Path(index_path) if index_path else self.DEFAULT_INDEX_PATH
        self.cache_dir = self.index_path.parent / "windows"
        self.orbit_dir = self.index_path.parent / "orbits"
        self.auto_save = auto_save
        self._lock = Lock()

        # 确保目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.orbit_dir.mkdir(parents=True, exist_ok=True)

        # 加载索引
        self._index = self._load_index()

    def _load_index(self) -> Dict[str, CacheIndexEntry]:
        """加载索引文件"""
        if not self.index_path.exists():
            return {}

        try:
            with open(self.index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            entries = {}
            for hash_key, entry_data in data.get('entries', {}).items():
                try:
                    entries[hash_key] = CacheIndexEntry.from_dict(entry_data)
                except Exception as e:
                    logger.warning(f"Failed to parse cache entry {hash_key}: {e}")

            return entries
        except Exception as e:
            logger.warning(f"Failed to load cache index: {e}")
            return {}

    def _save_index(self) -> None:
        """保存索引文件"""
        data = {
            'version': '1.0',
            'last_updated': datetime.now().isoformat(),
            'entries': {
                k: v.to_dict() for k, v in self._index.items()
            }
        }

        # 原子写入（使用shutil.move确保跨平台兼容）
        temp_path = self.index_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            shutil.move(str(temp_path), str(self.index_path))
        except Exception:
            # 清理临时文件
            if temp_path.exists():
                temp_path.unlink()
            raise

    def register(
        self,
        fingerprint: ScenarioFingerprint,
        cache_file: str,
        orbit_file: Optional[str] = None,
        stats: Optional[Dict[str, Any]] = None
    ) -> CacheIndexEntry:
        """
        注册新缓存

        Args:
            fingerprint: 场景指纹
            cache_file: 缓存文件路径
            orbit_file: 轨道文件路径(可选)
            stats: 统计信息

        Returns:
            CacheIndexEntry: 创建的索引条目
        """
        with self._lock:
            # 获取文件大小
            cache_size = Path(cache_file).stat().st_size / (1024 * 1024) if Path(cache_file).exists() else 0
            orbit_size = Path(orbit_file).stat().st_size / (1024 * 1024) if orbit_file and Path(orbit_file).exists() else 0

            entry = CacheIndexEntry(
                full_hash=fingerprint.full_hash,
                cache_file=str(cache_file),
                orbit_file=str(orbit_file) if orbit_file else None,
                satellites_hash=fingerprint.satellites.hash_value,
                ground_stations_hash=fingerprint.ground_stations.hash_value,
                targets_hash=fingerprint.targets.hash_value,
                time_range_hash=fingerprint.time_range.hash_value,
                scenario_name=fingerprint.scenario_name,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                access_count=0,
                file_size_mb=cache_size + orbit_size,
                status=CacheStatus.VALID,
                stats=stats or {}
            )

            key = fingerprint.get_cache_key()
            self._index[key] = entry

            if self.auto_save:
                self._save_index()

            return entry

    def find(self, fingerprint: ScenarioFingerprint) -> Optional[CacheIndexEntry]:
        """
        查找匹配的缓存条目

        查找优先级：
        1. 完全匹配（完整哈希相同）
        2. 检查文件是否存在且有效
        """
        with self._lock:
            key = fingerprint.get_cache_key()
            entry = self._index.get(key)

            if entry is None:
                return None

            # 检查状态
            if entry.status != CacheStatus.VALID:
                return None

            # 检查文件是否存在
            if not Path(entry.cache_file).exists():
                entry.status = CacheStatus.CORRUPTED
                if self.auto_save:
                    self._save_index()
                return None

            # 更新访问统计
            entry.access_count += 1
            entry.accessed_at = datetime.now()

            if self.auto_save:
                self._save_index()

            return entry

    def find_by_components(
        self,
        fingerprint: ScenarioFingerprint,
        required_components: List[str]
    ) -> List[Tuple[CacheIndexEntry, float]]:
        """
        按组件查找缓存

        Args:
            fingerprint: 场景指纹
            required_components: 必须匹配的组件列表
                ['satellites', 'ground_stations', 'targets', 'time_range']

        Returns:
            List[(entry, match_score)]: 匹配的条目及匹配度(0-1)
        """
        with self._lock:
            matches = []

            for entry in self._index.values():
                if entry.status != CacheStatus.VALID:
                    continue

                score = 0
                total = len(required_components)

                if 'satellites' in required_components:
                    if entry.satellites_hash == fingerprint.satellites.hash_value:
                        score += 1

                if 'ground_stations' in required_components:
                    if entry.ground_stations_hash == fingerprint.ground_stations.hash_value:
                        score += 1

                if 'targets' in required_components:
                    if entry.targets_hash == fingerprint.targets.hash_value:
                        score += 1

                if 'time_range' in required_components:
                    if entry.time_range_hash == fingerprint.time_range.hash_value:
                        score += 1

                match_ratio = score / total if total > 0 else 0
                if match_ratio > 0:
                    matches.append((entry, match_ratio))

            # 按匹配度排序
            matches.sort(key=lambda x: x[1], reverse=True)
            return matches

    def find_reusable_orbit_cache(
        self,
        fingerprint: ScenarioFingerprint
    ) -> Optional[CacheIndexEntry]:
        """
        查找可复用的轨道缓存

        条件：卫星配置和时间范围相同
        """
        matches = self.find_by_components(
            fingerprint,
            ['satellites', 'time_range']
        )

        # 返回最佳匹配
        for entry, score in matches:
            if score == 1.0 and entry.orbit_file and Path(entry.orbit_file).exists():
                return entry

        return None

    def list_entries(
        self,
        status: Optional[CacheStatus] = None
    ) -> List[CacheIndexEntry]:
        """列出所有缓存条目"""
        with self._lock:
            entries = list(self._index.values())
            if status:
                entries = [e for e in entries if e.status == status]
            return sorted(entries, key=lambda e: e.accessed_at, reverse=True)

    def cleanup(
        self,
        older_than_days: Optional[int] = None,
        max_size_mb: Optional[float] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        清理缓存

        Args:
            older_than_days: 删除超过指定天数的缓存
            max_size_mb: 当总大小超过此值时，删除最少使用的缓存
            dry_run: 仅统计不实际删除

        Returns:
            清理统计信息
        """
        with self._lock:
            stats = {'deleted': 0, 'freed_mb': 0.0, 'entries': []}
            to_delete = []

            # 按过期时间清理
            if older_than_days:
                cutoff = datetime.now() - timedelta(days=older_than_days)
                for entry in self._index.values():
                    if entry.accessed_at < cutoff:
                        to_delete.append(entry)

            # 按大小清理
            if max_size_mb:
                total_size = sum(e.file_size_mb for e in self._index.values())
                if total_size > max_size_mb:
                    # 按访问时间排序，删除最少使用的
                    sorted_entries = sorted(
                        self._index.values(),
                        key=lambda e: (e.access_count, e.accessed_at)
                    )

                    size_to_free = total_size - max_size_mb
                    freed = 0.0
                    for entry in sorted_entries:
                        if entry not in to_delete:
                            to_delete.append(entry)
                            freed += entry.file_size_mb
                            if freed >= size_to_free:
                                break

            # 执行删除
            for entry in to_delete:
                if not dry_run:
                    # 删除文件
                    try:
                        if Path(entry.cache_file).exists():
                            Path(entry.cache_file).unlink()
                        if entry.orbit_file and Path(entry.orbit_file).exists():
                            Path(entry.orbit_file).unlink()
                    except Exception as e:
                        print(f"Warning: Failed to delete cache files: {e}")

                    # 更新索引
                    key = entry.full_hash[:HASH_TRUNCATE_LENGTH]
                    if key in self._index:
                        del self._index[key]

                stats['deleted'] += 1
                stats['freed_mb'] += entry.file_size_mb
                stats['entries'].append({
                    'hash': entry.full_hash[:HASH_TRUNCATE_LENGTH],
                    'name': entry.scenario_name,
                    'size_mb': entry.file_size_mb
                })

            if not dry_run and self.auto_save:
                self._save_index()

            return stats

    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        with self._lock:
            entries = list(self._index.values())
            valid_entries = [e for e in entries if e.status == CacheStatus.VALID]

            return {
                'total_entries': len(entries),
                'valid_entries': len(valid_entries),
                'total_size_mb': sum(e.file_size_mb for e in valid_entries),
                'avg_access_count': sum(e.access_count for e in valid_entries) / len(valid_entries) if valid_entries else 0,
                'oldest_cache': min((e.created_at for e in valid_entries), default=None),
                'newest_cache': max((e.created_at for e in valid_entries), default=None)
            }

    def invalidate(self, full_hash: str) -> bool:
        """
        使指定缓存条目失效

        Args:
            full_hash: 完整哈希值

        Returns:
            是否成功找到并失效
        """
        with self._lock:
            key = full_hash[:HASH_TRUNCATE_LENGTH]
            entry = self._index.get(key)

            if entry:
                entry.status = CacheStatus.STALE
                if self.auto_save:
                    self._save_index()
                return True

            return False
