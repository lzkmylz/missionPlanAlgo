"""
缓存索引管理器测试

测试缓存索引的注册、查找和清理功能。
"""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path

from core.cache.index_manager import CacheIndexManager
from core.cache.index import CacheIndexEntry, CacheStatus
from core.cache.fingerprint import ScenarioFingerprint, ComponentHash


class TestCacheIndexManager:
    """测试缓存索引管理器"""

    @pytest.fixture
    def temp_index_dir(self, tmp_path):
        """创建临时索引目录"""
        index_dir = tmp_path / "cache"
        index_dir.mkdir()
        return index_dir

    @pytest.fixture
    def sample_fingerprint(self):
        """创建示例指纹"""
        return ScenarioFingerprint(
            full_hash="a1b2c3d4e5f6789012345678901234567890abcd",
            satellites=ComponentHash("sat1234567890abcd", "satellites", 2, {"SAT-01"}),
            ground_stations=ComponentHash("gs1234567890abcd", "ground_stations", 1, {"GS-01"}),
            targets=ComponentHash("tgt1234567890abcd", "targets", 10, {"TGT-01"}),
            time_range=ComponentHash("tr1234567890abcd", "time_range", 1, set()),
            scenario_name="Test Scenario",
            created_at=datetime.now()
        )

    def test_initialization(self, temp_index_dir):
        """测试初始化"""
        manager = CacheIndexManager(
            index_path=str(temp_index_dir / "index.json"),
            auto_save=False
        )

        assert manager.index_path == temp_index_dir / "index.json"
        assert manager.cache_dir.exists()
        assert manager.orbit_dir.exists()

    def test_register_and_find(self, temp_index_dir, sample_fingerprint):
        """测试注册和查找"""
        manager = CacheIndexManager(
            index_path=str(temp_index_dir / "index.json"),
            auto_save=False
        )

        # 创建虚拟缓存文件
        cache_file = temp_index_dir / "test_cache.json"
        cache_file.write_text("{}")

        # 注册
        entry = manager.register(
            fingerprint=sample_fingerprint,
            cache_file=str(cache_file),
            orbit_file=None,
            stats={"total_windows": 100}
        )

        assert entry.full_hash == sample_fingerprint.full_hash
        assert entry.cache_file == str(cache_file)
        assert entry.stats["total_windows"] == 100

        # 查找
        found = manager.find(sample_fingerprint)
        assert found is not None
        assert found.full_hash == sample_fingerprint.full_hash
        assert found.access_count == 1  # 访问次数增加

    def test_find_nonexistent(self, temp_index_dir, sample_fingerprint):
        """测试查找不存在的缓存"""
        manager = CacheIndexManager(
            index_path=str(temp_index_dir / "index.json"),
            auto_save=False
        )

        found = manager.find(sample_fingerprint)
        assert found is None

    def test_find_by_components(self, temp_index_dir):
        """测试按组件查找"""
        manager = CacheIndexManager(
            index_path=str(temp_index_dir / "index.json"),
            auto_save=False
        )

        # 创建两个指纹，部分相同
        fp1 = ScenarioFingerprint(
            full_hash="hash1" + "a" * 60,
            satellites=ComponentHash("sat_same", "satellites", 2, {"SAT-01"}),
            ground_stations=ComponentHash("gs_diff1", "ground_stations", 1, {"GS-01"}),
            targets=ComponentHash("tgt_diff1", "targets", 10, {"TGT-01"}),
            time_range=ComponentHash("tr_same", "time_range", 1, set()),
            scenario_name="Scene 1",
            created_at=datetime.now()
        )

        fp2 = ScenarioFingerprint(
            full_hash="hash2" + "b" * 60,
            satellites=ComponentHash("sat_same", "satellites", 2, {"SAT-01"}),
            ground_stations=ComponentHash("gs_diff2", "ground_stations", 1, {"GS-02"}),
            targets=ComponentHash("tgt_diff2", "targets", 10, {"TGT-02"}),
            time_range=ComponentHash("tr_same", "time_range", 1, set()),
            scenario_name="Scene 2",
            created_at=datetime.now()
        )

        # 注册第一个
        cache_file = temp_index_dir / "cache1.json"
        cache_file.write_text("{}")
        manager.register(fp1, str(cache_file))

        # 按卫星查找第二个（应该匹配）
        matches = manager.find_by_components(fp2, ['satellites'])
        assert len(matches) == 1
        assert matches[0][1] == 1.0  # 完全匹配

        # 按地面站查找（应该不匹配）
        matches = manager.find_by_components(fp2, ['ground_stations'])
        assert len(matches) == 0

    def test_find_reusable_orbit_cache(self, temp_index_dir):
        """测试查找可复用的轨道缓存"""
        manager = CacheIndexManager(
            index_path=str(temp_index_dir / "index.json"),
            auto_save=False
        )

        # 创建轨道缓存
        fp = ScenarioFingerprint(
            full_hash="hash1" + "a" * 60,
            satellites=ComponentHash("sat_same", "satellites", 2, {"SAT-01"}),
            ground_stations=ComponentHash("gs1", "ground_stations", 1, {"GS-01"}),
            targets=ComponentHash("tgt1", "targets", 10, {"TGT-01"}),
            time_range=ComponentHash("tr_same", "time_range", 1, set()),
            scenario_name="Original",
            created_at=datetime.now()
        )

        cache_file = temp_index_dir / "cache.json"
        orbit_file = temp_index_dir / "orbit.json.gz"
        cache_file.write_text("{}")
        orbit_file.write_text("orbit data")

        manager.register(fp, str(cache_file), str(orbit_file))

        # 创建新场景，相同卫星和时间范围
        new_fp = ScenarioFingerprint(
            full_hash="hash2" + "b" * 60,
            satellites=ComponentHash("sat_same", "satellites", 2, {"SAT-01"}),
            ground_stations=ComponentHash("gs2", "ground_stations", 1, {"GS-02"}),
            targets=ComponentHash("tgt2", "targets", 5, {"TGT-02"}),
            time_range=ComponentHash("tr_same", "time_range", 1, set()),
            scenario_name="New",
            created_at=datetime.now()
        )

        # 应该找到可复用的轨道缓存
        found = manager.find_reusable_orbit_cache(new_fp)
        assert found is not None
        assert found.orbit_file == str(orbit_file)

    def test_list_entries(self, temp_index_dir, sample_fingerprint):
        """测试列出条目"""
        manager = CacheIndexManager(
            index_path=str(temp_index_dir / "index.json"),
            auto_save=False
        )

        # 注册多个
        for i in range(3):
            fp = ScenarioFingerprint(
                full_hash=f"hash{i}" + "a" * 60,
                satellites=ComponentHash(f"sat{i}", "satellites", i+1, set()),
                ground_stations=ComponentHash(f"gs{i}", "ground_stations", 1, set()),
                targets=ComponentHash(f"tgt{i}", "targets", 10, set()),
                time_range=ComponentHash(f"tr{i}", "time_range", 1, set()),
                scenario_name=f"Scene {i}",
                created_at=datetime.now()
            )
            cache_file = temp_index_dir / f"cache{i}.json"
            cache_file.write_text("{}")
            manager.register(fp, str(cache_file))

        entries = manager.list_entries()
        assert len(entries) == 3

    def test_cleanup_by_age(self, temp_index_dir, sample_fingerprint):
        """测试按年龄清理"""
        manager = CacheIndexManager(
            index_path=str(temp_index_dir / "index.json"),
            auto_save=False
        )

        # 注册
        cache_file = temp_index_dir / "old_cache.json"
        cache_file.write_text("{}")
        entry = manager.register(sample_fingerprint, str(cache_file))

        # 修改访问时间为很久以前
        entry.accessed_at = datetime.now() - timedelta(days=100)

        # 清理60天前的
        stats = manager.cleanup(older_than_days=60, dry_run=False)

        assert stats['deleted'] == 1
        assert not cache_file.exists()

    def test_get_stats(self, temp_index_dir, sample_fingerprint):
        """测试获取统计信息"""
        manager = CacheIndexManager(
            index_path=str(temp_index_dir / "index.json"),
            auto_save=False
        )

        # 初始为空
        stats = manager.get_stats()
        assert stats['total_entries'] == 0
        assert stats['valid_entries'] == 0

        # 注册一个
        cache_file = temp_index_dir / "cache.json"
        cache_file.write_text("{}")
        manager.register(sample_fingerprint, str(cache_file))

        stats = manager.get_stats()
        assert stats['total_entries'] == 1
        assert stats['valid_entries'] == 1

    def test_persistence(self, temp_index_dir, sample_fingerprint):
        """测试索引持久化"""
        index_path = temp_index_dir / "index.json"

        # 创建并注册
        manager1 = CacheIndexManager(index_path=str(index_path), auto_save=True)
        cache_file = temp_index_dir / "cache.json"
        cache_file.write_text("{}")
        manager1.register(sample_fingerprint, str(cache_file))

        # 重新加载
        manager2 = CacheIndexManager(index_path=str(index_path), auto_save=False)

        found = manager2.find(sample_fingerprint)
        assert found is not None
        assert found.full_hash == sample_fingerprint.full_hash
