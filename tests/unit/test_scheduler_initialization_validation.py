"""
调度器初始化验证测试

TDD测试文件 - 验证所有调度器的初始化检查

测试覆盖：
1. 基础GreedyScheduler初始化检查
2. EDDScheduler初始化检查
3. SPTScheduler初始化检查
4. 元启发式调度器初始化检查（GA, ACO, PSO, SA, Tabu）
5. 各种缺少字段的情况
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from core.models import Mission, Satellite, SatelliteType, Target, TargetType
from scheduler.greedy.greedy_scheduler import GreedyScheduler
from scheduler.greedy.edd_scheduler import EDDScheduler
from scheduler.greedy.spt_scheduler import SPTScheduler
from scheduler.metaheuristic.ga_scheduler import GAScheduler
from scheduler.metaheuristic.aco_scheduler import ACOScheduler
from scheduler.metaheuristic.pso_scheduler import PSOScheduler
from scheduler.metaheuristic.sa_scheduler import SAScheduler
from scheduler.metaheuristic.tabu_scheduler import TabuScheduler


class TestGreedySchedulerInitialization:
    """测试GreedyScheduler初始化验证"""

    def setup_method(self):
        """设置测试数据"""
        self.satellite = Satellite(
            id="SAT-01",
            name="测试卫星",
            sat_type=SatelliteType.OPTICAL_1
        )
        self.target = Target(
            id="TARGET-01",
            name="测试目标",
            target_type=TargetType.POINT,
            longitude=116.4,
            latitude=39.9,
            priority=5
        )
        self.mission = Mission(
            name="测试场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[self.satellite],
            targets=[self.target]
        )

    def test_schedule_without_initialize_raises_error(self):
        """测试未初始化时调度抛出错误"""
        scheduler = GreedyScheduler()

        with pytest.raises(RuntimeError, match="Scheduler not initialized"):
            scheduler.schedule()

    def test_schedule_with_none_mission_raises_error(self):
        """测试mission为None时调度抛出错误"""
        scheduler = GreedyScheduler()
        scheduler.mission = None

        with pytest.raises(RuntimeError, match="Scheduler not initialized: mission is None"):
            scheduler.schedule()

    def test_schedule_with_empty_satellites_raises_error(self):
        """测试satellites为空列表时调度抛出错误"""
        scheduler = GreedyScheduler()

        # 创建没有卫星的mission
        mission_no_sats = Mission(
            name="无卫星场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[],
            targets=[self.target]
        )
        scheduler.initialize(mission_no_sats)

        with pytest.raises(RuntimeError, match="Scheduler not initialized: no satellites available"):
            scheduler.schedule()

    def test_schedule_with_empty_targets_raises_error(self):
        """测试targets为空列表时调度抛出错误"""
        scheduler = GreedyScheduler()

        # 创建没有目标的mission
        mission_no_targets = Mission(
            name="无目标场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[self.satellite],
            targets=[]
        )
        scheduler.initialize(mission_no_targets)

        with pytest.raises(RuntimeError, match="Scheduler not initialized: no targets available"):
            scheduler.schedule()

    def test_schedule_with_valid_mission_succeeds(self):
        """测试有效mission时调度成功"""
        scheduler = GreedyScheduler()
        scheduler.initialize(self.mission)

        # 应该正常执行不抛出异常
        result = scheduler.schedule()
        assert result is not None


class TestEDDSchedulerInitialization:
    """测试EDDScheduler初始化验证"""

    def setup_method(self):
        """设置测试数据"""
        self.satellite = Satellite(
            id="SAT-01",
            name="测试卫星",
            sat_type=SatelliteType.OPTICAL_1
        )
        self.target = Target(
            id="TARGET-01",
            name="测试目标",
            target_type=TargetType.POINT,
            longitude=116.4,
            latitude=39.9,
            priority=5
        )
        self.mission = Mission(
            name="测试场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[self.satellite],
            targets=[self.target]
        )

    def test_schedule_without_initialize_raises_error(self):
        """测试未初始化时调度抛出错误"""
        scheduler = EDDScheduler()

        with pytest.raises(RuntimeError, match="Scheduler not initialized"):
            scheduler.schedule()

    def test_schedule_with_none_mission_raises_error(self):
        """测试mission为None时调度抛出错误"""
        scheduler = EDDScheduler()
        scheduler.mission = None

        with pytest.raises(RuntimeError, match="Scheduler not initialized: mission is None"):
            scheduler.schedule()

    def test_schedule_with_empty_satellites_raises_error(self):
        """测试satellites为空列表时调度抛出错误"""
        scheduler = EDDScheduler()

        mission_no_sats = Mission(
            name="无卫星场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[],
            targets=[self.target]
        )
        scheduler.initialize(mission_no_sats)

        with pytest.raises(RuntimeError, match="Scheduler not initialized: no satellites available"):
            scheduler.schedule()

    def test_schedule_with_empty_targets_raises_error(self):
        """测试targets为空列表时调度抛出错误"""
        scheduler = EDDScheduler()

        mission_no_targets = Mission(
            name="无目标场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[self.satellite],
            targets=[]
        )
        scheduler.initialize(mission_no_targets)

        with pytest.raises(RuntimeError, match="Scheduler not initialized: no targets available"):
            scheduler.schedule()

    def test_schedule_with_valid_mission_succeeds(self):
        """测试有效mission时调度成功"""
        scheduler = EDDScheduler()
        scheduler.initialize(self.mission)

        result = scheduler.schedule()
        assert result is not None


class TestSPTSchedulerInitialization:
    """测试SPTScheduler初始化验证"""

    def setup_method(self):
        """设置测试数据"""
        self.satellite = Satellite(
            id="SAT-01",
            name="测试卫星",
            sat_type=SatelliteType.OPTICAL_1
        )
        self.target = Target(
            id="TARGET-01",
            name="测试目标",
            target_type=TargetType.POINT,
            longitude=116.4,
            latitude=39.9,
            priority=5
        )
        self.mission = Mission(
            name="测试场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[self.satellite],
            targets=[self.target]
        )

    def test_schedule_without_initialize_raises_error(self):
        """测试未初始化时调度抛出错误"""
        scheduler = SPTScheduler()

        with pytest.raises(RuntimeError, match="Scheduler not initialized"):
            scheduler.schedule()

    def test_schedule_with_none_mission_raises_error(self):
        """测试mission为None时调度抛出错误"""
        scheduler = SPTScheduler()
        scheduler.mission = None

        with pytest.raises(RuntimeError, match="Scheduler not initialized: mission is None"):
            scheduler.schedule()

    def test_schedule_with_empty_satellites_raises_error(self):
        """测试satellites为空列表时调度抛出错误"""
        scheduler = SPTScheduler()

        mission_no_sats = Mission(
            name="无卫星场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[],
            targets=[self.target]
        )
        scheduler.initialize(mission_no_sats)

        with pytest.raises(RuntimeError, match="Scheduler not initialized: no satellites available"):
            scheduler.schedule()

    def test_schedule_with_empty_targets_raises_error(self):
        """测试targets为空列表时调度抛出错误"""
        scheduler = SPTScheduler()

        mission_no_targets = Mission(
            name="无目标场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[self.satellite],
            targets=[]
        )
        scheduler.initialize(mission_no_targets)

        with pytest.raises(RuntimeError, match="Scheduler not initialized: no targets available"):
            scheduler.schedule()

    def test_schedule_with_valid_mission_succeeds(self):
        """测试有效mission时调度成功"""
        scheduler = SPTScheduler()
        scheduler.initialize(self.mission)

        result = scheduler.schedule()
        assert result is not None


class TestGASchedulerInitialization:
    """测试GAScheduler初始化验证"""

    def setup_method(self):
        """设置测试数据"""
        self.satellite = Satellite(
            id="SAT-01",
            name="测试卫星",
            sat_type=SatelliteType.OPTICAL_1
        )
        self.target = Target(
            id="TARGET-01",
            name="测试目标",
            target_type=TargetType.POINT,
            longitude=116.4,
            latitude=39.9,
            priority=5
        )
        self.mission = Mission(
            name="测试场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[self.satellite],
            targets=[self.target]
        )

    def test_schedule_without_initialize_raises_error(self):
        """测试未初始化时调度抛出错误"""
        scheduler = GAScheduler()

        with pytest.raises(RuntimeError, match="Scheduler not initialized"):
            scheduler.schedule()

    def test_schedule_with_none_mission_raises_error(self):
        """测试mission为None时调度抛出错误"""
        scheduler = GAScheduler()
        scheduler.mission = None

        with pytest.raises(RuntimeError, match="Scheduler not initialized: mission is None"):
            scheduler.schedule()

    def test_schedule_with_empty_satellites_raises_error(self):
        """测试satellites为空列表时调度抛出错误"""
        scheduler = GAScheduler()

        mission_no_sats = Mission(
            name="无卫星场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[],
            targets=[self.target]
        )
        scheduler.initialize(mission_no_sats)

        with pytest.raises(RuntimeError, match="Scheduler not initialized: no satellites available"):
            scheduler.schedule()

    def test_schedule_with_empty_targets_raises_error(self):
        """测试targets为空列表时调度抛出错误"""
        scheduler = GAScheduler()

        mission_no_targets = Mission(
            name="无目标场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[self.satellite],
            targets=[]
        )
        scheduler.initialize(mission_no_targets)

        with pytest.raises(RuntimeError, match="Scheduler not initialized: no targets available"):
            scheduler.schedule()

    def test_schedule_with_valid_mission_succeeds(self):
        """测试有效mission时调度成功"""
        scheduler = GAScheduler()
        scheduler.initialize(self.mission)

        result = scheduler.schedule()
        assert result is not None


class TestACOSchedulerInitialization:
    """测试ACOScheduler初始化验证"""

    def setup_method(self):
        """设置测试数据"""
        self.satellite = Satellite(
            id="SAT-01",
            name="测试卫星",
            sat_type=SatelliteType.OPTICAL_1
        )
        self.target = Target(
            id="TARGET-01",
            name="测试目标",
            target_type=TargetType.POINT,
            longitude=116.4,
            latitude=39.9,
            priority=5
        )
        self.mission = Mission(
            name="测试场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[self.satellite],
            targets=[self.target]
        )

    def test_schedule_without_initialize_raises_error(self):
        """测试未初始化时调度抛出错误"""
        scheduler = ACOScheduler()

        with pytest.raises(RuntimeError, match="Scheduler not initialized"):
            scheduler.schedule()

    def test_schedule_with_none_mission_raises_error(self):
        """测试mission为None时调度抛出错误"""
        scheduler = ACOScheduler()
        scheduler.mission = None

        with pytest.raises(RuntimeError, match="Scheduler not initialized: mission is None"):
            scheduler.schedule()

    def test_schedule_with_empty_satellites_raises_error(self):
        """测试satellites为空列表时调度抛出错误"""
        scheduler = ACOScheduler()

        mission_no_sats = Mission(
            name="无卫星场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[],
            targets=[self.target]
        )
        scheduler.initialize(mission_no_sats)

        with pytest.raises(RuntimeError, match="Scheduler not initialized: no satellites available"):
            scheduler.schedule()

    def test_schedule_with_empty_targets_raises_error(self):
        """测试targets为空列表时调度抛出错误"""
        scheduler = ACOScheduler()

        mission_no_targets = Mission(
            name="无目标场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[self.satellite],
            targets=[]
        )
        scheduler.initialize(mission_no_targets)

        with pytest.raises(RuntimeError, match="Scheduler not initialized: no targets available"):
            scheduler.schedule()

    def test_schedule_with_valid_mission_succeeds(self):
        """测试有效mission时调度成功"""
        scheduler = ACOScheduler()
        scheduler.initialize(self.mission)

        result = scheduler.schedule()
        assert result is not None


class TestPSOSchedulerInitialization:
    """测试PSOScheduler初始化验证"""

    def setup_method(self):
        """设置测试数据"""
        self.satellite = Satellite(
            id="SAT-01",
            name="测试卫星",
            sat_type=SatelliteType.OPTICAL_1
        )
        self.target = Target(
            id="TARGET-01",
            name="测试目标",
            target_type=TargetType.POINT,
            longitude=116.4,
            latitude=39.9,
            priority=5
        )
        self.mission = Mission(
            name="测试场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[self.satellite],
            targets=[self.target]
        )

    def test_schedule_without_initialize_raises_error(self):
        """测试未初始化时调度抛出错误"""
        scheduler = PSOScheduler()

        with pytest.raises(RuntimeError, match="Scheduler not initialized"):
            scheduler.schedule()

    def test_schedule_with_none_mission_raises_error(self):
        """测试mission为None时调度抛出错误"""
        scheduler = PSOScheduler()
        scheduler.mission = None

        with pytest.raises(RuntimeError, match="Scheduler not initialized: mission is None"):
            scheduler.schedule()

    def test_schedule_with_empty_satellites_raises_error(self):
        """测试satellites为空列表时调度抛出错误"""
        scheduler = PSOScheduler()

        mission_no_sats = Mission(
            name="无卫星场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[],
            targets=[self.target]
        )
        scheduler.initialize(mission_no_sats)

        with pytest.raises(RuntimeError, match="Scheduler not initialized: no satellites available"):
            scheduler.schedule()

    def test_schedule_with_empty_targets_raises_error(self):
        """测试targets为空列表时调度抛出错误"""
        scheduler = PSOScheduler()

        mission_no_targets = Mission(
            name="无目标场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[self.satellite],
            targets=[]
        )
        scheduler.initialize(mission_no_targets)

        with pytest.raises(RuntimeError, match="Scheduler not initialized: no targets available"):
            scheduler.schedule()

    def test_schedule_with_valid_mission_succeeds(self):
        """测试有效mission时调度成功"""
        scheduler = PSOScheduler()
        scheduler.initialize(self.mission)

        result = scheduler.schedule()
        assert result is not None


class TestSASchedulerInitialization:
    """测试SAScheduler初始化验证"""

    def setup_method(self):
        """设置测试数据"""
        self.satellite = Satellite(
            id="SAT-01",
            name="测试卫星",
            sat_type=SatelliteType.OPTICAL_1
        )
        self.target = Target(
            id="TARGET-01",
            name="测试目标",
            target_type=TargetType.POINT,
            longitude=116.4,
            latitude=39.9,
            priority=5
        )
        self.mission = Mission(
            name="测试场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[self.satellite],
            targets=[self.target]
        )

    def test_schedule_without_initialize_raises_error(self):
        """测试未初始化时调度抛出错误"""
        scheduler = SAScheduler()

        with pytest.raises(RuntimeError, match="Scheduler not initialized"):
            scheduler.schedule()

    def test_schedule_with_none_mission_raises_error(self):
        """测试mission为None时调度抛出错误"""
        scheduler = SAScheduler()
        scheduler.mission = None

        with pytest.raises(RuntimeError, match="Scheduler not initialized: mission is None"):
            scheduler.schedule()

    def test_schedule_with_empty_satellites_raises_error(self):
        """测试satellites为空列表时调度抛出错误"""
        scheduler = SAScheduler()

        mission_no_sats = Mission(
            name="无卫星场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[],
            targets=[self.target]
        )
        scheduler.initialize(mission_no_sats)

        with pytest.raises(RuntimeError, match="Scheduler not initialized: no satellites available"):
            scheduler.schedule()

    def test_schedule_with_empty_targets_raises_error(self):
        """测试targets为空列表时调度抛出错误"""
        scheduler = SAScheduler()

        mission_no_targets = Mission(
            name="无目标场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[self.satellite],
            targets=[]
        )
        scheduler.initialize(mission_no_targets)

        with pytest.raises(RuntimeError, match="Scheduler not initialized: no targets available"):
            scheduler.schedule()

    def test_schedule_with_valid_mission_succeeds(self):
        """测试有效mission时调度成功"""
        scheduler = SAScheduler()
        scheduler.initialize(self.mission)

        result = scheduler.schedule()
        assert result is not None


class TestTabuSchedulerInitialization:
    """测试TabuScheduler初始化验证"""

    def setup_method(self):
        """设置测试数据"""
        self.satellite = Satellite(
            id="SAT-01",
            name="测试卫星",
            sat_type=SatelliteType.OPTICAL_1
        )
        self.target = Target(
            id="TARGET-01",
            name="测试目标",
            target_type=TargetType.POINT,
            longitude=116.4,
            latitude=39.9,
            priority=5
        )
        self.mission = Mission(
            name="测试场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[self.satellite],
            targets=[self.target]
        )

    def test_schedule_without_initialize_raises_error(self):
        """测试未初始化时调度抛出错误"""
        scheduler = TabuScheduler()

        with pytest.raises(RuntimeError, match="Scheduler not initialized"):
            scheduler.schedule()

    def test_schedule_with_none_mission_raises_error(self):
        """测试mission为None时调度抛出错误"""
        scheduler = TabuScheduler()
        scheduler.mission = None

        with pytest.raises(RuntimeError, match="Scheduler not initialized: mission is None"):
            scheduler.schedule()

    def test_schedule_with_empty_satellites_raises_error(self):
        """测试satellites为空列表时调度抛出错误"""
        scheduler = TabuScheduler()

        mission_no_sats = Mission(
            name="无卫星场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[],
            targets=[self.target]
        )
        scheduler.initialize(mission_no_sats)

        with pytest.raises(RuntimeError, match="Scheduler not initialized: no satellites available"):
            scheduler.schedule()

    def test_schedule_with_empty_targets_raises_error(self):
        """测试targets为空列表时调度抛出错误"""
        scheduler = TabuScheduler()

        mission_no_targets = Mission(
            name="无目标场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[self.satellite],
            targets=[]
        )
        scheduler.initialize(mission_no_targets)

        with pytest.raises(RuntimeError, match="Scheduler not initialized: no targets available"):
            scheduler.schedule()

    def test_schedule_with_valid_mission_succeeds(self):
        """测试有效mission时调度成功"""
        scheduler = TabuScheduler()
        scheduler.initialize(self.mission)

        result = scheduler.schedule()
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
