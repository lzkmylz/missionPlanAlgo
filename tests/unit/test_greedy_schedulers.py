"""
贪心调度器完整测试

TDD测试文件 - 第8章设计实现
包含：
- EDD（最早截止时间优先）调度算法测试
- SPT（最短处理时间优先）调度算法测试

测试覆盖：
1. 基础功能测试
2. 边界条件测试
3. 资源约束测试
4. 失败原因分析测试
5. 调度所有任务（非仅前10个）
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from core.models import Mission, Satellite, SatelliteType, Target, TargetType, ImagingMode
from scheduler.greedy.edd_scheduler import EDDScheduler
from scheduler.greedy.spt_scheduler import SPTScheduler
from scheduler.base_scheduler import ScheduleResult, TaskFailureReason, ScheduledTask


class TestEDDScheduler:
    """测试EDD（最早截止时间优先）调度器"""

    def setup_method(self):
        """每个测试方法前设置"""
        self.satellite = Satellite(
            id="SAT-01",
            name="测试卫星",
            sat_type=SatelliteType.OPTICAL_1
        )

        self.target1 = Target(
            id="TARGET-01",
            name="目标1",
            target_type=TargetType.POINT,
            longitude=116.4,
            latitude=39.9,
            priority=5,
            time_window_end=datetime(2024, 1, 1, 12, 0)
        )
        self.target2 = Target(
            id="TARGET-02",
            name="目标2",
            target_type=TargetType.POINT,
            longitude=121.5,
            latitude=31.2,
            priority=5,
            time_window_end=datetime(2024, 1, 1, 10, 0)  # 更早截止
        )
        self.target3 = Target(
            id="TARGET-03",
            name="目标3",
            target_type=TargetType.POINT,
            longitude=113.3,
            latitude=23.1,
            priority=5,
            time_window_end=datetime(2024, 1, 1, 14, 0)
        )

        self.mission = Mission(
            name="测试场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[self.satellite],
            targets=[self.target1, self.target2, self.target3]
        )

    def test_scheduler_initialization(self):
        """测试调度器初始化"""
        scheduler = EDDScheduler()
        assert scheduler.name == "EDD"
        assert scheduler.heuristic == "due_date"
        assert scheduler.consider_power is True
        assert scheduler.consider_storage is True
        assert scheduler.allow_tardiness is False

    def test_scheduler_initialization_with_config(self):
        """测试带配置的调度器初始化"""
        config = {
            'consider_power': False,
            'consider_storage': False,
            'allow_tardiness': True
        }
        scheduler = EDDScheduler(config)
        assert scheduler.consider_power is False
        assert scheduler.consider_storage is False
        assert scheduler.allow_tardiness is True

    def test_edd_sorts_by_due_date(self):
        """测试EDD按截止时间排序"""
        scheduler = EDDScheduler()
        scheduler.initialize(self.mission)

        sorted_tasks = scheduler._sort_tasks_by_due_date(self.mission.targets)

        # 应该按截止时间排序：target2(10:00) -> target1(12:00) -> target3(14:00)
        assert sorted_tasks[0].id == "TARGET-02"
        assert sorted_tasks[1].id == "TARGET-01"
        assert sorted_tasks[2].id == "TARGET-03"

    def test_edd_sorts_tasks_without_due_date_last(self):
        """测试无截止时间的任务排在最后"""
        target_no_due = Target(
            id="TARGET-NO-DUE",
            name="无截止时间目标",
            target_type=TargetType.POINT,
            longitude=100.0,
            latitude=30.0,
            priority=10  # 高优先级
        )

        mission = Mission(
            name="测试场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[self.satellite],
            targets=[target_no_due, self.target1]  # 无截止时间的先添加
        )

        scheduler = EDDScheduler()
        scheduler.initialize(mission)

        sorted_tasks = scheduler._sort_tasks_by_due_date(mission.targets)

        # 有截止时间的应该排在前面
        assert sorted_tasks[0].id == "TARGET-01"
        assert sorted_tasks[1].id == "TARGET-NO-DUE"

    def test_ties_broken_by_priority(self):
        """测试截止时间相同时按优先级打破平局"""
        target_high = Target(
            id="TARGET-HIGH",
            name="高优先级",
            target_type=TargetType.POINT,
            longitude=116.4,
            latitude=39.9,
            priority=9,
            time_window_end=datetime(2024, 1, 1, 12, 0)
        )
        target_low = Target(
            id="TARGET-LOW",
            name="低优先级",
            target_type=TargetType.POINT,
            longitude=121.5,
            latitude=31.2,
            priority=3,
            time_window_end=datetime(2024, 1, 1, 12, 0)
        )

        mission = Mission(
            name="优先级测试",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[self.satellite],
            targets=[target_low, target_high]
        )

        scheduler = EDDScheduler()
        scheduler.initialize(mission)

        sorted_tasks = scheduler._sort_tasks_by_due_date(mission.targets)

        # 截止时间相同时，高优先级应该排在前面
        assert sorted_tasks[0].id == "TARGET-HIGH"
        assert sorted_tasks[1].id == "TARGET-LOW"

    def test_schedule_returns_result(self):
        """测试调度返回结果对象"""
        scheduler = EDDScheduler()
        scheduler.initialize(self.mission)

        result = scheduler.schedule()

        assert isinstance(result, ScheduleResult)
        assert hasattr(result, 'scheduled_tasks')
        assert hasattr(result, 'unscheduled_tasks')
        assert hasattr(result, 'makespan')
        assert hasattr(result, 'computation_time')
        assert hasattr(result, 'iterations')
        assert hasattr(result, 'convergence_curve')

    def test_schedule_without_initialize_raises_error(self):
        """测试未初始化时调度抛出错误"""
        scheduler = EDDScheduler()

        with pytest.raises(RuntimeError, match="Scheduler not initialized"):
            scheduler.schedule()

    def test_empty_mission_raises_error(self):
        """测试空场景（无卫星、无目标）应抛出初始化错误"""
        empty_mission = Mission(
            name="空场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[],
            targets=[]
        )

        scheduler = EDDScheduler()
        scheduler.initialize(empty_mission)

        with pytest.raises(RuntimeError, match="Scheduler not initialized: no satellites available"):
            scheduler.schedule()

    def test_get_parameters(self):
        """测试获取参数配置"""
        scheduler = EDDScheduler(config={'consider_power': False})
        params = scheduler.get_parameters()

        assert 'heuristic' in params
        assert 'consider_power' in params
        assert 'consider_storage' in params
        assert 'allow_tardiness' in params
        assert params['heuristic'] == 'due_date'

    def test_can_satellite_perform_task(self):
        """测试卫星能力检查"""
        scheduler = EDDScheduler()
        scheduler.initialize(self.mission)

        # 有成像能力的卫星应该能执行任务
        assert scheduler._can_satellite_perform_task(self.satellite, self.target1) is True

        # 移除成像能力
        self.satellite.capabilities.imaging_modes = []
        assert scheduler._can_satellite_perform_task(self.satellite, self.target1) is False


class TestEDDFailureReasons:
    """测试EDD调度器的失败原因追踪"""

    def setup_method(self):
        """设置资源受限的场景"""
        self.satellite = Satellite(
            id="SAT-01",
            name="小存储卫星",
            sat_type=SatelliteType.OPTICAL_1
        )
        self.satellite.capabilities.storage_capacity = 1.0  # 只有1GB

        self.targets = [
            Target(
                id=f"TARGET-{i:02d}",
                name=f"目标{i}",
                target_type=TargetType.POINT,
                longitude=116.0 + i,
                latitude=39.0,
                priority=5,
                time_window_end=datetime(2024, 1, 1, 10 + i, 0)
            )
            for i in range(5)
        ]

        self.mission = Mission(
            name="资源受限场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[self.satellite],
            targets=self.targets
        )

    def test_determine_failure_reason_no_window(self):
        """测试确定失败原因：无可见窗口"""
        scheduler = EDDScheduler()
        scheduler.initialize(self.mission)

        # 模拟资源使用情况 - 资源充足
        sat_resource_usage = {
            "SAT-01": {
                'power': 1000.0,
                'storage': 0.0
            }
        }

        reason = scheduler._determine_failure_reason(self.targets[0], sat_resource_usage)
        # 资源充足时，失败原因应该是无可见窗口
        assert reason == TaskFailureReason.NO_VISIBLE_WINDOW

    def test_determine_failure_reason_power_constraint(self):
        """测试确定失败原因：电量约束"""
        scheduler = EDDScheduler()
        scheduler.initialize(self.mission)

        # 模拟资源使用情况 - 电量不足
        sat_resource_usage = {
            "SAT-01": {
                'power': 50.0,  # 低电量（低于10%阈值）
                'storage': 0.0
            }
        }

        reason = scheduler._determine_failure_reason(self.targets[0], sat_resource_usage)
        assert reason == TaskFailureReason.POWER_CONSTRAINT


class TestSPTScheduler:
    """测试SPT（最短处理时间优先）调度器"""

    def setup_method(self):
        """每个测试方法前设置"""
        self.satellite = Satellite(
            id="SAT-01",
            name="测试卫星",
            sat_type=SatelliteType.OPTICAL_1
        )

        # 点目标 - 处理时间短
        self.point_target = Target(
            id="POINT-01",
            name="点目标",
            target_type=TargetType.POINT,
            longitude=116.4,
            latitude=39.9,
            priority=5
        )

        # 区域目标 - 处理时间长
        self.area_target = Target(
            id="AREA-01",
            name="区域目标",
            target_type=TargetType.AREA,
            area_vertices=[(116.0, 39.0), (117.0, 39.0), (117.0, 40.0), (116.0, 40.0)],
            priority=5
        )

        self.mission = Mission(
            name="测试场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[self.satellite],
            targets=[self.area_target, self.point_target]
        )

    def test_scheduler_initialization(self):
        """测试SPT调度器初始化"""
        scheduler = SPTScheduler()
        assert scheduler.name == "SPT"
        assert scheduler.heuristic == "processing_time"
        assert scheduler.consider_power is True
        assert scheduler.consider_storage is True

    def test_scheduler_initialization_with_config(self):
        """测试带配置的SPT调度器初始化"""
        config = {
            'consider_power': False,
            'consider_storage': False,
            'allow_tardiness': True
        }
        scheduler = SPTScheduler(config)
        assert scheduler.consider_power is False
        assert scheduler.consider_storage is False
        assert scheduler.allow_tardiness is True

    def test_spt_sorts_by_processing_time(self):
        """测试SPT按处理时间排序"""
        scheduler = SPTScheduler()
        scheduler.initialize(self.mission)

        sorted_tasks = scheduler._sort_tasks_by_processing_time(self.mission.targets)

        # 点目标应该排在区域目标前面（处理时间更短）
        assert sorted_tasks[0].id == "POINT-01"
        assert sorted_tasks[1].id == "AREA-01"

    def test_spt_ties_broken_by_priority(self):
        """测试SPT处理时间相同时按优先级排序"""
        # 创建两个点目标，处理时间相同但优先级不同
        target_high = Target(
            id="HIGH-PRIORITY",
            name="高优先级",
            target_type=TargetType.POINT,
            longitude=116.4,
            latitude=39.9,
            priority=9
        )
        target_low = Target(
            id="LOW-PRIORITY",
            name="低优先级",
            target_type=TargetType.POINT,
            longitude=121.5,
            latitude=31.2,
            priority=3
        )

        mission = Mission(
            name="优先级测试",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[self.satellite],
            targets=[target_low, target_high]
        )

        scheduler = SPTScheduler()
        scheduler.initialize(mission)

        sorted_tasks = scheduler._sort_tasks_by_processing_time(mission.targets)

        # 处理时间相同时，高优先级应该排在前面
        assert sorted_tasks[0].id == "HIGH-PRIORITY"
        assert sorted_tasks[1].id == "LOW-PRIORITY"

    def test_estimate_processing_time(self):
        """测试处理时间估计"""
        scheduler = SPTScheduler()
        scheduler.initialize(self.mission)

        point_time = scheduler._estimate_processing_time(self.point_target)
        area_time = scheduler._estimate_processing_time(self.area_target)

        # 区域目标的处理时间应该比点目标长
        assert area_time > point_time

    def test_schedule_returns_result(self):
        """测试SPT调度返回结果对象"""
        scheduler = SPTScheduler()
        scheduler.initialize(self.mission)

        result = scheduler.schedule()

        assert isinstance(result, ScheduleResult)
        assert hasattr(result, 'scheduled_tasks')
        assert hasattr(result, 'unscheduled_tasks')
        assert hasattr(result, 'makespan')
        assert hasattr(result, 'computation_time')

    def test_schedule_without_initialize_raises_error(self):
        """测试SPT未初始化时调度抛出错误"""
        scheduler = SPTScheduler()

        with pytest.raises(RuntimeError, match="Scheduler not initialized"):
            scheduler.schedule()

    def test_empty_mission_raises_error(self):
        """测试SPT空场景（无卫星、无目标）应抛出初始化错误"""
        empty_mission = Mission(
            name="空场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[],
            targets=[]
        )

        scheduler = SPTScheduler()
        scheduler.initialize(empty_mission)

        with pytest.raises(RuntimeError, match="Scheduler not initialized: no satellites available"):
            scheduler.schedule()

    def test_get_parameters(self):
        """测试SPT获取参数配置"""
        scheduler = SPTScheduler(config={'consider_storage': False})
        params = scheduler.get_parameters()

        assert 'heuristic' in params
        assert 'consider_power' in params
        assert 'consider_storage' in params
        assert 'allow_tardiness' in params
        assert params['heuristic'] == 'processing_time'


class TestSPTFailureReasons:
    """测试SPT调度器的失败原因追踪"""

    def setup_method(self):
        """设置资源受限的场景"""
        self.satellite = Satellite(
            id="SAT-01",
            name="低电量卫星",
            sat_type=SatelliteType.OPTICAL_1
        )
        self.satellite.capabilities.power_capacity = 100.0  # 低电量

        self.targets = [
            Target(
                id=f"TARGET-{i:02d}",
                name=f"目标{i}",
                target_type=TargetType.POINT,
                longitude=116.0 + i,
                latitude=39.0,
                priority=5
            )
            for i in range(5)
        ]

        self.mission = Mission(
            name="资源受限场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[self.satellite],
            targets=self.targets
        )

    def test_determine_failure_reason_no_window(self):
        """测试SPT确定失败原因：无可见窗口"""
        scheduler = SPTScheduler()
        scheduler.initialize(self.mission)

        sat_resource_usage = {
            "SAT-01": {
                'power': 1000.0,
                'storage': 0.0
            }
        }

        reason = scheduler._determine_failure_reason(self.targets[0], sat_resource_usage)
        assert reason == TaskFailureReason.NO_VISIBLE_WINDOW

    def test_determine_failure_reason_storage(self):
        """测试SPT确定失败原因：存储约束"""
        scheduler = SPTScheduler()
        scheduler.initialize(self.mission)

        # 模拟存储已满
        sat_resource_usage = {
            "SAT-01": {
                'power': 1000.0,
                'storage': 499.0  # 接近容量上限
            }
        }

        # 添加data_size_gb属性
        self.targets[0].data_size_gb = 2.0

        reason = scheduler._determine_failure_reason(self.targets[0], sat_resource_usage)
        assert reason == TaskFailureReason.STORAGE_CONSTRAINT


class TestSPTResourceConstraints:
    """测试SPT调度器的资源约束"""

    def setup_method(self):
        self.satellite = Satellite(
            id="SAT-01",
            name="测试卫星",
            sat_type=SatelliteType.OPTICAL_1
        )
        self.satellite.capabilities.power_capacity = 1000.0
        self.satellite.capabilities.storage_capacity = 100.0

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

    def test_spt_power_constraint_check(self):
        """测试SPT电量约束检查"""
        from unittest.mock import MagicMock
        scheduler = SPTScheduler(config={'consider_power': True})
        scheduler.initialize(self.mission)

        sat_resource_usage = {
            "SAT-01": {
                'power': 1000.0,
                'storage': 0.0
            }
        }

        window = MagicMock()
        window.start_time = datetime(2024, 1, 1, 6, 0)
        window.end_time = datetime(2024, 1, 1, 6, 10)

        result = scheduler._check_resource_constraints(
            self.satellite, self.target, window, sat_resource_usage
        )
        assert result is True

    def test_spt_storage_constraint_violation(self):
        """测试SPT存储约束违反"""
        from unittest.mock import MagicMock
        scheduler = SPTScheduler(config={'consider_storage': True})
        scheduler.initialize(self.mission)

        sat_resource_usage = {
            "SAT-01": {
                'power': 1000.0,
                'storage': 95.0
            }
        }

        self.target.data_size_gb = 10.0

        window = MagicMock()
        window.start_time = datetime(2024, 1, 1, 6, 0)
        window.end_time = datetime(2024, 1, 1, 6, 10)

        result = scheduler._check_resource_constraints(
            self.satellite, self.target, window, sat_resource_usage
        )
        assert result is False

    def test_spt_can_satellite_perform_task(self):
        """测试SPT卫星能力检查"""
        scheduler = SPTScheduler()
        scheduler.initialize(self.mission)

        assert scheduler._can_satellite_perform_task(self.satellite, self.target) is True

        self.satellite.capabilities.imaging_modes = []
        assert scheduler._can_satellite_perform_task(self.satellite, self.target) is False


class TestSPTWithWindows:
    """测试SPT带窗口的调度"""

    def setup_method(self):
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

    def test_spt_find_best_assignment_with_mock_window(self):
        """测试SPT带模拟窗口的最佳分配查找"""
        scheduler = SPTScheduler()
        scheduler.initialize(self.mission)

        mock_window = {
            'start': datetime(2024, 1, 1, 6, 0),
            'end': datetime(2024, 1, 1, 6, 10)
        }

        mock_cache = MagicMock()
        mock_cache.get_windows.return_value = [mock_window]
        scheduler.set_window_cache(mock_cache)

        sat_resource_usage = {
            "SAT-01": {
                'power': 1000.0,
                'storage': 0.0,
                'last_task_end': self.mission.start_time
            }
        }

        result = scheduler._find_best_assignment(self.target, sat_resource_usage)

        assert result is not None
        sat_id, window, imaging_mode = result
        assert sat_id == "SAT-01"

    def test_spt_find_best_assignment_no_windows(self):
        """测试SPT无可用窗口时的最佳分配查找"""
        scheduler = SPTScheduler()
        scheduler.initialize(self.mission)

        mock_cache = MagicMock()
        mock_cache.get_windows.return_value = []
        scheduler.set_window_cache(mock_cache)

        sat_resource_usage = {
            "SAT-01": {
                'power': 1000.0,
                'storage': 0.0,
                'last_task_end': self.mission.start_time
            }
        }

        result = scheduler._find_best_assignment(self.target, sat_resource_usage)
        assert result is None

    def test_spt_deadline_constraint_with_allow_tardiness(self):
        """测试SPT允许延迟时的截止时间约束"""
        scheduler = SPTScheduler(config={'allow_tardiness': True})
        scheduler.initialize(self.mission)

        past_target = Target(
            id="PAST-TARGET",
            name="已过期目标",
            target_type=TargetType.POINT,
            longitude=116.0,
            latitude=39.0,
            priority=5,
            time_window_end=datetime(2020, 1, 1, 0, 0)
        )

        future_window = {
            'start': datetime(2024, 1, 1, 6, 0),
            'end': datetime(2024, 1, 1, 6, 10)
        }

        mock_cache = MagicMock()
        mock_cache.get_windows.return_value = [future_window]
        scheduler.set_window_cache(mock_cache)

        sat_resource_usage = {
            "SAT-01": {
                'power': 1000.0,
                'storage': 0.0,
                'last_task_end': self.mission.start_time
            }
        }

        result = scheduler._find_best_assignment(past_target, sat_resource_usage)
        assert result is not None


class TestSPTResourceUpdate:
    """测试SPT调度器资源更新"""

    def setup_method(self):
        self.satellite = Satellite(
            id="SAT-01",
            name="测试卫星",
            sat_type=SatelliteType.OPTICAL_1
        )
        self.satellite.capabilities.power_capacity = 1000.0
        self.satellite.capabilities.storage_capacity = 100.0

        self.target = Target(
            id="TARGET-01",
            name="测试目标",
            target_type=TargetType.POINT,
            longitude=116.4,
            latitude=39.9,
            priority=5
        )
        self.target.data_size_gb = 5.0

        self.mission = Mission(
            name="测试场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[self.satellite],
            targets=[self.target]
        )

    def test_spt_update_resource_usage(self):
        """测试SPT资源使用更新"""
        scheduler = SPTScheduler()
        scheduler.initialize(self.mission)

        initial_power = 1000.0
        initial_storage = 0.0

        sat_resource_usage = {
            "SAT-01": {
                'power': initial_power,
                'storage': initial_storage,
                'last_task_end': self.mission.start_time
            }
        }

        window = {
            'start': datetime(2024, 1, 1, 6, 0),
            'end': datetime(2024, 1, 1, 6, 10)
        }

        scheduler._update_resource_usage("SAT-01", self.target, window, sat_resource_usage)

        assert sat_resource_usage["SAT-01"]['power'] < initial_power
        assert sat_resource_usage["SAT-01"]['storage'] == initial_storage + 5.0
        assert sat_resource_usage["SAT-01"]['last_task_end'] == window['end']

    def test_spt_update_resource_usage_power_disabled(self):
        """测试SPT禁用电量更新"""
        scheduler = SPTScheduler(config={'consider_power': False})
        scheduler.initialize(self.mission)

        initial_power = 1000.0

        sat_resource_usage = {
            "SAT-01": {
                'power': initial_power,
                'storage': 0.0,
                'last_task_end': self.mission.start_time
            }
        }

        window = {
            'start': datetime(2024, 1, 1, 6, 0),
            'end': datetime(2024, 1, 1, 6, 10)
        }

        scheduler._update_resource_usage("SAT-01", self.target, window, sat_resource_usage)

        assert sat_resource_usage["SAT-01"]['power'] == initial_power

    def test_spt_update_resource_usage_storage_disabled(self):
        """测试SPT禁用存储更新"""
        scheduler = SPTScheduler(config={'consider_storage': False})
        scheduler.initialize(self.mission)

        initial_storage = 0.0

        sat_resource_usage = {
            "SAT-01": {
                'power': 1000.0,
                'storage': initial_storage,
                'last_task_end': self.mission.start_time
            }
        }

        window = {
            'start': datetime(2024, 1, 1, 6, 0),
            'end': datetime(2024, 1, 1, 6, 10)
        }

        scheduler._update_resource_usage("SAT-01", self.target, window, sat_resource_usage)

        assert sat_resource_usage["SAT-01"]['storage'] == initial_storage


class TestSPTSelectImagingMode:
    """测试SPT成像模式选择"""

    def setup_method(self):
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

    def test_spt_select_imaging_mode_with_modes(self):
        """测试SPT有可用模式时的成像模式选择"""
        from core.models import ImagingMode
        scheduler = SPTScheduler()
        scheduler.initialize(self.mission)

        mode = scheduler._select_imaging_mode(self.satellite, self.target)

        assert mode is not None
        assert isinstance(mode, ImagingMode)

    def test_spt_select_imaging_mode_no_modes(self):
        """测试SPT无可用模式时的成像模式选择"""
        from core.models import ImagingMode
        scheduler = SPTScheduler()
        scheduler.initialize(self.mission)

        self.satellite.capabilities.imaging_modes = []

        mode = scheduler._select_imaging_mode(self.satellite, self.target)

        assert mode == ImagingMode.PUSH_BROOM


class TestResourceConstraints:
    """测试资源约束检查"""

    def setup_method(self):
        self.satellite = Satellite(
            id="SAT-01",
            name="测试卫星",
            sat_type=SatelliteType.OPTICAL_1
        )
        self.satellite.capabilities.power_capacity = 1000.0
        self.satellite.capabilities.storage_capacity = 100.0

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

    def test_power_constraint_check(self):
        """测试电量约束检查"""
        from unittest.mock import MagicMock
        scheduler = EDDScheduler(config={'consider_power': True})
        scheduler.initialize(self.mission)

        # 模拟资源使用情况 - 电量充足
        sat_resource_usage = {
            "SAT-01": {
                'power': 1000.0,
                'storage': 0.0
            }
        }

        # 创建一个模拟窗口对象
        window = MagicMock()
        window.start_time = datetime(2024, 1, 1, 6, 0)
        window.end_time = datetime(2024, 1, 1, 6, 10)

        result = scheduler._check_resource_constraints(
            self.satellite, self.target, window, sat_resource_usage
        )
        assert result is True

    def test_power_constraint_violation(self):
        """测试电量约束违反"""
        from unittest.mock import MagicMock
        scheduler = EDDScheduler(config={'consider_power': True})
        scheduler.initialize(self.mission)

        # 模拟资源使用情况 - 电量不足
        sat_resource_usage = {
            "SAT-01": {
                'power': 0.0,  # 无电量
                'storage': 0.0
            }
        }

        window = MagicMock()
        window.start_time = datetime(2024, 1, 1, 6, 0)
        window.end_time = datetime(2024, 1, 1, 6, 10)

        result = scheduler._check_resource_constraints(
            self.satellite, self.target, window, sat_resource_usage
        )
        assert result is False

    def test_storage_constraint_check(self):
        """测试存储约束检查"""
        from unittest.mock import MagicMock
        scheduler = EDDScheduler(config={'consider_storage': True})
        scheduler.initialize(self.mission)

        # 模拟资源使用情况 - 存储充足
        sat_resource_usage = {
            "SAT-01": {
                'power': 1000.0,
                'storage': 50.0  # 已使用50GB，还剩50GB
            }
        }

        # 设置任务数据大小
        self.target.data_size_gb = 10.0

        window = MagicMock()
        window.start_time = datetime(2024, 1, 1, 6, 0)
        window.end_time = datetime(2024, 1, 1, 6, 10)

        result = scheduler._check_resource_constraints(
            self.satellite, self.target, window, sat_resource_usage
        )
        assert result is True

    def test_storage_constraint_violation(self):
        """测试存储约束违反"""
        from unittest.mock import MagicMock
        scheduler = EDDScheduler(config={'consider_storage': True})
        scheduler.initialize(self.mission)

        # 模拟资源使用情况 - 存储已满
        sat_resource_usage = {
            "SAT-01": {
                'power': 1000.0,
                'storage': 95.0  # 已使用95GB，只剩5GB
            }
        }

        # 设置任务数据大小超过剩余空间
        self.target.data_size_gb = 10.0

        window = MagicMock()
        window.start_time = datetime(2024, 1, 1, 6, 0)
        window.end_time = datetime(2024, 1, 1, 6, 10)

        result = scheduler._check_resource_constraints(
            self.satellite, self.target, window, sat_resource_usage
        )
        assert result is False

    def test_resource_constraints_disabled(self):
        """测试禁用资源约束检查"""
        scheduler = EDDScheduler(config={
            'consider_power': False,
            'consider_storage': False
        })
        scheduler.initialize(self.mission)

        # 即使资源不足，也应该通过
        sat_resource_usage = {
            "SAT-01": {
                'power': 0.0,
                'storage': 100.0  # 已满
            }
        }

        self.target.data_size_gb = 10.0

        window = {
            'start': datetime(2024, 1, 1, 6, 0),
            'end': datetime(2024, 1, 1, 6, 10)
        }

        result = scheduler._check_resource_constraints(
            self.satellite, self.target, window, sat_resource_usage
        )
        assert result is True


class TestSchedulerSchedulesAllTasks:
    """测试调度器处理所有任务，而非仅前10个"""

    def test_edd_schedules_all_20_tasks(self):
        """测试EDD调度所有20个任务"""
        satellite = Satellite(
            id="SAT-01",
            name="测试卫星",
            sat_type=SatelliteType.OPTICAL_1
        )

        # 创建20个任务
        targets = [
            Target(
                id=f"TARGET-{i:02d}",
                name=f"目标{i}",
                target_type=TargetType.POINT,
                longitude=116.0 + i * 0.1,
                latitude=39.0,
                priority=5,
                time_window_end=datetime(2024, 1, 1, 6 + i % 18, 0)  # 确保小时在0-23范围内
            )
            for i in range(20)
        ]

        mission = Mission(
            name="20任务场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[satellite],
            targets=targets
        )

        scheduler = EDDScheduler()
        scheduler.initialize(mission)

        # 验证任务数量
        assert len(mission.targets) == 20

        # 验证排序处理所有任务
        sorted_tasks = scheduler._sort_tasks_by_due_date(mission.targets)
        assert len(sorted_tasks) == 20

        # 验证按截止时间正确排序（早的在前）
        for i in range(len(sorted_tasks) - 1):
            if (sorted_tasks[i].time_window_end and sorted_tasks[i+1].time_window_end):
                assert sorted_tasks[i].time_window_end <= sorted_tasks[i+1].time_window_end

    def test_spt_schedules_all_20_tasks(self):
        """测试SPT调度所有20个任务"""
        satellite = Satellite(
            id="SAT-01",
            name="测试卫星",
            sat_type=SatelliteType.OPTICAL_1
        )

        # 创建20个任务
        targets = [
            Target(
                id=f"TARGET-{i:02d}",
                name=f"目标{i}",
                target_type=TargetType.POINT,
                longitude=116.0 + i * 0.1,
                latitude=39.0,
                priority=5
            )
            for i in range(20)
        ]

        mission = Mission(
            name="20任务场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[satellite],
            targets=targets
        )

        scheduler = SPTScheduler()
        scheduler.initialize(mission)

        # 验证任务数量
        assert len(mission.targets) == 20

        # 验证排序处理所有任务
        sorted_tasks = scheduler._sort_tasks_by_processing_time(mission.targets)
        assert len(sorted_tasks) == 20


class TestSchedulerWithWindows:
    """测试带可见窗口的调度"""

    def setup_method(self):
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

    def test_find_best_assignment_with_mock_window(self):
        """测试带模拟窗口的最佳分配查找"""
        scheduler = EDDScheduler()
        scheduler.initialize(self.mission)

        # 创建模拟窗口缓存
        mock_window = {
            'start': datetime(2024, 1, 1, 6, 0),
            'end': datetime(2024, 1, 1, 6, 10)
        }

        mock_cache = MagicMock()
        mock_cache.get_windows.return_value = [mock_window]
        scheduler.set_window_cache(mock_cache)

        sat_resource_usage = {
            "SAT-01": {
                'power': 1000.0,
                'storage': 0.0,
                'last_task_end': self.mission.start_time
            }
        }

        result = scheduler._find_best_assignment(self.target, sat_resource_usage)

        # 应该找到分配
        assert result is not None
        sat_id, window, imaging_mode = result
        assert sat_id == "SAT-01"

    def test_find_best_assignment_no_windows(self):
        """测试无可用窗口时的最佳分配查找"""
        scheduler = EDDScheduler()
        scheduler.initialize(self.mission)

        # 创建空窗口缓存
        mock_cache = MagicMock()
        mock_cache.get_windows.return_value = []
        scheduler.set_window_cache(mock_cache)

        sat_resource_usage = {
            "SAT-01": {
                'power': 1000.0,
                'storage': 0.0,
                'last_task_end': self.mission.start_time
            }
        }

        result = scheduler._find_best_assignment(self.target, sat_resource_usage)

        # 应该返回None
        assert result is None

    def test_deadline_constraint_with_allow_tardiness(self):
        """测试允许延迟时的截止时间约束"""
        scheduler = EDDScheduler(config={'allow_tardiness': True})
        scheduler.initialize(self.mission)

        # 创建一个已过截止时间的任务
        past_target = Target(
            id="PAST-TARGET",
            name="已过期目标",
            target_type=TargetType.POINT,
            longitude=116.0,
            latitude=39.0,
            priority=5,
            time_window_end=datetime(2020, 1, 1, 0, 0)
        )

        # 创建一个在未来时间的窗口
        future_window = {
            'start': datetime(2024, 1, 1, 6, 0),  # 晚于截止时间
            'end': datetime(2024, 1, 1, 6, 10)
        }

        mock_cache = MagicMock()
        mock_cache.get_windows.return_value = [future_window]
        scheduler.set_window_cache(mock_cache)

        sat_resource_usage = {
            "SAT-01": {
                'power': 1000.0,
                'storage': 0.0,
                'last_task_end': self.mission.start_time
            }
        }

        # 允许延迟时，应该能找到分配
        result = scheduler._find_best_assignment(past_target, sat_resource_usage)
        assert result is not None

    def test_deadline_constraint_without_allow_tardiness(self):
        """测试不允许延迟时的截止时间约束"""
        scheduler = EDDScheduler(config={'allow_tardiness': False})
        scheduler.initialize(self.mission)

        # 创建一个已过截止时间的任务
        past_target = Target(
            id="PAST-TARGET",
            name="已过期目标",
            target_type=TargetType.POINT,
            longitude=116.0,
            latitude=39.0,
            priority=5,
            time_window_end=datetime(2020, 1, 1, 0, 0)
        )

        # 创建一个在未来时间的窗口
        future_window = {
            'start': datetime(2024, 1, 1, 6, 0),  # 晚于截止时间
            'end': datetime(2024, 1, 1, 6, 10)
        }

        mock_cache = MagicMock()
        mock_cache.get_windows.return_value = [future_window]
        scheduler.set_window_cache(mock_cache)

        sat_resource_usage = {
            "SAT-01": {
                'power': 1000.0,
                'storage': 0.0,
                'last_task_end': self.mission.start_time
            }
        }

        # 不允许延迟时，应该找不到分配
        result = scheduler._find_best_assignment(past_target, sat_resource_usage)
        assert result is None


class TestSchedulerResourceUpdate:
    """测试调度器资源更新"""

    def setup_method(self):
        self.satellite = Satellite(
            id="SAT-01",
            name="测试卫星",
            sat_type=SatelliteType.OPTICAL_1
        )
        self.satellite.capabilities.power_capacity = 1000.0
        self.satellite.capabilities.storage_capacity = 100.0

        self.target = Target(
            id="TARGET-01",
            name="测试目标",
            target_type=TargetType.POINT,
            longitude=116.4,
            latitude=39.9,
            priority=5
        )
        self.target.data_size_gb = 5.0

        self.mission = Mission(
            name="测试场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[self.satellite],
            targets=[self.target]
        )

    def test_update_resource_usage(self):
        """测试资源使用更新"""
        scheduler = EDDScheduler()
        scheduler.initialize(self.mission)

        initial_power = 1000.0
        initial_storage = 0.0

        sat_resource_usage = {
            "SAT-01": {
                'power': initial_power,
                'storage': initial_storage,
                'last_task_end': self.mission.start_time
            }
        }

        # 使用字典格式的窗口（与EDD调度器实现一致）
        window = {
            'start': datetime(2024, 1, 1, 6, 0),
            'end': datetime(2024, 1, 1, 6, 10)
        }

        scheduler._update_resource_usage("SAT-01", self.target, window, sat_resource_usage)

        # 验证资源被更新
        assert sat_resource_usage["SAT-01"]['power'] < initial_power
        assert sat_resource_usage["SAT-01"]['storage'] == initial_storage + 5.0
        assert sat_resource_usage["SAT-01"]['last_task_end'] == window['end']

    def test_update_resource_usage_power_disabled(self):
        """测试禁用电量更新"""
        scheduler = EDDScheduler(config={'consider_power': False})
        scheduler.initialize(self.mission)

        initial_power = 1000.0

        sat_resource_usage = {
            "SAT-01": {
                'power': initial_power,
                'storage': 0.0,
                'last_task_end': self.mission.start_time
            }
        }

        window = {
            'start': datetime(2024, 1, 1, 6, 0),
            'end': datetime(2024, 1, 1, 6, 10)
        }

        scheduler._update_resource_usage("SAT-01", self.target, window, sat_resource_usage)

        # 电量不应该被更新
        assert sat_resource_usage["SAT-01"]['power'] == initial_power

    def test_update_resource_usage_storage_disabled(self):
        """测试禁用存储更新"""
        scheduler = EDDScheduler(config={'consider_storage': False})
        scheduler.initialize(self.mission)

        initial_storage = 0.0

        sat_resource_usage = {
            "SAT-01": {
                'power': 1000.0,
                'storage': initial_storage,
                'last_task_end': self.mission.start_time
            }
        }

        window = {
            'start': datetime(2024, 1, 1, 6, 0),
            'end': datetime(2024, 1, 1, 6, 10)
        }

        scheduler._update_resource_usage("SAT-01", self.target, window, sat_resource_usage)

        # 存储不应该被更新
        assert sat_resource_usage["SAT-01"]['storage'] == initial_storage


class TestSchedulerSelectImagingMode:
    """测试成像模式选择"""

    def setup_method(self):
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

    def test_select_imaging_mode_with_modes(self):
        """测试有可用模式时的成像模式选择"""
        from core.models import ImagingMode
        scheduler = EDDScheduler()
        scheduler.initialize(self.mission)

        mode = scheduler._select_imaging_mode(self.satellite, self.target)

        # 应该返回一个有效的成像模式（ImagingMode枚举）
        assert mode is not None
        assert isinstance(mode, ImagingMode)

    def test_select_imaging_mode_no_modes(self):
        """测试无可用模式时的成像模式选择"""
        from core.models import ImagingMode
        scheduler = EDDScheduler()
        scheduler.initialize(self.mission)

        # 移除所有成像模式
        self.satellite.capabilities.imaging_modes = []

        mode = scheduler._select_imaging_mode(self.satellite, self.target)

        # 应该返回默认模式
        assert mode == ImagingMode.PUSH_BROOM


class TestSchedulerIntegration:
    """集成测试 - 测试完整调度流程"""

    def setup_method(self):
        """设置多卫星多任务场景"""
        self.sat1 = Satellite(
            id="SAT-01",
            name="卫星1",
            sat_type=SatelliteType.OPTICAL_1
        )
        self.sat2 = Satellite(
            id="SAT-02",
            name="卫星2",
            sat_type=SatelliteType.SAR_1
        )

        self.targets = [
            Target(
                id=f"TARGET-{i:02d}",
                name=f"目标{i}",
                target_type=TargetType.POINT,
                longitude=116.0 + i * 2,
                latitude=39.0 + i * 0.5,
                priority=i % 10 + 1,
                time_window_end=datetime(2024, 1, 1, 8 + i, 0)
            )
            for i in range(5)
        ]

        self.mission = Mission(
            name="集成测试场景",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0),
            satellites=[self.sat1, self.sat2],
            targets=self.targets
        )

    def test_edd_integration(self):
        """测试EDD完整调度流程"""
        scheduler = EDDScheduler()
        scheduler.initialize(self.mission)

        result = scheduler.schedule()

        # 验证结果结构
        assert isinstance(result, ScheduleResult)
        assert isinstance(result.scheduled_tasks, list)
        assert isinstance(result.unscheduled_tasks, dict)
        assert result.makespan >= 0.0
        assert result.computation_time >= 0.0

    def test_spt_integration(self):
        """测试SPT完整调度流程"""
        scheduler = SPTScheduler()
        scheduler.initialize(self.mission)

        result = scheduler.schedule()

        # 验证结果结构
        assert isinstance(result, ScheduleResult)
        assert isinstance(result.scheduled_tasks, list)
        assert isinstance(result.unscheduled_tasks, dict)
        assert result.makespan >= 0.0
        assert result.computation_time >= 0.0

    def test_both_schedulers_produce_valid_results(self):
        """测试两个调度器都产生有效结果"""
        edd_scheduler = EDDScheduler()
        edd_scheduler.initialize(self.mission)
        edd_result = edd_scheduler.schedule()

        spt_scheduler = SPTScheduler()
        spt_scheduler.initialize(self.mission)
        spt_result = spt_scheduler.schedule()

        # 两个结果都应该是有效的
        assert isinstance(edd_result, ScheduleResult)
        assert isinstance(spt_result, ScheduleResult)

        # 已调度任务数 + 未调度任务数应该等于总任务数
        edd_total = len(edd_result.scheduled_tasks) + len(edd_result.unscheduled_tasks)
        spt_total = len(spt_result.scheduled_tasks) + len(spt_result.unscheduled_tasks)

        assert edd_total == len(self.mission.targets)
        assert spt_total == len(self.mission.targets)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
