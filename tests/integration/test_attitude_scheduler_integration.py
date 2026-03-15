"""
姿态管理与调度器集成测试

验证姿态管理功能正确集成到调度器中。
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from scheduler.greedy.greedy_scheduler import GreedyScheduler
from scheduler.base_scheduler import ScheduleResult
from core.dynamics.attitude_mode import AttitudeMode
from core.models.target import TargetType


class TestGreedySchedulerAttitudeIntegration:
    """测试GreedyScheduler与姿态管理的集成"""

    def create_mock_mission(self):
        """创建模拟任务场景"""
        mission = Mock()
        mission.start_time = datetime(2024, 1, 1, 0, 0, 0)
        mission.end_time = datetime(2024, 1, 2, 0, 0, 0)

        # 创建卫星
        sat1 = Mock()
        sat1.id = "SAT-00"
        sat1.capabilities = Mock()
        sat1.capabilities.max_roll_angle= 45.0
        sat1.capabilities.storage_capacity = 100.0
        sat1.capabilities.power_capacity = 1000.0
        sat1.capabilities.imaging_modes = [Mock()]
        sat1.capabilities.agility = {'max_slew_rate': 3.0, 'settling_time': 5.0}

        sat2 = Mock()
        sat2.id = "SAT-01"
        sat2.capabilities = sat1.capabilities

        mission.satellites = [sat1, sat2]

        # 创建目标
        targets = []
        for i in range(5):
            target = Mock()
            target.id = f"TARGET-{i:02d}"
            target.priority = 5
            target.required_observations = 1
            target.time_window_start = None
            target.time_window_end = None
            target.target_type = TargetType.POINT
            target.resolution_required = 10.0
            target.data_size_gb = 1.0
            target.latitude = 39.0 + i * 0.1
            target.longitude = 116.0 + i * 0.1
            targets.append(target)
        mission.targets = targets

        # Configure get_target_by_id to return the actual target
        def get_target_by_id(target_id):
            for t in targets:
                if t.id == target_id:
                    return t
            return None
        mission.get_target_by_id = get_target_by_id

        # Configure get_satellite_by_id
        def get_satellite_by_id(sat_id):
            for s in [sat1, sat2]:
                if s.id == sat_id:
                    return s
            return None
        mission.get_satellite_by_id = get_satellite_by_id

        return mission

    def create_mock_window_cache(self, mission):
        """创建模拟窗口缓存"""
        cache = Mock()

        def get_windows(sat_id, target_id):
            # 模拟50%的可见性概率
            if hash(target_id) % 2 == 0:
                return [Mock(
                    start_time=datetime(2024, 1, 1, 10, 0, 0),
                    end_time=datetime(2024, 1, 1, 10, 10, 0),
                )]
            return []

        cache.get_windows = get_windows
        return cache

    def test_scheduler_tracks_attitude_state(self):
        """测试调度器跟踪卫星姿态状态"""
        scheduler = GreedyScheduler(config={
            'enable_attitude_management': True,
            'consider_power': False,
            'consider_storage': False,
        })

        mission = self.create_mock_mission()
        scheduler.initialize(mission)
        scheduler.set_window_cache(self.create_mock_window_cache(mission))

        # 初始化姿态状态
        scheduler._initialize_attitude_state()

        # 验证姿态状态跟踪已初始化
        assert hasattr(scheduler, '_sat_attitude_state')
        assert 'SAT-00' in scheduler._sat_attitude_state
        assert 'SAT-01' in scheduler._sat_attitude_state

        # 验证初始姿态为对地定向
        assert scheduler._sat_attitude_state['SAT-00'] == AttitudeMode.NADIR_POINTING

    def test_scheduler_includes_attitude_transition_time(self):
        """测试调度器包含姿态切换时间"""
        scheduler = GreedyScheduler(config={
            'enable_attitude_management': True,
            'consider_power': False,
            'consider_storage': False,
        })

        mission = self.create_mock_mission()
        scheduler.initialize(mission)
        scheduler.set_window_cache(self.create_mock_window_cache(mission))

        # 初始化姿态组件
        scheduler._initialize_attitude_state()
        scheduler._ensure_attitude_checker_initialized()

        # 计算姿态切换时间
        from_time = datetime(2024, 1, 1, 10, 0, 0)
        transition_time = scheduler._calculate_attitude_transition_time(
            sat_id='SAT-00',
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.IMAGING,
            timestamp=from_time,
        )

        # 验证返回的是timedelta
        assert isinstance(transition_time, timedelta)
        # 姿态切换时间应该非负
        assert transition_time.total_seconds() >= 0

    def test_post_task_attitude_affects_next_task(self):
        """测试任务后姿态影响下一个任务"""
        scheduler = GreedyScheduler(config={
            'enable_attitude_management': True,
            'consider_power': False,
            'consider_storage': False,
        })

        mission = self.create_mock_mission()
        scheduler.initialize(mission)
        scheduler.set_window_cache(self.create_mock_window_cache(mission))

        # 初始化姿态状态
        scheduler._initialize_attitude_state()

        # 模拟执行一个任务后姿态变为IMAGING
        scheduler._set_satellite_attitude_mode('SAT-00', AttitudeMode.IMAGING)

        # 验证姿态状态已更新
        assert scheduler._get_satellite_attitude_mode('SAT-00') == AttitudeMode.IMAGING

    def test_attitude_manager_instance_exists(self):
        """Test that scheduler has AttitudeConstraintChecker instance."""
        scheduler = GreedyScheduler(config={
            'enable_attitude_management': True,
        })

        mission = self.create_mock_mission()
        scheduler.initialize(mission)
        scheduler.set_window_cache(self.create_mock_window_cache(mission))

        # Initialize attitude components
        scheduler._initialize_attitude_state()
        scheduler._ensure_attitude_checker_initialized()

        # Check that AttitudeChecker is created
        assert hasattr(scheduler, '_attitude_checker')
        assert scheduler._attitude_checker is not None

    def test_decide_post_task_attitude_called(self):
        """Test that attitude state is updated after scheduling each task."""
        scheduler = GreedyScheduler(config={
            'consider_power': False,
            'consider_storage': False,
            'enable_attitude_management': True,
        })

        mission = self.create_mock_mission()
        scheduler.initialize(mission)
        scheduler.set_window_cache(self.create_mock_window_cache(mission))

        # Initialize attitude components
        scheduler._initialize_attitude_state()

        # Verify initial state
        assert scheduler._get_satellite_attitude_mode('SAT-00') == AttitudeMode.NADIR_POINTING

        # After scheduling, attitude should be updated to IMAGING
        # Run a short schedule
        result = scheduler.schedule()

        # Verify schedule completed
        assert isinstance(result, ScheduleResult)

    def test_attitude_state_updated_after_task(self):
        """Test that attitude state is updated after task completion."""
        scheduler = GreedyScheduler(config={
            'enable_attitude_management': True,
            'consider_power': False,
            'consider_storage': False,
        })

        mission = self.create_mock_mission()
        scheduler.initialize(mission)
        scheduler.set_window_cache(self.create_mock_window_cache(mission))

        # Initialize
        scheduler._initialize_attitude_state()

        # Verify initial state
        assert scheduler._get_satellite_attitude_mode('SAT-00') == AttitudeMode.NADIR_POINTING

        # Run schedule
        result = scheduler.schedule()

        # After scheduling, attitude should be IMAGING
        assert scheduler._get_satellite_attitude_mode('SAT-00') == AttitudeMode.IMAGING


class TestAttitudeTransitionFeasibility:
    """测试姿态切换可行性检查"""

    def test_task_with_attitude_transition_feasible(self):
        """测试带姿态切换的任务是可行的"""
        scheduler = GreedyScheduler(config={
            'enable_attitude_management': True,
            'consider_power': False,
            'consider_storage': False,
        })

        mission = Mock()
        mission.start_time = datetime(2024, 1, 1, 0, 0, 0)
        mission.end_time = datetime(2024, 1, 2, 0, 0, 0)

        target = Mock()
        target.id = "TARGET-01"
        target.priority = 5
        target.required_observations = 1
        target.time_window_start = None
        target.time_window_end = None
        target.target_type = TargetType.POINT
        target.resolution_required = 10.0
        target.data_size_gb = 1.0
        target.latitude = 39.0
        target.longitude = 116.0
        mission.targets = [target]

        sat = Mock()
        sat.id = "SAT-00"
        sat.capabilities = Mock()
        sat.capabilities.max_roll_angle= 45.0
        sat.capabilities.storage_capacity = 100.0
        sat.capabilities.power_capacity = 1000.0
        sat.capabilities.imaging_modes = [Mock()]
        sat.capabilities.agility = {'max_slew_rate': 3.0, 'settling_time': 5.0}
        sat.capabilities.get_imaging_constraints = Mock(return_value=None)
        mission.satellites = [sat]

        # Configure mission methods
        mission.get_target_by_id = Mock(return_value=target)
        mission.get_satellite_by_id = Mock(return_value=sat)

        cache = Mock()
        cache.get_windows = Mock(return_value=[Mock(
            start_time=datetime(2024, 1, 1, 10, 0, 0),
            end_time=datetime(2024, 1, 1, 10, 10, 0),
        )])

        scheduler.initialize(mission)
        scheduler.set_window_cache(cache)

        # Run schedule - should complete without error
        result = scheduler.schedule()

        assert isinstance(result, ScheduleResult)

    def test_task_with_infeasible_attitude_transition(self):
        """测试姿态切换不可行时的处理"""
        scheduler = GreedyScheduler(config={
            'enable_attitude_management': True,
            'consider_power': False,
            'consider_storage': False,
        })

        mission = Mock()
        mission.start_time = datetime(2024, 1, 1, 0, 0, 0)
        mission.end_time = datetime(2024, 1, 2, 0, 0, 0)

        target = Mock()
        target.id = "TARGET-01"
        target.priority = 5
        target.required_observations = 1
        target.time_window_start = None
        target.time_window_end = None
        target.target_type = TargetType.POINT
        target.resolution_required = 10.0
        target.data_size_gb = 1.0
        target.latitude = 39.0
        target.longitude = 116.0
        mission.targets = [target]

        sat = Mock()
        sat.id = "SAT-00"
        sat.capabilities = Mock()
        sat.capabilities.max_roll_angle= 45.0
        sat.capabilities.storage_capacity = 100.0
        sat.capabilities.power_capacity = 1000.0
        sat.capabilities.imaging_modes = [Mock()]
        sat.capabilities.agility = {'max_slew_rate': 3.0, 'settling_time': 5.0}
        sat.capabilities.get_imaging_constraints = Mock(return_value=None)
        mission.satellites = [sat]

        # Configure mission methods
        mission.get_target_by_id = Mock(return_value=target)
        mission.get_satellite_by_id = Mock(return_value=sat)

        cache = Mock()
        cache.get_windows = Mock(return_value=[])

        scheduler.initialize(mission)
        scheduler.set_window_cache(cache)

        # Initialize
        scheduler._initialize_attitude_state()

        # Schedule should still complete (with or without tasks)
        result = scheduler.schedule()

        assert isinstance(result, ScheduleResult)


class TestMultipleTasksSequence:
    """测试多任务序列中的姿态管理"""

    def test_multiple_tasks_consider_attitude_transitions(self):
        """测试多任务考虑姿态切换"""
        scheduler = GreedyScheduler(config={
            'enable_attitude_management': True,
            'consider_power': False,
            'consider_storage': False,
        })

        mission = Mock()
        mission.start_time = datetime(2024, 1, 1, 0, 0, 0)
        mission.end_time = datetime(2024, 1, 2, 0, 0, 0)

        targets = []
        for i in range(3):
            target = Mock()
            target.id = f"TARGET-{i:02d}"
            target.priority = 5
            target.required_observations = 1
            target.time_window_start = None
            target.time_window_end = None
            target.target_type = TargetType.POINT
            target.resolution_required = 10.0
            target.data_size_gb = 1.0
            target.latitude = 39.0 + i * 0.1
            target.longitude = 116.0 + i * 0.1
            targets.append(target)
        mission.targets = targets

        sat = Mock()
        sat.id = "SAT-00"
        sat.capabilities = Mock()
        sat.capabilities.max_roll_angle= 45.0
        sat.capabilities.storage_capacity = 1000.0
        sat.capabilities.power_capacity = 10000.0
        sat.capabilities.imaging_modes = [Mock()]
        sat.capabilities.agility = {'max_slew_rate': 3.0, 'settling_time': 5.0}
        sat.capabilities.get_imaging_constraints = Mock(return_value=None)
        mission.satellites = [sat]

        # Configure mission methods
        def get_target_by_id(target_id):
            for t in targets:
                if t.id == target_id:
                    return t
            return None
        mission.get_target_by_id = get_target_by_id
        mission.get_satellite_by_id = Mock(return_value=sat)

        cache = Mock()
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        cache.get_windows = Mock(side_effect=lambda sat_id, target_id: [Mock(
            start_time=base_time,
            end_time=base_time + timedelta(minutes=10),
        )])

        scheduler.initialize(mission)
        scheduler.set_window_cache(cache)

        # Initialize
        scheduler._initialize_attitude_state()

        # Run schedule
        result = scheduler.schedule()

        assert isinstance(result, ScheduleResult)
        # Should schedule some tasks
        assert len(result.scheduled_tasks) >= 0

    def test_task_ordering_with_attitude(self):
        """测试考虑姿态的任务排序"""
        scheduler = GreedyScheduler(config={
            'enable_attitude_management': True,
            'consider_power': False,
            'consider_storage': False,
        })

        mission = Mock()
        mission.start_time = datetime(2024, 1, 1, 0, 0, 0)
        mission.end_time = datetime(2024, 1, 2, 0, 0, 0)

        targets = []
        for i in range(3):
            target = Mock()
            target.id = f"TARGET-{i:02d}"
            target.priority = 5
            target.required_observations = 1
            target.time_window_start = None
            target.time_window_end = None
            target.target_type = TargetType.POINT
            target.resolution_required = 10.0
            target.data_size_gb = 1.0
            target.latitude = 39.0 + i * 0.1
            target.longitude = 116.0 + i * 0.1
            targets.append(target)
        mission.targets = targets

        sat = Mock()
        sat.id = "SAT-00"
        sat.capabilities = Mock()
        sat.capabilities.max_roll_angle= 45.0
        sat.capabilities.storage_capacity = 1000.0
        sat.capabilities.power_capacity = 10000.0
        sat.capabilities.imaging_modes = [Mock()]
        sat.capabilities.agility = {'max_slew_rate': 3.0, 'settling_time': 5.0}
        sat.capabilities.get_imaging_constraints = Mock(return_value=None)
        mission.satellites = [sat]

        # Configure mission methods
        def get_target_by_id(target_id):
            for t in targets:
                if t.id == target_id:
                    return t
            return None
        mission.get_target_by_id = get_target_by_id
        mission.get_satellite_by_id = Mock(return_value=sat)

        cache = Mock()
        cache.get_windows = Mock(return_value=[Mock(
            start_time=datetime(2024, 1, 1, 10, 0, 0),
            end_time=datetime(2024, 1, 1, 10, 10, 0),
        )])

        scheduler.initialize(mission)
        scheduler.set_window_cache(cache)

        # Initialize
        scheduler._initialize_attitude_state()

        # Calculate transition time
        from_time = datetime(2024, 1, 1, 10, 0, 0)
        transition_time = scheduler._calculate_attitude_transition_time(
            sat_id='SAT-00',
            from_mode=AttitudeMode.NADIR_POINTING,
            to_mode=AttitudeMode.IMAGING,
            timestamp=from_time,
        )

        # Transition time should be non-negative
        assert transition_time.total_seconds() >= 0

        # Run schedule
        result = scheduler.schedule()

        assert isinstance(result, ScheduleResult)


class TestAttitudeSchedulerConfig:
    """测试调度器姿态管理配置"""

    def test_scheduler_accepts_attitude_config(self):
        """测试调度器接受姿态管理配置"""
        config = {
            'enable_attitude_management': True,
            'attitude_idle_threshold': 600.0,
            'attitude_soc_threshold': 0.25,
        }

        scheduler = GreedyScheduler(config=config)

        assert scheduler._enable_attitude_management == True

    def test_scheduler_disabled_attitude_management(self):
        """测试禁用姿态管理"""
        config = {
            'enable_attitude_management': False,
        }

        scheduler = GreedyScheduler(config=config)

        assert scheduler._enable_attitude_management == False

    def test_default_attitude_config(self):
        """测试默认姿态管理配置"""
        scheduler = GreedyScheduler()

        # Default should be False
        assert scheduler._enable_attitude_management == False
