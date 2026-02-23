"""
强化学习接口测试

测试第15章设计的RLSchedulerInterface、Observation、Action类
遵循TDD原则：先写测试，再实现代码
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any


class TestObservation:
    """测试Observation观测空间类"""

    def test_observation_creation(self):
        """测试Observation基本创建"""
        from scheduler.rl_interface import Observation

        # 创建基本观测
        obs = Observation(
            task_features=np.array([[1.0, 2.0, 3.0]]),
            satellite_features=np.array([[4.0, 5.0]]),
            visibility_matrix=np.array([[1]]),
            current_time=datetime(2024, 1, 1, 12, 0)
        )

        assert obs.task_features is not None
        assert obs.satellite_features is not None
        assert obs.visibility_matrix is not None
        assert obs.current_time == datetime(2024, 1, 1, 12, 0)

    def test_observation_shapes(self):
        """测试Observation各矩阵形状"""
        from scheduler.rl_interface import Observation

        num_tasks = 10
        num_satellites = 5
        task_feature_dim = 8
        sat_feature_dim = 6

        obs = Observation(
            task_features=np.random.randn(num_tasks, task_feature_dim),
            satellite_features=np.random.randn(num_satellites, sat_feature_dim),
            visibility_matrix=np.random.randint(0, 2, (num_tasks, num_satellites)),
            current_time=datetime(2024, 1, 1, 12, 0)
        )

        assert obs.task_features.shape == (num_tasks, task_feature_dim)
        assert obs.satellite_features.shape == (num_satellites, sat_feature_dim)
        assert obs.visibility_matrix.shape == (num_tasks, num_satellites)

    def test_observation_to_vector(self):
        """测试Observation向量化转换"""
        from scheduler.rl_interface import Observation

        obs = Observation(
            task_features=np.array([[1.0, 2.0], [3.0, 4.0]]),
            satellite_features=np.array([[5.0, 6.0]]),
            visibility_matrix=np.array([[1, 0], [0, 1]]),
            current_time=datetime(2024, 1, 1, 12, 0)
        )

        vector = obs.to_vector()
        assert isinstance(vector, np.ndarray)
        assert len(vector) > 0

    def test_observation_normalization(self):
        """测试观测特征归一化"""
        from scheduler.rl_interface import Observation

        obs = Observation(
            task_features=np.array([[100.0, 200.0], [300.0, 400.0]]),
            satellite_features=np.array([[50.0, 60.0]]),
            visibility_matrix=np.array([[1, 0], [0, 1]]),
            current_time=datetime(2024, 1, 1, 12, 0)
        )

        normalized = obs.normalize()
        assert normalized.task_features is not None
        # 归一化后值应在合理范围内
        assert np.all(normalized.task_features >= -10)
        assert np.all(normalized.task_features <= 10)


class TestAction:
    """测试Action动作空间类"""

    def test_action_creation_schedule(self):
        """测试创建schedule动作"""
        from scheduler.rl_interface import Action, ActionType

        action = Action(
            action_type=ActionType.SCHEDULE,
            task_index=0,
            satellite_index=0
        )

        assert action.action_type == ActionType.SCHEDULE
        assert action.task_index == 0
        assert action.satellite_index == 0

    def test_action_creation_skip(self):
        """测试创建skip动作"""
        from scheduler.rl_interface import Action, ActionType

        action = Action(
            action_type=ActionType.SKIP,
            task_index=5
        )

        assert action.action_type == ActionType.SKIP
        assert action.task_index == 5
        assert action.satellite_index is None

    def test_action_creation_wait(self):
        """测试创建wait动作"""
        from scheduler.rl_interface import Action, ActionType

        action = Action(
            action_type=ActionType.WAIT,
            wait_duration=timedelta(minutes=10)
        )

        assert action.action_type == ActionType.WAIT
        assert action.wait_duration == timedelta(minutes=10)
        assert action.task_index is None

    def test_action_validation(self):
        """测试动作有效性验证"""
        from scheduler.rl_interface import Action, ActionType

        # 有效的schedule动作
        valid_schedule = Action(
            action_type=ActionType.SCHEDULE,
            task_index=0,
            satellite_index=0
        )
        assert valid_schedule.is_valid()

        # 无效的schedule动作（缺少satellite_index）
        invalid_schedule = Action(
            action_type=ActionType.SCHEDULE,
            task_index=0
        )
        assert not invalid_schedule.is_valid()

        # 有效的skip动作
        valid_skip = Action(action_type=ActionType.SKIP, task_index=0)
        assert valid_skip.is_valid()


class TestRLSchedulerInterface:
    """测试RLSchedulerInterface强化学习调度器接口"""

    @pytest.fixture
    def mock_mission(self):
        """创建测试用的mission"""
        from core.models import Mission, Satellite, Target, SatelliteType

        mission = Mission(
            name="test_mission",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0)
        )

        # 添加卫星
        sat1 = Satellite(
            id="sat_001",
            name="Test Satellite 1",
            sat_type=SatelliteType.OPTICAL_1
        )
        sat2 = Satellite(
            id="sat_002",
            name="Test Satellite 2",
            sat_type=SatelliteType.SAR_1
        )
        mission.add_satellite(sat1)
        mission.add_satellite(sat2)

        # 添加目标
        for i in range(5):
            target = Target(
                id=f"target_{i:03d}",
                name=f"Target {i}",
                longitude=100.0 + i * 5,
                latitude=30.0 + i * 2,
                priority=i + 1
            )
            mission.add_target(target)

        return mission

    def test_interface_initialization(self, mock_mission):
        """测试接口初始化"""
        from scheduler.rl_interface import RLSchedulerInterface

        interface = RLSchedulerInterface(mock_mission)

        assert interface.mission == mock_mission
        assert interface.current_time == mock_mission.start_time
        assert len(interface.scheduled_tasks) == 0
        assert len(interface.pending_tasks) == 5

    def test_reset_method(self, mock_mission):
        """测试reset方法"""
        from scheduler.rl_interface import RLSchedulerInterface

        interface = RLSchedulerInterface(mock_mission)

        # 先执行一些操作
        interface.current_time = datetime(2024, 1, 1, 6, 0)
        interface.scheduled_tasks.append("task_1")

        # 重置
        obs = interface.reset()

        assert interface.current_time == mock_mission.start_time
        assert len(interface.scheduled_tasks) == 0
        assert len(interface.pending_tasks) == 5
        assert obs is not None
        assert isinstance(obs.task_features, np.ndarray)

    def test_step_method(self, mock_mission):
        """测试step方法"""
        from scheduler.rl_interface import RLSchedulerInterface, Action, ActionType

        interface = RLSchedulerInterface(mock_mission)
        interface.reset()

        # 执行一个动作
        action = Action(
            action_type=ActionType.SKIP,
            task_index=0
        )

        obs, reward, done, info = interface.step(action)

        assert obs is not None
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_observe_method(self, mock_mission):
        """测试observe方法"""
        from scheduler.rl_interface import RLSchedulerInterface

        interface = RLSchedulerInterface(mock_mission)
        obs = interface.observe()

        assert obs is not None
        assert obs.task_features.shape[0] == len(mock_mission.targets)
        assert obs.satellite_features.shape[0] == len(mock_mission.satellites)
        assert obs.visibility_matrix.shape == (5, 2)

    def test_calculate_reward(self, mock_mission):
        """测试奖励计算"""
        from scheduler.rl_interface import RLSchedulerInterface, Action, ActionType

        interface = RLSchedulerInterface(mock_mission)
        interface.reset()

        # 调度高优先级任务应获得更高奖励
        action_schedule = Action(
            action_type=ActionType.SCHEDULE,
            task_index=0,
            satellite_index=0
        )

        reward = interface.calculate_reward(action_schedule)
        assert isinstance(reward, float)
        assert reward > 0  # 成功调度应获得正奖励

        # 跳过任务应获得较小负奖励
        action_skip = Action(
            action_type=ActionType.SKIP,
            task_index=1
        )
        reward_skip = interface.calculate_reward(action_skip)
        assert reward_skip <= 0  # 跳过任务应为负奖励或零

    def test_get_observation_space(self, mock_mission):
        """测试获取观测空间维度"""
        from scheduler.rl_interface import RLSchedulerInterface

        interface = RLSchedulerInterface(mock_mission)
        space = interface.get_observation_space()

        assert isinstance(space, dict)
        assert 'task_features' in space
        assert 'satellite_features' in space
        assert 'visibility_matrix' in space

    def test_get_action_space(self, mock_mission):
        """测试获取动作空间"""
        from scheduler.rl_interface import RLSchedulerInterface

        interface = RLSchedulerInterface(mock_mission)
        space = interface.get_action_space()

        assert isinstance(space, dict)
        assert 'n_tasks' in space
        assert 'n_satellites' in space
        assert space['n_tasks'] == 5
        assert space['n_satellites'] == 2

    def test_episode_termination(self, mock_mission):
        """测试回合终止条件"""
        from scheduler.rl_interface import RLSchedulerInterface, Action, ActionType

        interface = RLSchedulerInterface(mock_mission)
        interface.reset()

        # 跳过所有任务（总是跳过第一个，因为列表会缩短）
        for i in range(5):
            action = Action(action_type=ActionType.SKIP, task_index=0)
            obs, reward, done, info = interface.step(action)

        # 所有任务处理完后应该结束
        assert done is True

    def test_invalid_action_handling(self, mock_mission):
        """测试无效动作处理"""
        from scheduler.rl_interface import RLSchedulerInterface, Action, ActionType

        interface = RLSchedulerInterface(mock_mission)
        interface.reset()

        # 尝试调度不存在的任务
        action = Action(
            action_type=ActionType.SCHEDULE,
            task_index=100,  # 不存在的索引
            satellite_index=0
        )

        obs, reward, done, info = interface.step(action)

        # 应返回负奖励
        assert reward < 0
        assert 'error' in info


class TestRLSchedulerIntegration:
    """测试RL调度器集成场景"""

    @pytest.fixture
    def simple_mission(self):
        """创建简单测试场景"""
        from core.models import Mission, Satellite, Target, SatelliteType

        mission = Mission(
            name="simple_test",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 1, 12, 0)
        )

        sat = Satellite(
            id="sat_001",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_1
        )
        mission.add_satellite(sat)

        for i in range(3):
            target = Target(
                id=f"target_{i:03d}",
                name=f"Target {i}",
                longitude=100.0 + i * 10,
                latitude=30.0,
                priority=i + 1
            )
            mission.add_target(target)

        return mission

    def test_full_episode(self, simple_mission):
        """测试完整回合"""
        from scheduler.rl_interface import RLSchedulerInterface, Action, ActionType

        interface = RLSchedulerInterface(simple_mission)
        obs = interface.reset()

        total_reward = 0
        done = False
        step_count = 0
        max_steps = 100

        while not done and step_count < max_steps:
            # 简单的随机策略：交替调度和跳过
            if step_count % 2 == 0:
                action = Action(
                    action_type=ActionType.SCHEDULE,
                    task_index=step_count % 3,
                    satellite_index=0
                )
            else:
                action = Action(
                    action_type=ActionType.SKIP,
                    task_index=step_count % 3
                )

            obs, reward, done, info = interface.step(action)
            total_reward += reward
            step_count += 1

        assert step_count > 0
        assert isinstance(total_reward, float)

    def test_state_consistency(self, simple_mission):
        """测试状态一致性"""
        from scheduler.rl_interface import RLSchedulerInterface, Action, ActionType

        interface = RLSchedulerInterface(simple_mission)
        interface.reset()

        # 执行动作
        action = Action(
            action_type=ActionType.SCHEDULE,
            task_index=0,
            satellite_index=0
        )

        obs1, _, _, _ = interface.step(action)

        # 再次observe应得到相同状态
        obs2 = interface.observe()

        np.testing.assert_array_equal(obs1.task_features, obs2.task_features)
        np.testing.assert_array_equal(obs1.satellite_features, obs2.satellite_features)


class TestObservationDesignDocRequirements:
    """测试Observation设计文档第15章要求的功能"""

    def test_observation_with_planning_horizon(self):
        """测试Observation包含planning_horizon字段"""
        from scheduler.rl_interface import Observation

        planning_horizon = datetime(2024, 1, 2, 0, 0)
        obs = Observation(
            task_features=np.array([[1.0, 2.0]]),
            satellite_features=np.array([[3.0, 4.0]]),
            visibility_matrix=np.array([[1]]),
            current_time=datetime(2024, 1, 1, 12, 0),
            planning_horizon=planning_horizon
        )

        assert hasattr(obs, 'planning_horizon')
        assert obs.planning_horizon == planning_horizon

    def test_observation_with_time_progress(self):
        """测试Observation包含time_progress字段（0-1范围）"""
        from scheduler.rl_interface import Observation

        obs = Observation(
            task_features=np.array([[1.0, 2.0]]),
            satellite_features=np.array([[3.0, 4.0]]),
            visibility_matrix=np.array([[1]]),
            current_time=datetime(2024, 1, 1, 12, 0),
            time_progress=0.5
        )

        assert hasattr(obs, 'time_progress')
        assert 0.0 <= obs.time_progress <= 1.0
        assert obs.time_progress == 0.5

    def test_observation_with_valid_action_mask(self):
        """测试Observation包含valid_action_mask字段"""
        from scheduler.rl_interface import Observation

        num_satellites = 2
        num_tasks = 3
        max_windows = 5

        valid_mask = np.ones((num_satellites, num_tasks, max_windows), dtype=np.int32)
        valid_mask[0, 1, 2] = 0  # 某些动作无效

        obs = Observation(
            task_features=np.random.randn(num_tasks, 8),
            satellite_features=np.random.randn(num_satellites, 6),
            visibility_matrix=np.random.randint(0, 2, (num_tasks, num_satellites)),
            current_time=datetime(2024, 1, 1, 12, 0),
            valid_action_mask=valid_mask
        )

        assert hasattr(obs, 'valid_action_mask')
        assert obs.valid_action_mask.shape == (num_satellites, num_tasks, max_windows)
        assert obs.valid_action_mask[0, 1, 2] == 0

    def test_observation_to_dict_method(self):
        """测试Observation的to_dict方法"""
        from scheduler.rl_interface import Observation

        obs = Observation(
            task_features=np.array([[1.0, 2.0], [3.0, 4.0]]),
            satellite_features=np.array([[5.0, 6.0]]),
            visibility_matrix=np.array([[1, 0], [0, 1]]),
            current_time=datetime(2024, 1, 1, 12, 0),
            planning_horizon=datetime(2024, 1, 2, 0, 0),
            time_progress=0.5
        )

        obs_dict = obs.to_dict()

        assert isinstance(obs_dict, dict)
        assert 'task_features' in obs_dict
        assert 'satellite_features' in obs_dict
        assert 'visibility_matrix' in obs_dict
        assert 'current_time' in obs_dict
        assert 'planning_horizon' in obs_dict
        assert 'time_progress' in obs_dict

    def test_observation_to_dict_with_numpy_conversion(self):
        """测试to_dict方法将numpy数组转换为列表"""
        from scheduler.rl_interface import Observation

        obs = Observation(
            task_features=np.array([[1.0, 2.0]]),
            satellite_features=np.array([[3.0, 4.0]]),
            visibility_matrix=np.array([[1]]),
            current_time=datetime(2024, 1, 1, 12, 0)
        )

        obs_dict = obs.to_dict()

        # 验证numpy数组被转换为Python列表
        assert isinstance(obs_dict['task_features'], list)
        assert isinstance(obs_dict['satellite_features'], list)
        assert isinstance(obs_dict['visibility_matrix'], list)

    def test_observation_default_values(self):
        """测试Observation字段的默认值"""
        from scheduler.rl_interface import Observation

        obs = Observation(
            task_features=np.array([[1.0, 2.0]]),
            satellite_features=np.array([[3.0, 4.0]]),
            visibility_matrix=np.array([[1]]),
            current_time=datetime(2024, 1, 1, 12, 0)
        )

        # 验证可选字段有合理的默认值
        assert hasattr(obs, 'planning_horizon')
        assert hasattr(obs, 'time_progress')
        assert hasattr(obs, 'valid_action_mask')


class TestActionDesignDocRequirements:
    """测试Action设计文档第15章要求的功能"""

    def test_action_with_task_id(self):
        """测试Action包含task_id字段"""
        from scheduler.rl_interface import Action, ActionType

        action = Action(
            action_type=ActionType.SCHEDULE,
            task_id="task_001",
            satellite_id="sat_001"
        )

        assert hasattr(action, 'task_id')
        assert action.task_id == "task_001"

    def test_action_with_satellite_id(self):
        """测试Action包含satellite_id字段"""
        from scheduler.rl_interface import Action, ActionType

        action = Action(
            action_type=ActionType.SCHEDULE,
            task_id="task_001",
            satellite_id="sat_001"
        )

        assert hasattr(action, 'satellite_id')
        assert action.satellite_id == "sat_001"

    def test_action_with_window_index(self):
        """测试Action包含window_index字段"""
        from scheduler.rl_interface import Action, ActionType

        action = Action(
            action_type=ActionType.SCHEDULE,
            task_id="task_001",
            satellite_id="sat_001",
            window_index=2
        )

        assert hasattr(action, 'window_index')
        assert action.window_index == 2

    def test_action_with_imaging_mode(self):
        """测试Action包含imaging_mode字段"""
        from scheduler.rl_interface import Action, ActionType

        action = Action(
            action_type=ActionType.SCHEDULE,
            task_id="task_001",
            satellite_id="sat_001",
            imaging_mode="high_resolution"
        )

        assert hasattr(action, 'imaging_mode')
        assert action.imaging_mode == "high_resolution"

    def test_action_with_processing_decision(self):
        """测试Action包含processing_decision字段"""
        from scheduler.rl_interface import Action, ActionType

        action = Action(
            action_type=ActionType.SCHEDULE,
            task_id="task_001",
            satellite_id="sat_001",
            processing_decision="onboard"
        )

        assert hasattr(action, 'processing_decision')
        assert action.processing_decision == "onboard"

    def test_action_processing_decision_valid_values(self):
        """测试processing_decision的有效值"""
        from scheduler.rl_interface import Action, ActionType

        # 测试有效值: onboard, downlink, auto
        for decision in ["onboard", "downlink", "auto"]:
            action = Action(
                action_type=ActionType.SCHEDULE,
                task_id="task_001",
                satellite_id="sat_001",
                processing_decision=decision
            )
            assert action.processing_decision == decision

    def test_action_backward_compatibility_with_index(self):
        """测试Action向后兼容task_index和satellite_index"""
        from scheduler.rl_interface import Action, ActionType

        action = Action(
            action_type=ActionType.SCHEDULE,
            task_index=0,
            satellite_index=1
        )

        assert hasattr(action, 'task_index')
        assert hasattr(action, 'satellite_index')
        assert action.task_index == 0
        assert action.satellite_index == 1


class TestRLSchedulerInterfaceDesignDocRequirements:
    """测试RLSchedulerInterface设计文档第15章要求的功能"""

    @pytest.fixture
    def mock_scenario(self):
        """创建测试用的scenario"""
        from core.models import Mission, Satellite, Target, SatelliteType

        mission = Mission(
            name="test_scenario",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0)
        )

        sat = Satellite(
            id="sat_001",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_1
        )
        mission.add_satellite(sat)

        for i in range(3):
            target = Target(
                id=f"target_{i:03d}",
                name=f"Target {i}",
                longitude=100.0 + i * 10,
                latitude=30.0,
                priority=i + 1
            )
            mission.add_target(target)

        return mission

    def test_reset_with_scenario_parameter(self, mock_scenario):
        """测试reset方法接受scenario参数"""
        from scheduler.rl_interface import RLSchedulerInterface

        interface = RLSchedulerInterface(mock_scenario)

        # reset应接受scenario参数
        obs = interface.reset(scenario=mock_scenario)

        assert obs is not None
        assert isinstance(obs.task_features, np.ndarray)

    def test_reset_without_scenario_uses_existing(self, mock_scenario):
        """测试reset方法无scenario参数时使用现有mission"""
        from scheduler.rl_interface import RLSchedulerInterface

        interface = RLSchedulerInterface(mock_scenario)

        # reset无参数时应使用初始化时的mission
        obs = interface.reset()

        assert obs is not None
        assert interface.mission == mock_scenario

    def test_get_valid_actions_method_exists(self, mock_scenario):
        """测试get_valid_actions方法存在"""
        from scheduler.rl_interface import RLSchedulerInterface

        interface = RLSchedulerInterface(mock_scenario)
        interface.reset()

        assert hasattr(interface, 'get_valid_actions')
        valid_actions = interface.get_valid_actions()

        assert isinstance(valid_actions, list)

    def test_get_valid_actions_returns_action_objects(self, mock_scenario):
        """测试get_valid_actions返回Action对象列表"""
        from scheduler.rl_interface import RLSchedulerInterface, Action

        interface = RLSchedulerInterface(mock_scenario)
        interface.reset()

        valid_actions = interface.get_valid_actions()

        for action in valid_actions:
            assert isinstance(action, Action)

    def test_step_returns_tuple_of_four(self, mock_scenario):
        """测试step方法返回(obs, reward, done, info)四元组"""
        from scheduler.rl_interface import RLSchedulerInterface, Action, ActionType

        interface = RLSchedulerInterface(mock_scenario)
        interface.reset()

        action = Action(
            action_type=ActionType.SKIP,
            task_index=0
        )

        result = interface.step(action)

        assert isinstance(result, tuple)
        assert len(result) == 4

        obs, reward, done, info = result
        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_calculate_reward_signature(self, mock_scenario):
        """测试calculate_reward方法接受正确参数"""
        from scheduler.rl_interface import RLSchedulerInterface, Action, ActionType

        interface = RLSchedulerInterface(mock_scenario)
        interface.reset()

        action = Action(
            action_type=ActionType.SCHEDULE,
            task_index=0,
            satellite_index=0
        )

        # 方法应接受action, prev_state, curr_state参数
        prev_state = interface.observe()
        reward = interface.calculate_reward(action, prev_state, prev_state)

        assert isinstance(reward, float)


class TestRLEdgeCases:
    """测试RL接口的边缘情况"""

    def test_observation_empty_arrays(self):
        """测试Observation处理空数组"""
        from scheduler.rl_interface import Observation

        obs = Observation(
            task_features=np.array([]).reshape(0, 0),
            satellite_features=np.array([]).reshape(0, 0),
            visibility_matrix=np.array([]).reshape(0, 0),
            current_time=datetime(2024, 1, 1, 12, 0)
        )

        # 验证空数组情况下不会崩溃
        assert obs.valid_action_mask is not None
        assert obs.planning_horizon is not None

    def test_observation_to_dict_with_none_values(self):
        """测试to_dict处理None值"""
        from scheduler.rl_interface import Observation

        obs = Observation(
            task_features=np.array([[1.0, 2.0]]),
            satellite_features=np.array([[3.0, 4.0]]),
            visibility_matrix=np.array([[1]]),
            current_time=datetime(2024, 1, 1, 12, 0),
            planning_horizon=None,
            valid_action_mask=None
        )

        obs_dict = obs.to_dict()

        # __post_init__会自动设置默认值，所以planning_horizon不会是None
        assert obs_dict['planning_horizon'] is not None
        # valid_action_mask会被自动设置
        assert obs_dict['valid_action_mask'] is not None

    def test_action_with_task_id_only(self):
        """测试仅使用task_id的Action验证"""
        from scheduler.rl_interface import Action, ActionType

        action = Action(
            action_type=ActionType.SKIP,
            task_id="task_001"
        )

        assert action.is_valid()
        assert action.get_task_identifier() == "task_001"

    def test_action_with_satellite_id_only(self):
        """测试仅使用satellite_id的Action"""
        from scheduler.rl_interface import Action, ActionType

        action = Action(
            action_type=ActionType.SCHEDULE,
            task_id="task_001",
            satellite_id="sat_001"
        )

        assert action.is_valid()
        assert action.get_satellite_identifier() == "sat_001"

    def test_calculate_reward_with_task_id(self):
        """测试使用task_id计算奖励"""
        from core.models import Mission, Satellite, Target, SatelliteType
        from scheduler.rl_interface import RLSchedulerInterface, Action, ActionType

        mission = Mission(
            name="test",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0)
        )

        sat = Satellite(id="sat_001", name="Test", sat_type=SatelliteType.OPTICAL_1)
        mission.add_satellite(sat)

        target = Target(id="target_001", name="Target", longitude=100.0, latitude=30.0, priority=5)
        mission.add_target(target)

        interface = RLSchedulerInterface(mission)
        interface.reset()

        # 使用task_id而非task_index
        action = Action(
            action_type=ActionType.SCHEDULE,
            task_id="target_001",
            satellite_id="sat_001"
        )

        reward = interface.calculate_reward(action)
        assert isinstance(reward, float)
        assert reward > 0

    def test_calculate_reward_invalid_task_id(self):
        """测试使用无效task_id计算奖励"""
        from core.models import Mission, Satellite, Target, SatelliteType
        from scheduler.rl_interface import RLSchedulerInterface, Action, ActionType

        mission = Mission(
            name="test",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0)
        )

        sat = Satellite(id="sat_001", name="Test", sat_type=SatelliteType.OPTICAL_1)
        mission.add_satellite(sat)

        target = Target(id="target_001", name="Target", longitude=100.0, latitude=30.0, priority=5)
        mission.add_target(target)

        interface = RLSchedulerInterface(mission)
        interface.reset()

        # 使用无效的task_id
        action = Action(
            action_type=ActionType.SCHEDULE,
            task_id="non_existent_task",
            satellite_id="sat_001"
        )

        reward = interface.calculate_reward(action)
        assert reward < 0  # 应该返回负奖励

    def test_get_valid_actions_empty_pending(self):
        """测试没有待处理任务时的有效动作"""
        from core.models import Mission, Satellite, Target, SatelliteType
        from scheduler.rl_interface import RLSchedulerInterface

        mission = Mission(
            name="test",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0)
        )

        sat = Satellite(id="sat_001", name="Test", sat_type=SatelliteType.OPTICAL_1)
        mission.add_satellite(sat)

        interface = RLSchedulerInterface(mission)
        interface.reset()

        # 清空待处理任务
        interface.pending_tasks = []
        valid_actions = interface.get_valid_actions()

        # 至少应该有WAIT动作
        assert len(valid_actions) >= 1

    def test_build_valid_action_mask_empty(self):
        """测试构建空的有效动作掩码"""
        from core.models import Mission, Satellite, Target, SatelliteType
        from scheduler.rl_interface import RLSchedulerInterface

        mission = Mission(
            name="test",
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0)
        )

        sat = Satellite(id="sat_001", name="Test", sat_type=SatelliteType.OPTICAL_1)
        mission.add_satellite(sat)

        target = Target(id="target_001", name="Target", longitude=100.0, latitude=30.0, priority=5)
        mission.add_target(target)

        interface = RLSchedulerInterface(mission)
        interface.reset()

        # 清空待处理任务
        interface.pending_tasks = []
        mask = interface._build_valid_action_mask()

        assert mask.shape == (1, 0, 1)  # n_satellites=1, n_pending=0, max_windows=1
