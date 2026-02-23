"""
强化学习调度器接口

实现第15章设计：
- Observation类 - 观测空间
- Action类 - 动作空间
- RLSchedulerInterface - 强化学习调度器接口
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np


class ActionType(Enum):
    """动作类型枚举"""
    SCHEDULE = "schedule"  # 调度任务
    SKIP = "skip"          # 跳过任务
    WAIT = "wait"          # 等待


@dataclass
class Observation:
    """
    观测空间类

    包含任务特征、卫星特征和可见性矩阵
    实现第15章设计文档要求
    """
    task_features: np.ndarray           # 任务特征矩阵 (n_tasks, task_feature_dim)
    satellite_features: np.ndarray      # 卫星特征矩阵 (n_satellites, sat_feature_dim)
    visibility_matrix: np.ndarray       # 可见性矩阵 (n_tasks, n_satellites)
    current_time: datetime              # 当前时间
    planning_horizon: Optional[datetime] = None  # 规划时间范围
    time_progress: float = 0.0          # 时间进度 (0-1)
    valid_action_mask: Optional[np.ndarray] = None  # 有效动作掩码 [n_satellites, n_tasks, max_windows]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后处理，设置默认值"""
        if self.planning_horizon is None:
            # 默认规划范围为当前时间后24小时
            self.planning_horizon = self.current_time + timedelta(hours=24)
        if self.valid_action_mask is None:
            # 如果没有提供有效动作掩码，默认所有动作都有效
            n_tasks = self.task_features.shape[0] if len(self.task_features.shape) > 0 else 0
            n_satellites = self.satellite_features.shape[0] if len(self.satellite_features.shape) > 0 else 0
            if n_tasks > 0 and n_satellites > 0:
                self.valid_action_mask = np.ones((n_satellites, n_tasks, 1), dtype=np.int32)
            else:
                self.valid_action_mask = np.array([])

    def to_vector(self) -> np.ndarray:
        """将观测转换为向量表示"""
        # 展平所有特征
        task_flat = self.task_features.flatten()
        sat_flat = self.satellite_features.flatten()
        vis_flat = self.visibility_matrix.flatten()

        return np.concatenate([task_flat, sat_flat, vis_flat])

    def normalize(self) -> 'Observation':
        """归一化观测特征"""
        # 对任务特征进行z-score归一化
        task_mean = np.mean(self.task_features, axis=0)
        task_std = np.std(self.task_features, axis=0)
        task_std[task_std == 0] = 1.0  # 避免除零
        normalized_tasks = (self.task_features - task_mean) / task_std

        # 对卫星特征进行z-score归一化
        sat_mean = np.mean(self.satellite_features, axis=0)
        sat_std = np.std(self.satellite_features, axis=0)
        sat_std[sat_std == 0] = 1.0
        normalized_sats = (self.satellite_features - sat_mean) / sat_std

        return Observation(
            task_features=normalized_tasks,
            satellite_features=normalized_sats,
            visibility_matrix=self.visibility_matrix.copy(),
            current_time=self.current_time,
            planning_horizon=self.planning_horizon,
            time_progress=self.time_progress,
            valid_action_mask=self.valid_action_mask.copy() if self.valid_action_mask is not None else None,
            metadata=self.metadata.copy()
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        将观测转换为字典表示

        Returns:
            包含所有观测字段的字典，numpy数组被转换为列表
        """
        def convert_to_list(obj):
            """递归转换numpy数组为列表"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        return {
            'task_features': convert_to_list(self.task_features),
            'satellite_features': convert_to_list(self.satellite_features),
            'visibility_matrix': convert_to_list(self.visibility_matrix),
            'current_time': self.current_time.isoformat() if self.current_time else None,
            'planning_horizon': self.planning_horizon.isoformat() if self.planning_horizon else None,
            'time_progress': float(self.time_progress),
            'valid_action_mask': convert_to_list(self.valid_action_mask) if self.valid_action_mask is not None else None,
            'metadata': self.metadata
        }


@dataclass
class Action:
    """
    动作空间类

    定义强化学习代理可以执行的动作
    实现第15章设计文档要求
    """
    action_type: ActionType
    # 原有字段（保持向后兼容）
    task_index: Optional[int] = None
    satellite_index: Optional[int] = None
    wait_duration: Optional[timedelta] = None
    # 第15章设计文档新增字段
    task_id: Optional[str] = None
    satellite_id: Optional[str] = None
    window_index: int = 0
    imaging_mode: Optional[str] = None
    processing_decision: Optional[str] = None  # 'onboard', 'downlink', 'auto'

    def is_valid(self) -> bool:
        """验证动作是否有效"""
        if self.action_type == ActionType.SCHEDULE:
            # 支持通过索引或ID指定任务和卫星
            has_task = self.task_index is not None or self.task_id is not None
            has_satellite = self.satellite_index is not None or self.satellite_id is not None
            return has_task and has_satellite
        elif self.action_type == ActionType.SKIP:
            return self.task_index is not None or self.task_id is not None
        elif self.action_type == ActionType.WAIT:
            return self.wait_duration is not None
        return False

    def get_task_identifier(self) -> Optional[Any]:
        """获取任务标识符（task_id优先于task_index）"""
        return self.task_id if self.task_id is not None else self.task_index

    def get_satellite_identifier(self) -> Optional[Any]:
        """获取卫星标识符（satellite_id优先于satellite_index）"""
        return self.satellite_id if self.satellite_id is not None else self.satellite_index


class RLSchedulerInterface:
    """
    强化学习调度器接口

    提供与强化学习代理交互的标准接口：
    - reset: 重置环境
    - step: 执行动作
    - observe: 获取当前观测
    - calculate_reward: 计算奖励
    """

    # 任务特征维度
    TASK_FEATURE_DIM = 8
    # 卫星特征维度
    SAT_FEATURE_DIM = 6

    def __init__(self, mission: Any, config: Dict[str, Any] = None):
        """
        初始化RL调度器接口

        Args:
            mission: 任务场景
            config: 配置参数
        """
        self.mission = mission
        self.config = config or {}

        # 状态跟踪
        self.current_time: datetime = mission.start_time
        self.scheduled_tasks: List[str] = []
        self.pending_tasks: List[str] = [t.id for t in mission.targets]
        self.skipped_tasks: List[str] = []

        # 跟踪每个卫星的已分配任务
        self.satellite_assignments: Dict[str, List[str]] = {
            sat.id: [] for sat in mission.satellites
        }

        # 回合状态
        self._episode_done = False
        self._step_count = 0
        self._max_steps = self.config.get('max_steps', 1000)

    def reset(self, scenario: Optional[Any] = None) -> Observation:
        """
        重置环境到初始状态

        Args:
            scenario: 可选的新场景，如果提供则替换当前mission

        Returns:
            初始观测
        """
        # 如果提供了新场景，则更新mission
        if scenario is not None:
            self.mission = scenario

        self.current_time = self.mission.start_time
        self.scheduled_tasks = []
        self.pending_tasks = [t.id for t in self.mission.targets]
        self.skipped_tasks = []
        self.satellite_assignments = {
            sat.id: [] for sat in self.mission.satellites
        }
        self._episode_done = False
        self._step_count = 0

        return self.observe()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        执行一个动作

        Args:
            action: 要执行的动作

        Returns:
            (观测, 奖励, 是否结束, 信息)
        """
        self._step_count += 1
        info = {'action_valid': action.is_valid()}

        if not action.is_valid():
            reward = -10.0  # 无效动作的惩罚
            info['error'] = 'Invalid action'
        else:
            # 检查任务索引有效性
            if action.task_index is not None and action.task_index >= len(self.pending_tasks):
                reward = -5.0  # 无效任务索引的惩罚
                info['error'] = f'Task index {action.task_index} out of range'
            else:
                reward = self.calculate_reward(action)

                if action.action_type == ActionType.SCHEDULE:
                    task_id = self.pending_tasks[action.task_index]
                    self.scheduled_tasks.append(task_id)
                    self.pending_tasks.pop(action.task_index)

                    sat_id = self.mission.satellites[action.satellite_index].id
                    self.satellite_assignments[sat_id].append(task_id)

                elif action.action_type == ActionType.SKIP:
                    task_id = self.pending_tasks[action.task_index]
                    self.skipped_tasks.append(task_id)
                    self.pending_tasks.pop(action.task_index)

                elif action.action_type == ActionType.WAIT:
                    self.current_time += action.wait_duration

        # 检查回合是否结束
        self._episode_done = (
            len(self.pending_tasks) == 0 or
            self._step_count >= self._max_steps
        )

        obs = self.observe()
        return obs, reward, self._episode_done, info

    def observe(self) -> Observation:
        """
        获取当前观测

        Returns:
            当前观测状态
        """
        # 构建任务特征
        task_features = self._build_task_features()

        # 构建卫星特征
        satellite_features = self._build_satellite_features()

        # 构建可见性矩阵（简化实现）
        visibility_matrix = self._build_visibility_matrix()

        # 计算时间进度
        total_duration = (self.mission.end_time - self.mission.start_time).total_seconds()
        elapsed = (self.current_time - self.mission.start_time).total_seconds()
        time_progress = elapsed / total_duration if total_duration > 0 else 0.0

        # 构建有效动作掩码
        valid_action_mask = self._build_valid_action_mask()

        return Observation(
            task_features=task_features,
            satellite_features=satellite_features,
            visibility_matrix=visibility_matrix,
            current_time=self.current_time,
            planning_horizon=self.mission.end_time,
            time_progress=max(0.0, min(1.0, time_progress)),
            valid_action_mask=valid_action_mask,
            metadata={
                'scheduled_count': len(self.scheduled_tasks),
                'pending_count': len(self.pending_tasks),
                'step_count': self._step_count
            }
        )

    def calculate_reward(self, action: Action, prev_state: Optional[Observation] = None, curr_state: Optional[Observation] = None) -> float:
        """
        计算动作的奖励

        Args:
            action: 执行的动作
            prev_state: 执行动作前的状态（可选，用于设计文档兼容性）
            curr_state: 执行动作后的状态（可选，用于设计文档兼容性）

        Returns:
            奖励值
        """
        if not action.is_valid():
            return -10.0

        if action.action_type == ActionType.SKIP:
            return -1.0  # 跳过任务的轻微惩罚

        elif action.action_type == ActionType.WAIT:
            return -0.1  # 等待的小惩罚

        elif action.action_type == ActionType.SCHEDULE:
            # 支持通过task_index或task_id获取任务
            task = None
            if action.task_index is not None:
                if action.task_index >= len(self.pending_tasks):
                    return -5.0  # 无效任务索引
                task_id = self.pending_tasks[action.task_index]
                task = self.mission.get_target_by_id(task_id)
            elif action.task_id is not None:
                task = self.mission.get_target_by_id(action.task_id)
                if task and task.id not in self.pending_tasks:
                    return -5.0  # 任务不在待处理列表中

            if task is None:
                return -5.0

            # 基于优先级计算奖励
            base_reward = task.priority * 10.0

            # 检查卫星能力匹配
            sat_index = action.satellite_index
            if sat_index is None and action.satellite_id is not None:
                # 通过satellite_id查找索引
                for i, sat in enumerate(self.mission.satellites):
                    if sat.id == action.satellite_id:
                        sat_index = i
                        break

            if sat_index is not None and sat_index < len(self.mission.satellites):
                satellite = self.mission.satellites[sat_index]
                # 如果卫星类型匹配目标分辨率要求，给予额外奖励
                if satellite.capabilities.resolution <= task.resolution_required:
                    base_reward *= 1.5

            return base_reward

        return 0.0

    def get_observation_space(self) -> Dict[str, Any]:
        """
        获取观测空间定义

        Returns:
            观测空间维度信息
        """
        n_tasks = len(self.mission.targets)
        n_satellites = len(self.mission.satellites)

        return {
            'task_features': (n_tasks, self.TASK_FEATURE_DIM),
            'satellite_features': (n_satellites, self.SAT_FEATURE_DIM),
            'visibility_matrix': (n_tasks, n_satellites),
            'total_dim': (n_tasks * self.TASK_FEATURE_DIM +
                         n_satellites * self.SAT_FEATURE_DIM +
                         n_tasks * n_satellites)
        }

    def get_action_space(self) -> Dict[str, Any]:
        """
        获取动作空间定义

        Returns:
            动作空间信息
        """
        return {
            'n_tasks': len(self.mission.targets),
            'n_satellites': len(self.mission.satellites),
            'action_types': [t.value for t in ActionType]
        }

    def get_valid_actions(self) -> List[Action]:
        """
        获取当前有效的动作列表

        Returns:
            Action对象列表
        """
        valid_actions = []

        # 为每个待处理任务生成可能的动作
        for task_idx, task_id in enumerate(self.pending_tasks):
            task = self.mission.get_target_by_id(task_id)
            if not task:
                continue

            # SKIP动作对所有待处理任务都有效
            valid_actions.append(Action(
                action_type=ActionType.SKIP,
                task_index=task_idx,
                task_id=task_id
            ))

            # SCHEDULE动作需要检查卫星匹配
            for sat_idx, sat in enumerate(self.mission.satellites):
                # 简化的有效性检查：假设所有卫星都可以调度
                valid_actions.append(Action(
                    action_type=ActionType.SCHEDULE,
                    task_index=task_idx,
                    satellite_index=sat_idx,
                    task_id=task_id,
                    satellite_id=sat.id,
                    window_index=0
                ))

        # WAIT动作总是有效
        valid_actions.append(Action(
            action_type=ActionType.WAIT,
            wait_duration=timedelta(minutes=10)
        ))

        return valid_actions

    def _build_valid_action_mask(self) -> np.ndarray:
        """
        构建有效动作掩码

        Returns:
            三维数组 [n_satellites, n_tasks, max_windows] 表示哪些动作是有效的
        """
        n_pending = len(self.pending_tasks)
        n_satellites = len(self.mission.satellites)
        max_windows = 1  # 简化：每个任务-卫星对只有一个窗口

        if n_pending == 0 or n_satellites == 0:
            return np.zeros((n_satellites, n_pending, max_windows), dtype=np.int32)

        # 默认所有动作都有效
        mask = np.ones((n_satellites, n_pending, max_windows), dtype=np.int32)

        # 可以在这里添加更复杂的约束检查
        # 例如：检查卫星能力是否匹配任务要求

        return mask

    def _build_task_features(self) -> np.ndarray:
        """构建任务特征矩阵"""
        features = []

        for task_id in self.pending_tasks:
            task = self.mission.get_target_by_id(task_id)
            if task:
                feature = [
                    task.priority / 10.0,  # 归一化优先级
                    task.resolution_required / 10.0,  # 分辨率需求
                    1.0 if task.immediate_downlink else 0.0,  # 是否需要立即回传
                    task.required_observations,
                    task.completed_observations,
                    task.longitude if task.longitude else 0.0,
                    task.latitude if task.latitude else 0.0,
                    task.get_area() / 1000.0 if task.target_type.value == 'area' else 0.0
                ]
                features.append(feature)

        if not features:
            # 如果没有待处理任务，返回零矩阵
            return np.zeros((0, self.TASK_FEATURE_DIM))

        return np.array(features)

    def _build_satellite_features(self) -> np.ndarray:
        """构建卫星特征矩阵"""
        features = []

        for sat in self.mission.satellites:
            feature = [
                sat.capabilities.storage_capacity / 2000.0,  # 存储容量
                sat.capabilities.power_capacity / 5000.0,     # 能源容量
                sat.capabilities.resolution / 10.0,           # 分辨率
                sat.capabilities.max_off_nadir / 50.0,        # 最大侧摆角
                len(self.satellite_assignments[sat.id]),      # 已分配任务数
                sat.capabilities.data_rate / 500.0            # 数据速率
            ]
            features.append(feature)

        return np.array(features)

    def _build_visibility_matrix(self) -> np.ndarray:
        """构建可见性矩阵（简化实现）"""
        n_pending = len(self.pending_tasks)
        n_satellites = len(self.mission.satellites)

        if n_pending == 0:
            return np.zeros((0, n_satellites))

        # 简化：假设所有任务对所有卫星都可见
        # 实际实现应该基于轨道计算
        return np.ones((n_pending, n_satellites))
