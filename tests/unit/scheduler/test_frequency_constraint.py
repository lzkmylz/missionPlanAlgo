"""
测试调度算法的频次约束支持
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from scheduler.frequency_utils import (
    ObservationTask,
    create_observation_tasks,
    calculate_frequency_fitness,
    get_target_observation_requirement,
    is_target_fully_satisfied,
)


class TestCreateObservationTasks:
    """测试创建观测任务"""

    def test_single_observation_target(self):
        """测试只需要1次观测的目标"""
        target = Mock()
        target.id = "TARGET-001"
        target.name = "测试目标"
        target.priority = 5
        target.longitude = 116.4
        target.latitude = 39.9
        target.required_observations = 1

        tasks = create_observation_tasks([target])

        assert len(tasks) == 1
        assert tasks[0].task_id == "TARGET-001-OBS1"
        assert tasks[0].required_observations == 1

    def test_multiple_observations_target(self):
        """测试需要多次观测的目标"""
        target = Mock()
        target.id = "TARGET-002"
        target.name = "测试目标"
        target.priority = 5
        target.longitude = 116.4
        target.latitude = 39.9
        target.required_observations = 5

        tasks = create_observation_tasks([target])

        assert len(tasks) == 5
        assert tasks[0].task_id == "TARGET-002-OBS1"
        assert tasks[4].task_id == "TARGET-002-OBS5"

    def test_unlimited_observations_target(self):
        """测试不限频次的目标"""
        target = Mock()
        target.id = "TARGET-003"
        target.name = "测试目标"
        target.priority = 5
        target.longitude = 116.4
        target.latitude = 39.9
        target.required_observations = -1

        tasks = create_observation_tasks([target])

        # 不限频次应该创建10个任务
        assert len(tasks) == 10
        assert tasks[0].required_observations == -1

    def test_multiple_targets(self):
        """测试多个目标"""
        targets = []
        for i in range(3):
            target = Mock()
            target.id = f"TARGET-00{i+1}"
            target.name = f"目标{i+1}"
            target.priority = 5
            target.longitude = 116.0 + i
            target.latitude = 39.0 + i
            target.required_observations = i + 1
            targets.append(target)

        tasks = create_observation_tasks(targets)

        # 1 + 2 + 3 = 6
        assert len(tasks) == 6


class TestCalculateFrequencyFitness:
    """测试频次适应度计算"""

    def test_fully_satisfied_target(self):
        """测试完全满足的目标"""
        target = Mock()
        target.id = "TARGET-001"
        target.required_observations = 5

        target_obs_count = {"TARGET-001": 5}

        score = calculate_frequency_fitness(target_obs_count, [target], base_score=0)

        # 完成奖励 20 + 基础分
        assert score >= 20.0

    def test_partially_satisfied_target(self):
        """测试部分满足的目标"""
        target = Mock()
        target.id = "TARGET-001"
        target.required_observations = 10

        target_obs_count = {"TARGET-001": 5}

        score = calculate_frequency_fitness(target_obs_count, [target], base_score=0)

        # 50%完成 = 50% * 15 = 7.5
        assert 7.0 <= score <= 8.0

    def test_unlimited_target(self):
        """测试不限频次的目标"""
        target = Mock()
        target.id = "TARGET-001"
        target.required_observations = -1

        target_obs_count = {"TARGET-001": 5}

        score = calculate_frequency_fitness(target_obs_count, [target], base_score=0)

        # 5次 * 5 = 25
        assert score == 25.0

    def test_multiple_targets_mixed(self):
        """测试混合目标类型"""
        targets = []

        target1 = Mock()
        target1.id = "T1"
        target1.required_observations = 3
        targets.append(target1)

        target2 = Mock()
        target2.id = "T2"
        target2.required_observations = -1
        targets.append(target2)

        target_obs_count = {"T1": 3, "T2": 5}

        score = calculate_frequency_fitness(target_obs_count, targets, base_score=0)

        # T1: 完成 = 20
        # T2: 5次 * 5 = 25
        # 总计 = 45
        assert score == 45.0


class TestGetTargetObservationRequirement:
    """测试获取目标观测需求"""

    def test_with_required_observations(self):
        """测试有required_observations属性的目标"""
        target = Mock()
        target.required_observations = 5

        assert get_target_observation_requirement(target) == 5

    def test_without_required_observations(self):
        """测试没有required_observations属性的目标"""
        target = Mock()
        del target.required_observations  # 确保属性不存在

        # 应该使用默认值1
        # 但由于Mock的特性，getattr会返回1
        result = get_target_observation_requirement(target)
        assert result == 1

    def test_unlimited_observations(self):
        """测试不限频次"""
        target = Mock()
        target.required_observations = -1

        assert get_target_observation_requirement(target) == -1


class TestIsTargetFullySatisfied:
    """测试目标是否完全满足"""

    def test_satisfied_with_fixed_requirement(self):
        """测试固定需求已满足"""
        target = Mock()
        target.required_observations = 5

        assert is_target_fully_satisfied(target, 5) == True
        assert is_target_fully_satisfied(target, 6) == True

    def test_not_satisfied_with_fixed_requirement(self):
        """测试固定需求未满足"""
        target = Mock()
        target.required_observations = 5

        assert is_target_fully_satisfied(target, 3) == False

    def test_satisfied_with_unlimited(self):
        """测试不限频次只要有观测就算满足"""
        target = Mock()
        target.required_observations = -1

        assert is_target_fully_satisfied(target, 1) == True
        assert is_target_fully_satisfied(target, 10) == True

    def test_not_satisfied_with_unlimited(self):
        """测试不限频次没有观测就不满足"""
        target = Mock()
        target.required_observations = -1

        assert is_target_fully_satisfied(target, 0) == False
