"""
双后端可见性计算工厂测试

TDD测试文件 - 实现VisibilityCalculatorFactory
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from core.orbit.visibility.calculator_factory import VisibilityCalculatorFactory
from core.orbit.visibility.base import VisibilityCalculator


class TestVisibilityCalculatorFactory:
    """测试可见性计算工厂"""

    def test_factory_creates_stk_when_available(self):
        """测试当STK可用时创建STK计算器"""
        with patch.object(VisibilityCalculatorFactory, '_check_stk_available', return_value=True):
            with patch('core.orbit.visibility.calculator_factory.STKVisibilityCalculator') as mock_stk:
                mock_instance = MagicMock()
                mock_stk.return_value = mock_instance

                result = VisibilityCalculatorFactory.create(preferred='auto')

                mock_stk.assert_called_once()
                assert result == mock_instance

    def test_factory_creates_orekit_when_stk_unavailable(self):
        """测试当STK不可用时创建Orekit计算器"""
        with patch.object(VisibilityCalculatorFactory, '_check_stk_available', return_value=False):
            with patch('core.orbit.visibility.calculator_factory.OrekitVisibilityCalculator') as mock_orekit:
                mock_instance = MagicMock()
                mock_orekit.return_value = mock_instance

                result = VisibilityCalculatorFactory.create(preferred='auto')

                mock_orekit.assert_called_once()
                assert result == mock_instance

    def test_factory_explicit_stk_selection(self):
        """测试显式选择STK后端"""
        with patch('core.orbit.visibility.calculator_factory.STKVisibilityCalculator') as mock_stk:
            mock_instance = MagicMock()
            mock_stk.return_value = mock_instance

            result = VisibilityCalculatorFactory.create(preferred='stk')

            mock_stk.assert_called_once()
            assert result == mock_instance

    def test_factory_explicit_orekit_selection(self):
        """测试显式选择Orekit后端"""
        with patch('core.orbit.visibility.calculator_factory.OrekitVisibilityCalculator') as mock_orekit:
            mock_instance = MagicMock()
            mock_orekit.return_value = mock_instance

            result = VisibilityCalculatorFactory.create(preferred='orekit')

            mock_orekit.assert_called_once()
            assert result == mock_instance

    def test_factory_invalid_backend_raises_error(self):
        """测试无效后端类型抛出错误"""
        with pytest.raises(ValueError) as exc_info:
            VisibilityCalculatorFactory.create(preferred='invalid_backend')

        assert 'invalid_backend' in str(exc_info.value)

    def test_factory_passes_config_to_calculator(self):
        """测试工厂将配置传递给计算器"""
        config = {'timeout': 30, 'precision': 'high'}

        with patch.object(VisibilityCalculatorFactory, '_check_stk_available', return_value=True):
            with patch('core.orbit.visibility.calculator_factory.STKVisibilityCalculator') as mock_stk:
                mock_instance = MagicMock()
                mock_stk.return_value = mock_instance

                VisibilityCalculatorFactory.create(preferred='stk', config=config)

                mock_stk.assert_called_once_with(config)

    def test_stk_availability_check(self):
        """测试STK可用性检查"""
        # 测试STK可用的情况
        with patch('importlib.util.find_spec', return_value=MagicMock()):
            assert VisibilityCalculatorFactory._check_stk_available() == True

        # 测试STK不可用的情况
        with patch('importlib.util.find_spec', return_value=None):
            assert VisibilityCalculatorFactory._check_stk_available() == False

    def test_factory_returns_base_calculator_interface(self):
        """测试工厂返回的对象实现基类接口"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        # 直接使用实际类进行测试，而不是mock
        result = VisibilityCalculatorFactory.create(preferred='orekit')

        # 验证返回的对象有基类定义的方法
        assert isinstance(result, VisibilityCalculator)
        assert hasattr(result, 'calculate_windows')
        assert hasattr(result, 'is_visible')


class TestSTKVisibilityCalculator:
    """测试STK可见性计算器（stub实现）"""

    def test_stk_calculator_implements_base_interface(self):
        """测试STK计算器实现基类接口"""
        from core.orbit.visibility.stk_visibility import STKVisibilityCalculator

        calculator = STKVisibilityCalculator()

        assert isinstance(calculator, VisibilityCalculator)
        assert hasattr(calculator, 'calculate_windows')
        assert hasattr(calculator, 'is_visible')

    def test_stk_calculator_returns_empty_windows(self):
        """测试STK计算器stub返回空窗口列表"""
        from core.orbit.visibility.stk_visibility import STKVisibilityCalculator

        calculator = STKVisibilityCalculator()
        windows = calculator.calculate_windows(
            satellite_id='SAT-01',
            target_id='TARGET-01',
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2)
        )

        assert windows == []

    def test_stk_calculator_is_visible_returns_false(self):
        """测试STK计算器stub is_visible返回False"""
        from core.orbit.visibility.stk_visibility import STKVisibilityCalculator

        calculator = STKVisibilityCalculator()
        result = calculator.is_visible(
            satellite_id='SAT-01',
            target_id='TARGET-01',
            time=datetime(2024, 1, 1, 12, 0)
        )

        assert result == False


class TestOrekitVisibilityCalculator:
    """测试Orekit可见性计算器（stub实现）"""

    def test_orekit_calculator_implements_base_interface(self):
        """测试Orekit计算器实现基类接口"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()

        assert isinstance(calculator, VisibilityCalculator)
        assert hasattr(calculator, 'calculate_windows')
        assert hasattr(calculator, 'is_visible')

    def test_orekit_calculator_returns_empty_windows(self):
        """测试Orekit计算器stub返回空窗口列表"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        windows = calculator.calculate_windows(
            satellite_id='SAT-01',
            target_id='TARGET-01',
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2)
        )

        assert windows == []

    def test_orekit_calculator_is_visible_returns_false(self):
        """测试Orekit计算器stub is_visible返回False"""
        from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

        calculator = OrekitVisibilityCalculator()
        result = calculator.is_visible(
            satellite_id='SAT-01',
            target_id='TARGET-01',
            time=datetime(2024, 1, 1, 12, 0)
        )

        assert result == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
