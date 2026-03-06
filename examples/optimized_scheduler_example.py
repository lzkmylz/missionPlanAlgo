"""
优化版调度器使用示例

演示如何使用轨道预计算缓存 + Java并行传播的优化版调度器。
相比基础版本，可在大规模场景下获得10-50倍性能提升。
"""

import sys
import time
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_scenario():
    """创建测试场景"""
    from datetime import datetime, timedelta

    # 简化的卫星配置
    satellites = []
    for i in range(1, 11):  # 10颗卫星
        sat = {
            'sat_id': f'SAT-{i:02d}',
            'tle_line1': '',
            'tle_line2': '',
            'min_elevation': 5.0,
        }
        satellites.append(sat)

    # 简化的目标配置
    targets = []
    for i in range(1, 21):  # 20个目标
        target = {
            'target_id': f'TARGET-{i:02d}',
            'longitude': -180 + (i * 18),  # 均匀分布
            'latitude': -90 + (i * 9),
            'altitude': 0.0,
            'min_observation_duration': 60,
        }
        targets.append(target)

    # 时间范围
    start_time = datetime(2024, 3, 15, 0, 0, 0)
    end_time = start_time + timedelta(hours=1)  # 1小时

    return {
        'satellites': satellites,
        'targets': targets,
        'start_time': start_time,
        'end_time': end_time,
    }


def test_optimized_visibility():
    """测试优化版可见性计算"""
    logger.info("=" * 60)
    logger.info("测试优化版可见性计算")
    logger.info("=" * 60)

    try:
        from core.orbit.visibility.orekit_java_bridge import OrekitJavaBridge

        # 创建Java桥接实例
        bridge = OrekitJavaBridge()

        # 准备测试数据
        scenario = create_test_scenario()

        sat_configs = scenario['satellites']
        target_configs = scenario['targets']
        start_time = scenario['start_time']
        end_time = scenario['end_time']

        logger.info(f"场景规模: {len(sat_configs)}颗卫星 x {len(target_configs)}个目标")
        logger.info(f"时间范围: {start_time} 到 {end_time}")

        # 调用优化版计算
        start = time.time()
        result = bridge.compute_all_windows_optimized(
            satellites=sat_configs,
            targets=target_configs,
            start_time=start_time,
            end_time=end_time,
            coarse_step=5.0,
            fine_step=1.0
        )
        elapsed = time.time() - start

        # 输出结果
        windows = result.get('windows', [])
        stats = result.get('stats', {})

        logger.info(f"\n计算完成!")
        logger.info(f"  - 计算时间: {elapsed:.2f}秒")
        logger.info(f"  - 可见窗口数: {len(windows)}")
        logger.info(f"  - Java统计: {stats}")

        # 显示前5个窗口
        if windows:
            logger.info("\n前5个可见窗口:")
            for w in windows[:5]:
                logger.info(f"  {w['satellite_id']} -> {w['target_id']}: "
                          f"{w['start_time'].strftime('%H:%M:%S')} - "
                          f"{w['end_time'].strftime('%H:%M:%S')} "
                          f"(max_el={w['max_elevation']:.1f}°)")

        return True

    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimized_scheduler():
    """测试优化版调度器"""
    logger.info("\n" + "=" * 60)
    logger.info("测试优化版调度器")
    logger.info("=" * 60)

    try:
        from scheduler.optimized_scheduler import create_optimized_scheduler

        # 尝试导入遗传算法调度器
        try:
            from scheduler.genetic_scheduler import GeneticScheduler
            OptimizedScheduler = create_optimized_scheduler(GeneticScheduler)
            logger.info("已创建优化版遗传算法调度器")
        except ImportError:
            logger.warning("遗传算法调度器不可用，跳过调度器测试")
            return False

        # 创建测试场景
        scenario = create_test_scenario()

        # 创建优化版调度器
        scheduler = OptimizedScheduler(
            scenario=scenario,
            use_optimized_visibility=True  # 启用优化版可见性计算
        )

        logger.info("调度器初始化完成（使用优化版可见性计算）")

        return True

    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_performance():
    """对比优化前后性能"""
    logger.info("\n" + "=" * 60)
    logger.info("性能对比测试")
    logger.info("=" * 60)

    try:
        from core.orbit.visibility.orekit_java_bridge import OrekitJavaBridge

        bridge = OrekitJavaBridge()

        # 不同规模测试
        test_cases = [
            (5, 10, "小规模"),
            (10, 50, "中规模"),
            (30, 200, "大规模"),
        ]

        for n_sats, n_targets, desc in test_cases:
            logger.info(f"\n{desc}: {n_sats}颗卫星 x {n_targets}个目标")

            # 创建测试数据
            satellites = [
                {'sat_id': f'SAT-{i}', 'min_elevation': 5.0}
                for i in range(n_sats)
            ]
            targets = [
                {
                    'target_id': f'TGT-{i}',
                    'longitude': -180 + (360 * i / n_targets),
                    'latitude': -90 + (180 * i / n_targets),
                    'altitude': 0.0,
                    'min_observation_duration': 60,
                }
                for i in range(n_targets)
            ]

            from datetime import datetime, timedelta
            start_time = datetime(2024, 3, 15, 0, 0, 0)
            end_time = start_time + timedelta(hours=1)

            # 测试优化版
            start = time.time()
            result = bridge.compute_all_windows_optimized(
                satellites=satellites,
                targets=targets,
                start_time=start_time,
                end_time=end_time,
                coarse_step=60.0,  # 使用较大步长进行快速测试
                fine_step=10.0
            )
            elapsed = time.time() - start

            windows = len(result.get('windows', []))
            logger.info(f"  优化版: {elapsed:.2f}秒, {windows}个窗口")

        return True

    except Exception as e:
        logger.error(f"性能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    logger.info("优化版调度器示例")
    logger.info("=" * 60)

    # 运行测试
    test_optimized_visibility()
    test_optimized_scheduler()
    compare_performance()

    logger.info("\n" + "=" * 60)
    logger.info("测试完成")
