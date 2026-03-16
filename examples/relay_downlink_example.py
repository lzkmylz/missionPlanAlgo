"""
中继卫星数传示例

演示如何使用中继卫星回传数据功能。
"""

import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_relay_network():
    """创建示例中继卫星网络"""
    from core.network.relay_satellite import RelaySatellite, RelayNetwork

    relays = [
        RelaySatellite(
            id='RELAY-01',
            name='天链-1号',
            orbit_type='GEO',
            longitude=120.0,  # 东经120度
            uplink_capacity=450.0,  # 450 Mbps
            downlink_capacity=450.0,
            coverage_zones=['Asia', 'Pacific']
        ),
        RelaySatellite(
            id='RELAY-02',
            name='天链-2号',
            orbit_type='GEO',
            longitude=-60.0,  # 西经60度
            uplink_capacity=450.0,
            downlink_capacity=450.0,
            coverage_zones=['Atlantic', 'Americas']
        ),
        RelaySatellite(
            id='RELAY-03',
            name='天链-3号',
            orbit_type='GEO',
            longitude=80.0,  # 东经80度
            uplink_capacity=300.0,
            downlink_capacity=300.0,
            coverage_zones=['Indian_Ocean', 'Asia']
        ),
    ]

    return RelayNetwork(relays)


def run_relay_scheduling_example():
    """运行中继数传调度示例"""
    logger.info("=" * 70)
    logger.info("中继卫星数传调度示例")
    logger.info("=" * 70)

    # 创建中继网络
    relay_network = create_sample_relay_network()
    logger.info(f"\n配置 {len(relay_network.relays)} 颗中继卫星:")
    for relay in relay_network.relays.values():
        logger.info(f"  - {relay.name}: {relay.longitude}°E, {relay.uplink_capacity}Mbps")

    # 模拟添加可见窗口（实际应从Java后端计算）
    from datetime import timedelta
    now = datetime.now()

    # 为卫星SAT-01添加与中继卫星的可见窗口
    relay_network.add_visibility_window(
        satellite_id='SAT-01',
        relay_id='RELAY-01',
        start_time=now + timedelta(hours=1),
        end_time=now + timedelta(hours=1, minutes=15)
    )
    relay_network.add_visibility_window(
        satellite_id='SAT-01',
        relay_id='RELAY-03',
        start_time=now + timedelta(hours=3),
        end_time=now + timedelta(hours=3, minutes=20)
    )

    # 创建中继调度器
    from scheduler.relay import RelayScheduler

    scheduler = RelayScheduler(
        relay_network=relay_network,
        default_data_rate_mbps=450.0,
        link_setup_time_seconds=10.0
    )

    # 初始化卫星固存
    scheduler.initialize_satellite_storage(
        satellite_id='SAT-01',
        current_gb=0.0,
        capacity_gb=128.0
    )

    logger.info("\n中继调度器初始化完成")

    # 检查数据中继可行性
    can_relay, latency = relay_network.can_relay_data(
        source_satellite='SAT-01',
        relay_id='RELAY-01',
        data_size=10.0,  # 10 GB
        start_time=now + timedelta(hours=1, minutes=5)
    )

    if can_relay:
        logger.info(f"\n可以通过RELAY-01中继10GB数据")
        logger.info(f"预计传输时间: {latency:.1f}秒")
    else:
        logger.info("\n无法通过RELAY-01中继数据")

    # 查找最佳中继
    best_relay = relay_network.find_best_relay(
        satellite_id='SAT-01',
        data_size=10.0,
        start_time=now + timedelta(hours=1)
    )
    logger.info(f"\n最佳中继卫星: {best_relay}")

    logger.info("\n" + "=" * 70)
    logger.info("示例完成")
    logger.info("=" * 70)


def run_mixed_downlink_example():
    """运行混合数传示例（地面站+中继）"""
    logger.info("\n")
    logger.info("=" * 70)
    logger.info("混合数传调度示例（地面站+中继）")
    logger.info("=" * 70)

    from scheduler.unified_scheduler import UnifiedScheduler

    # 混合调度策略:
    # 'ground_station_first' - 优先使用地面站
    # 'relay_first' - 优先使用中继
    # 'best_effort' - 优先地面站，失败则尝试中继

    config = {
        'downlink_strategy': 'best_effort',  # 混合策略
        'enable_downlink': True,
    }

    logger.info("\n混合调度策略配置:")
    logger.info(f"  策略: {config['downlink_strategy']}")
    logger.info("  - 优先使用地面站进行数传")
    logger.info("  - 地面站不可用时，自动切换到中继卫星")

    logger.info("\n" + "=" * 70)
    logger.info("示例完成")
    logger.info("=" * 70)


if __name__ == '__main__':
    run_relay_scheduling_example()
    run_mixed_downlink_example()
