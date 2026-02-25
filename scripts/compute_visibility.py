"""
可见性计算脚本

计算场景中所有卫星-目标和卫星-地面站的可见窗口。
结果保存到缓存文件，供调度算法使用。
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator
from core.models import Mission


def compute_visibility_windows(scenario_path: str, output_dir: str = "data/visibility_cache"):
    """
    计算场景中所有可见窗口。
    
    Args:
        scenario_path: 场景文件路径
        output_dir: 缓存输出目录
    """
    print("=" * 60)
    print("可见性计算")
    print("=" * 60)
    
    # 1. 加载场景
    print(f"\n1. 加载场景: {scenario_path}")
    mission = Mission.load(scenario_path)
    print(f"   - 卫星: {len(mission.satellites)} 颗")
    print(f"   - 目标: {len(mission.targets)} 个")
    print(f"   - 地面站: {len(mission.ground_stations)} 个")
    
    # 2. 创建计算器
    print("\n2. 初始化可见性计算器")
    calculator = OrekitVisibilityCalculator(config={
        'min_elevation': 0.0,      # 最小仰角（地平线以上）
        'time_step': 1           # 时间步长(秒)
    })
    
    # 3. 计算时间范围
    start_time = mission.start_time
    end_time = mission.end_time
    print(f"\n3. 计算时间范围")
    print(f"   - 开始: {start_time}")
    print(f"   - 结束: {end_time}")
    
    # 4. 计算卫星-目标可见窗口
    print("\n4. 计算卫星-目标可见窗口")
    target_windows = []
    for sat in mission.satellites:
        for target in mission.targets:
            windows = calculator.compute_satellite_target_windows(
                satellite=sat,
                target=target,
                start_time=start_time,
                end_time=end_time
            )
            target_windows.extend(windows)
            if windows:
                print(f"   {sat.id} -> {target.id}: {len(windows)} 个窗口")
    
    # 5. 计算卫星-地面站可见窗口
    print("\n5. 计算卫星-地面站可见窗口")
    gs_windows = []
    for sat in mission.satellites:
        for gs in mission.ground_stations:
            windows = calculator.compute_satellite_ground_station_windows(
                satellite=sat,
                ground_station=gs,
                start_time=start_time,
                end_time=end_time
            )
            gs_windows.extend(windows)
            if windows:
                print(f"   {sat.id} -> {gs.id}: {len(windows)} 个窗口")
    
    # 6. 保存结果
    print("\n6. 保存缓存")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cache_file = Path(output_dir) / f"{Path(scenario_path).stem}_windows.json"
    cache_data = {
        'scenario': scenario_path,
        'computed_at': datetime.now().isoformat(),
        'target_windows': [
            {
                'satellite_id': w.satellite_id,
                'target_id': w.target_id,
                'start_time': w.start_time.isoformat(),
                'end_time': w.end_time.isoformat(),
                'duration': (w.end_time - w.start_time).total_seconds()
            }
            for w in target_windows
        ],
        'ground_station_windows': [
            {
                'satellite_id': w.satellite_id,
                'target_id': w.target_id,
                'start_time': w.start_time.isoformat(),
                'end_time': w.end_time.isoformat(),
                'duration': (w.end_time - w.start_time).total_seconds()
            }
            for w in gs_windows
        ]
    }
    
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)
    
    print(f"   缓存已保存: {cache_file}")
    
    # 7. 统计
    print("\n" + "=" * 60)
    print("计算完成")
    print("=" * 60)
    print(f"卫星-目标窗口: {len(target_windows)} 个")
    print(f"卫星-地面站窗口: {len(gs_windows)} 个")
    print(f"总计: {len(target_windows) + len(gs_windows)} 个窗口")
    
    return cache_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="计算场景可见性窗口")
    parser.add_argument("--scenario", "-s", default="scenarios/point_group_scenario.json",
                       help="场景文件路径")
    parser.add_argument("--output", "-o", default="data/visibility_cache",
                       help="缓存输出目录")
    
    args = parser.parse_args()
    
    compute_visibility_windows(args.scenario, args.output)
