#!/usr/bin/env python3
"""
光照约束检查与修复脚本

检查当前可见性窗口是否正确考虑了光照约束，并提供修复方案

功能：
1. 分析当前可见性窗口的光照条件
2. 计算每个窗口的太阳高度角
3. 过滤掉不满足光照约束的光学卫星窗口
4. 生成新的可见性缓存

用法:
    python scripts/check_and_fix_lighting.py --check
    python scripts/check_and_fix_lighting.py --fix
"""

import sys
import json
import math
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))


def calculate_sun_position(dt: datetime) -> Tuple[float, float]:
    """
    计算太阳在地心惯性系中的位置（简化算法）

    Returns:
        (ra, dec): 太阳赤经和赤纬（度）
    """
    # 使用简化算法计算太阳位置
    # 基于2024年3月15日春分附近

    # 年积日
    year_start = datetime(dt.year, 1, 1, tzinfo=dt.tzinfo)
    day_of_year = (dt - year_start).total_seconds() / 86400

    # 简化太阳黄经计算（春分附近太阳黄经约0度）
    # 每天约前进1度
    solar_longitude = (day_of_year - 74) * 0.9856  # 3月15日是第74天

    # 赤经和赤纬（简化）
    solar_ra = solar_longitude  # 近似
    solar_dec = 23.44 * math.sin(math.radians(solar_longitude))  # 黄赤交角23.44度

    return solar_ra, solar_dec


def calculate_solar_elevation(lat: float, lon: float, dt: datetime) -> float:
    """
    计算指定地点的太阳高度角

    Args:
        lat: 纬度（度）
        lon: 经度（度）
        dt: UTC时间

    Returns:
        太阳高度角（度）
    """
    # 计算年积日
    year_start = datetime(dt.year, 1, 1, tzinfo=dt.tzinfo)
    day_of_year = (dt - year_start).total_seconds() / 86400
    n = day_of_year - 1  # 从0开始的天数

    # 太阳平均黄经（度）
    L = (280.460 + 0.9856474 * n) % 360

    # 太阳平均近点角（度）
    g = (357.528 + 0.9856003 * n) % 360
    g_rad = math.radians(g)

    # 太阳黄经（度）
    lambda_sun = L + 1.915 * math.sin(g_rad) + 0.020 * math.sin(2 * g_rad)
    lambda_sun = lambda_sun % 360

    # 太阳赤纬（度）
    delta = math.degrees(math.asin(math.sin(math.radians(lambda_sun)) * math.sin(math.radians(23.44))))

    # 计算地方时（经度每15度对应1小时时差）
    # 东经为正，西经为负
    ut = dt.hour + dt.minute / 60 + dt.second / 3600  # UTC时间（小时）
    local_time = ut + lon / 15  # 地方时（小时）

    # 计算时角（度）
    # 地方时12:00为正午，时角为0
    # 每小时对应15度
    omega = (local_time - 12) * 15

    # 计算太阳高度角
    lat_rad = math.radians(lat)
    delta_rad = math.radians(delta)
    omega_rad = math.radians(omega)

    sin_h = math.sin(lat_rad) * math.sin(delta_rad) + \
            math.cos(lat_rad) * math.cos(delta_rad) * math.cos(omega_rad)

    # 限制范围避免数值误差
    sin_h = max(-1.0, min(1.0, sin_h))

    h = math.degrees(math.asin(sin_h))

    return h


def check_current_windows(visibility_path: str, scenario_path: str):
    """检查当前可见性窗口的光照条件"""

    print("=" * 80)
    print("光照约束检查报告")
    print("=" * 80)

    # 加载场景获取目标位置
    with open(scenario_path, 'r') as f:
        scenario = json.load(f)

    targets = {t['id']: t for t in scenario['targets']}

    # 加载可见性窗口
    with open(visibility_path, 'r') as f:
        vis_data = json.load(f)

    target_windows = vis_data['windows']['target_windows']

    print(f"\n总窗口数: {len(target_windows)}")

    # 分类统计
    opt_windows = [w for w in target_windows if w['satellite_id'].startswith('OPT')]
    sar_windows = [w for w in target_windows if w['satellite_id'].startswith('SAR')]

    print(f"光学卫星窗口: {len(opt_windows)}")
    print(f"SAR卫星窗口: {len(sar_windows)}")

    # 采样检查光学卫星窗口的光照条件
    print("\n" + "-" * 80)
    print("光学卫星窗口光照条件采样检查（每1000个检查1个）")
    print("-" * 80)

    lighting_stats = {'day': 0, 'night': 0, 'dawn_dusk': 0}

    for i, w in enumerate(opt_windows[::1000]):  # 采样
        target = targets.get(w['target_id'])
        if not target:
            continue

        dt = datetime.fromisoformat(w['start_time'])
        solar_elev = calculate_solar_elevation(
            target['location'][1],  # lat
            target['location'][0],  # lon
            dt
        )

        # 分类
        if solar_elev > 15:
            lighting_stats['day'] += 1
        elif solar_elev < -6:
            lighting_stats['night'] += 1
        else:
            lighting_stats['dawn_dusk'] += 1

        if i < 10:  # 只显示前10个样例
            status = "✓ 可用" if solar_elev > 15 else "✗ 光照不足"
            print(f"  {w['satellite_id']} → {w['target_id']}: "
                  f"{dt.strftime('%H:%M')} 太阳高度角={solar_elev:5.1f}° {status}")

    print(f"\n采样统计 (样本数: {sum(lighting_stats.values())}):")
    print(f"  白天 (>15°): {lighting_stats['day']} ({lighting_stats['day']/sum(lighting_stats.values())*100:.1f}%)")
    print(f"  黄昏 (-6°~15°): {lighting_stats['dawn_dusk']}")
    print(f"  夜间 (<-6°): {lighting_stats['night']} ({lighting_stats['night']/sum(lighting_stats.values())*100:.1f}%)")

    if lighting_stats['night'] / sum(lighting_stats.values()) > 0.3:
        print("\n⚠️ 警告: 大量光学卫星窗口在夜间，光照约束未生效！")
        return False
    else:
        print("\n✓ 光学卫星窗口主要在白天，光照约束已生效")
        return True


def fix_optical_windows(visibility_path: str, scenario_path: str,
                        output_path: str, min_solar_elevation: float = 15.0):
    """
    修复光学卫星窗口，过滤掉不满足光照约束的窗口
    """
    print("=" * 80)
    print("修复光学卫星窗口光照约束")
    print("=" * 80)

    # 加载场景
    with open(scenario_path, 'r') as f:
        scenario = json.load(f)

    targets = {t['id']: t for t in scenario['targets']}

    # 加载可见性窗口
    with open(visibility_path, 'r') as f:
        vis_data = json.load(f)

    print(f"\n最小太阳高度角约束: {min_solar_elevation}°")
    print(f"原始窗口总数: {vis_data['stats']['total_windows']}")

    # 处理目标窗口
    filtered_target_windows = []
    removed_count = 0
    kept_count = 0

    for w in vis_data['windows']['target_windows']:
        # SAR卫星不过滤
        if w['satellite_id'].startswith('SAR'):
            filtered_target_windows.append(w)
            kept_count += 1
            continue

        # 光学卫星检查光照条件
        target = targets.get(w['target_id'])
        if not target:
            filtered_target_windows.append(w)
            kept_count += 1
            continue

        dt = datetime.fromisoformat(w['start_time'])
        solar_elev = calculate_solar_elevation(
            target['location'][1],
            target['location'][0],
            dt
        )

        if solar_elev >= min_solar_elevation:
            # 添加太阳高度角信息
            w_with_lighting = w.copy()
            w_with_lighting['solar_elevation'] = round(solar_elev, 2)
            filtered_target_windows.append(w_with_lighting)
            kept_count += 1
        else:
            removed_count += 1

    # 地面站窗口不过滤（数据传输不需要光照）
    filtered_gs_windows = vis_data['windows']['ground_station_windows']

    print(f"\n过滤结果:")
    print(f"  保留窗口: {kept_count}")
    print(f"  移除窗口: {removed_count}")
    print(f"  移除比例: {removed_count/(kept_count+removed_count)*100:.1f}%")

    # 生成新数据
    new_data = vis_data.copy()
    new_data['windows']['target_windows'] = filtered_target_windows
    new_data['stats']['total_windows'] = len(filtered_target_windows) + len(filtered_gs_windows)
    new_data['stats']['target_windows'] = len(filtered_target_windows)
    new_data['stats']['lighting_constraint'] = {
        'min_solar_elevation': min_solar_elevation,
        'removed_windows': removed_count,
        'kept_windows': kept_count
    }
    new_data['metadata']['lighting_fixed'] = True
    new_data['metadata']['fixed_at'] = datetime.now().isoformat()

    # 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 修复后的可见性缓存已保存: {output_path}")

    # 验证修复结果
    print("\n验证修复结果:")
    opt_kept = [w for w in filtered_target_windows if w['satellite_id'].startswith('OPT')]

    # 按时段统计
    hours = Counter()
    for w in opt_kept:
        hour = datetime.fromisoformat(w['start_time']).hour
        hours[hour] += 1

    day_count = sum(hours[h] for h in range(6, 18))
    night_count = sum(hours[h] for h in list(range(18, 24)) + list(range(0, 6)))

    print(f"  光学卫星白天窗口(06-18): {day_count}")
    print(f"  光学卫星夜间窗口(18-06): {night_count}")
    print(f"  夜间占比: {night_count/(day_count+night_count)*100:.1f}%")

    if night_count / (day_count + night_count) < 0.1:
        print("\n✓ 修复成功！光学卫星窗口主要分布在白天")
    else:
        print("\n⚠️ 仍有较多夜间窗口，可能需要检查算法")


def main():
    parser = argparse.ArgumentParser(
        description='光照约束检查与修复',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='检查当前可见性窗口的光照条件'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='修复光学卫星窗口，过滤不满足光照约束的窗口'
    )
    parser.add_argument(
        '--scenario',
        default='scenarios/large_scale_frequency.json',
        help='场景文件路径'
    )
    parser.add_argument(
        '--visibility',
        default='results/large_scale_visibility.json',
        help='可见性缓存路径'
    )
    parser.add_argument(
        '--output',
        default='results/large_scale_visibility_lighting_fixed.json',
        help='修复后输出路径'
    )
    parser.add_argument(
        '--min-elevation',
        type=float,
        default=15.0,
        help='最小太阳高度角（度）'
    )

    args = parser.parse_args()

    if args.check:
        check_current_windows(args.visibility, args.scenario)
    elif args.fix:
        fix_optical_windows(
            args.visibility,
            args.scenario,
            args.output,
            args.min_elevation
        )
    else:
        # 默认执行检查和修复
        print("执行光照约束检查...\n")
        is_ok = check_current_windows(args.visibility, args.scenario)

        if not is_ok:
            print("\n" + "=" * 80)
            print("发现问题，执行修复...")
            print("=" * 80)
            fix_optical_windows(
                args.visibility,
                args.scenario,
                args.output,
                args.min_elevation
            )


if __name__ == '__main__':
    main()
