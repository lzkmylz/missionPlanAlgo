#!/usr/bin/env python3
"""
场景缓存管理工具

用法:
    # 检查场景是否有可用缓存
    python scripts/cache_manager.py check -s scenario.json

    # 列出所有缓存
    python scripts/cache_manager.py list

    # 清理过期缓存
    python scripts/cache_manager.py clean --older-than 30

    # 分析两个场景的配置复用可能性
    python scripts/cache_manager.py analyze -s scene1.json -S scene2.json

    # 预计算场景缓存（自动检测并复用已有缓存）
    python scripts/cache_manager.py compute -s scenario.json
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.cache.fingerprint_calculator import FingerprintCalculator, FingerprintComparator
from core.cache.index_manager import CacheIndexManager
from core.cache.fingerprint import ScenarioFingerprint


def format_size_mb(size_mb: float) -> str:
    """格式化大小显示"""
    if size_mb >= 1024:
        return f"{size_mb/1024:.2f} GB"
    else:
        return f"{size_mb:.1f} MB"


def cmd_check(args):
    """检查场景缓存状态"""
    # 规范化路径
    scenario_path = Path(args.scenario).resolve()
    if not scenario_path.exists():
        print(f"错误: 场景文件不存在: {scenario_path}")
        return 1

    print(f"检查场景: {scenario_path}")
    print("-" * 60)

    # 计算指纹
    calculator = FingerprintCalculator()
    fingerprint = calculator.calculate(str(scenario_path))

    print(f"场景名称: {fingerprint.scenario_name}")
    print(f"完整哈希: {fingerprint.full_hash}")
    print()
    print("组件哈希:")
    print(f"  卫星配置:     {fingerprint.satellites.hash_value} ({fingerprint.satellites.item_count}颗)")
    print(f"  地面站配置:   {fingerprint.ground_stations.hash_value} ({fingerprint.ground_stations.item_count}个)")
    print(f"  目标配置:     {fingerprint.targets.hash_value} ({fingerprint.targets.item_count}个)")
    print(f"  时间范围:     {fingerprint.time_range.hash_value}")
    print()

    # 查询缓存索引
    manager = CacheIndexManager()
    entry = manager.find(fingerprint)

    if entry:
        print("[OK] 找到匹配的缓存:")
        print(f"  缓存文件: {entry.cache_file}")
        if entry.orbit_file:
            print(f"  轨道文件: {entry.orbit_file}")
        print(f"  创建时间: {entry.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  访问次数: {entry.access_count}")
        print(f"  文件大小: {format_size_mb(entry.file_size_mb)}")
        if entry.stats:
            print("  统计信息:")
            for key, value in entry.stats.items():
                print(f"    {key}: {value}")
        return 0
    else:
        print("[NO] 未找到匹配缓存")
        print()
        # 查找可复用的部分缓存
        orbit_entry = manager.find_reusable_orbit_cache(fingerprint)
        if orbit_entry:
            print("[!] 发现可复用的轨道缓存:")
            print(f"  来源场景: {orbit_entry.scenario_name}")
            print(f"  轨道文件: {orbit_entry.orbit_file}")
            print(f"  可节省: 轨道计算时间 (~60-80秒)")
            print()
            print("  使用方式:")
            print(f"    python scripts/compute_visibility.py -s {args.scenario}")
            print("    # 将自动检测并复用轨道缓存")

        # 查找部分匹配
        partial_matches = manager.find_by_components(
            fingerprint,
            ['satellites', 'ground_stations', 'targets', 'time_range']
        )

        # 过滤掉完全匹配（已处理）和0匹配
        partial_matches = [(e, s) for e, s in partial_matches if s < 1.0 and s > 0]

        if partial_matches:
            print()
            print("  部分匹配的缓存:")
            for entry, score in partial_matches[:3]:  # 只显示前3个
                print(f"    - {entry.scenario_name}: {score*100:.0f}% 匹配")

        return 1


def cmd_list(args):
    """列出所有缓存"""
    manager = CacheIndexManager()
    entries = manager.list_entries()

    if not entries:
        print("缓存目录为空")
        return 0

    # 按大小排序如果指定了
    if args.sort == 'size':
        entries.sort(key=lambda e: e.file_size_mb, reverse=True)
    elif args.sort == 'access':
        entries.sort(key=lambda e: e.access_count, reverse=True)
    # 默认按访问时间排序

    print(f"{'哈希':<18} {'场景名':<30} {'状态':<10} {'大小':<12} {'访问':<8} {'最后访问'}")
    print("-" * 100)

    total_size = 0
    for entry in entries:
        name = entry.scenario_name[:28] if len(entry.scenario_name) <= 28 else entry.scenario_name[:25] + "..."
        last_access = entry.accessed_at.strftime("%Y-%m-%d %H:%M") if entry.accessed_at else "Never"
        size_str = format_size_mb(entry.file_size_mb)

        print(f"{entry.full_hash[:16]:<18} {name:<30} {entry.status.value:<10} "
              f"{size_str:<12} {entry.access_count:<8} {last_access}")
        total_size += entry.file_size_mb

    print("-" * 100)
    stats = manager.get_stats()
    print(f"总计: {stats['valid_entries']} 个有效缓存, "
          f"{format_size_mb(stats['total_size_mb'])}, "
          f"平均访问 {stats['avg_access_count']:.1f} 次")


def cmd_clean(args):
    """清理缓存"""
    manager = CacheIndexManager()

    print("缓存清理")
    print("-" * 60)

    if args.dry_run:
        print("【试运行模式】不会实际删除文件")
        print()

    # 先统计
    stats = manager.cleanup(
        older_than_days=args.older_than,
        max_size_mb=args.max_size,
        dry_run=True  # 先试运行看统计
    )

    if stats['deleted'] == 0:
        print("没有需要清理的缓存")
        return 0

    print(f"将清理 {stats['deleted']} 个缓存, 释放 {format_size_mb(stats['freed_mb'])}")
    print()

    if args.dry_run:
        print("缓存列表:")
        for entry_info in stats['entries']:
            print(f"  - {entry_info['name']} ({format_size_mb(entry_info['size_mb'])})")
    else:
        # 确认删除
        if not args.force:
            response = input("确认删除? [y/N]: ")
            if response.lower() != 'y':
                print("已取消")
                return 0

        # 实际清理
        stats = manager.cleanup(
            older_than_days=args.older_than,
            max_size_mb=args.max_size,
            dry_run=False
        )
        print(f"清理完成:")
        print(f"  删除缓存: {stats['deleted']} 个")
        print(f"  释放空间: {format_size_mb(stats['freed_mb'])}")


def cmd_analyze(args):
    """分析两个场景的复用可能性"""
    print("场景对比分析")
    print("-" * 60)

    # 规范化路径
    scenario1_path = Path(args.scenario).resolve()
    scenario2_path = Path(args.scenario2).resolve()

    if not scenario1_path.exists():
        print(f"错误: 场景文件不存在: {scenario1_path}")
        return 1
    if not scenario2_path.exists():
        print(f"错误: 场景文件不存在: {scenario2_path}")
        return 1

    calc = FingerprintCalculator()
    fp1 = calc.calculate(str(scenario1_path))
    fp2 = calc.calculate(str(scenario2_path))

    print(f"场景1: {fp1.scenario_name}")
    print(f"  路径: {args.scenario}")
    print(f"  哈希: {fp1.full_hash[:16]}...")
    print(f"  卫星: {fp1.satellites.item_count}颗, 目标: {fp1.targets.item_count}个")
    print()
    print(f"场景2: {fp2.scenario_name}")
    print(f"  路径: {args.scenario2}")
    print(f"  哈希: {fp2.full_hash[:16]}...")
    print(f"  卫星: {fp2.satellites.item_count}颗, 目标: {fp2.targets.item_count}个")
    print()

    comparator = FingerprintComparator()
    result = comparator.compare(fp1, fp2)

    print("对比结果:")
    print(f"  完全相同:       {'[YES]' if result['identical'] else '[NO]'}")
    print(f"  卫星配置相同:   {'[YES]' if result['same_satellites'] else '[NO]'} "
          f"(共同卫星: {len(result['common_satellites'])}颗)")
    print(f"  地面站配置相同: {'[YES]' if result['same_ground_stations'] else '[NO]'}")
    print(f"  目标配置相同:   {'[YES]' if result['same_targets'] else '[NO]'} "
          f"(共同目标: {len(result['common_targets'])}个)")
    print(f"  时间范围相同:   {'[YES]' if result['same_time_range'] else '[NO]'}")
    print()
    print(f"建议: {result['recommendation']}")

    if result['reusable_components']:
        print()
        print("可复用组件:")
        for comp in result['reusable_components']:
            if comp == 'all':
                print("  - 完整缓存 (直接复用)")
            elif comp == 'orbit':
                print("  - 轨道数据 (可跳过轨道计算)")
            elif comp == 'satellite_config':
                print("  - 卫星配置 (但需重新计算轨道)")
            elif comp == 'partial_orbit':
                common = len(result['common_satellites'])
                print(f"  - 部分轨道数据 ({common}颗卫星的轨道可复用)")


def cmd_compute(args):
    """预计算场景缓存（带智能复用检测）"""
    print("场景缓存预计算")
    print("-" * 60)

    # 规范化路径
    scenario_path = Path(args.scenario).resolve()
    if not scenario_path.exists():
        print(f"错误: 场景文件不存在: {scenario_path}")
        return 1

    # 计算指纹
    calculator = FingerprintCalculator()
    fingerprint = calculator.calculate(str(scenario_path))

    print(f"场景: {fingerprint.scenario_name}")
    print(f"哈希: {fingerprint.full_hash[:16]}...")
    print()

    # 检查已有缓存
    manager = CacheIndexManager()
    existing = manager.find(fingerprint)

    if existing and not args.force:
        print("[OK] 已存在匹配的缓存，跳过计算")
        print(f"  缓存文件: {existing.cache_file}")
        return 0

    # 检查可复用的轨道缓存
    orbit_entry = manager.find_reusable_orbit_cache(fingerprint)
    if orbit_entry and not args.no_reuse:
        print("[!] 发现可复用的轨道缓存:")
        print(f"  来源: {orbit_entry.scenario_name}")
        print(f"  将跳过轨道计算，直接计算可见性窗口")
        print()
        reuse_orbit = True
    else:
        reuse_orbit = False

    # 调用 compute_visibility.py 进行计算
    print("开始计算...")

    import subprocess
    import json

    cmd = [
        sys.executable,
        "scripts/compute_visibility.py",
        "-s", str(scenario_path),
        "-o", args.output or "results/visibility_cache.json"
    ]

    if args.coarse_step:
        cmd.extend(["--coarse-step", str(args.coarse_step)])
    if args.fine_step:
        cmd.extend(["--fine-step", str(args.fine_step)])

    # 如果可复用轨道，需要传递额外参数
    if reuse_orbit and orbit_entry:
        # 这里可以添加 --reuse-orbit 参数给 compute_visibility.py
        # 目前先打印信息
        print(f"  (将复用轨道缓存: {orbit_entry.orbit_file})")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print("[OK] 计算完成")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 计算失败: {e}")
        print(e.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(
        description='场景缓存管理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查场景缓存
  python scripts/cache_manager.py check -s scenarios/large_scale_frequency.json

  # 列出所有缓存（按访问时间排序）
  python scripts/cache_manager.py list

  # 列出缓存（按大小排序）
  python scripts/cache_manager.py list --sort size

  # 清理30天未访问的缓存
  python scripts/cache_manager.py clean --older-than 30

  # 分析两个场景的复用可能性
  python scripts/cache_manager.py analyze -s scene1.json -S scene2.json

  # 预计算场景（自动检测复用）
  python scripts/cache_manager.py compute -s scenario.json
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # check 命令
    check_parser = subparsers.add_parser('check', help='检查场景缓存状态')
    check_parser.add_argument('-s', '--scenario', required=True, help='场景文件路径')
    check_parser.set_defaults(func=cmd_check)

    # list 命令
    list_parser = subparsers.add_parser('list', help='列出所有缓存')
    list_parser.add_argument(
        '--sort',
        choices=['time', 'size', 'access'],
        default='time',
        help='排序方式 (默认: time)'
    )
    list_parser.set_defaults(func=cmd_list)

    # clean 命令
    clean_parser = subparsers.add_parser('clean', help='清理过期缓存')
    clean_parser.add_argument('--older-than', type=int, help='删除超过N天未访问的缓存')
    clean_parser.add_argument('--max-size', type=float, help='最大总大小(MB)，超出时删除最少使用的缓存')
    clean_parser.add_argument('--dry-run', action='store_true', help='试运行，不实际删除')
    clean_parser.add_argument('-f', '--force', action='store_true', help='强制删除，不提示确认')
    clean_parser.set_defaults(func=cmd_clean)

    # analyze 命令
    analyze_parser = subparsers.add_parser('analyze', help='分析两个场景的复用可能性')
    analyze_parser.add_argument('-s', '--scenario', required=True, help='场景1文件路径')
    analyze_parser.add_argument('-S', '--scenario2', required=True, help='场景2文件路径')
    analyze_parser.set_defaults(func=cmd_analyze)

    # compute 命令
    compute_parser = subparsers.add_parser('compute', help='预计算场景缓存（带智能复用检测）')
    compute_parser.add_argument('-s', '--scenario', required=True, help='场景文件路径')
    compute_parser.add_argument('-o', '--output', help='输出缓存文件路径')
    compute_parser.add_argument('--coarse-step', type=float, help='粗扫描步长(秒)')
    compute_parser.add_argument('--fine-step', type=float, help='精化步长(秒)')
    compute_parser.add_argument('--force', action='store_true', help='强制重新计算，即使已有缓存')
    compute_parser.add_argument('--no-reuse', action='store_true', help='禁用轨道缓存复用')
    compute_parser.set_defaults(func=cmd_compute)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
