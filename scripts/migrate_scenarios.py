#!/usr/bin/env python3
"""
场景自动迁移脚本 - 将旧格式场景转换为新payload_config格式

使用方法:
    python scripts/migrate_scenarios.py --input scenarios/old_scenario.json --output scenarios/new_scenario.json
    python scripts/migrate_scenarios.py --all --backup  # 迁移所有场景并备份

功能:
1. 将旧格式的卫星capabilities转换为新的payload_config格式
2. 为光学卫星创建被动推扫模式配置
3. 为SAR卫星创建条带/聚束/扫描模式配置
4. 支持批量迁移和备份
"""

import json
import argparse
import sys
import os
import copy
import glob
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.constants import (
    OPTICAL_PUSH_BROOM_POWER_W,
    OPTICAL_PUSH_BROOM_DATA_RATE_MBPS,
    SAR_STRIPMAP_POWER_W,
    SAR_STRIPMAP_DATA_RATE_MBPS,
    SAR_SPOTLIGHT_POWER_W,
    SAR_SPOTLIGHT_DATA_RATE_MBPS,
    SAR_SCAN_POWER_W,
    SAR_SCAN_DATA_RATE_MBPS,
    SAR_SLIDING_SPOTLIGHT_POWER_W,
    SAR_SLIDING_SPOTLIGHT_DATA_RATE_MBPS,
    OPTICAL_IMAGING_MIN_DURATION_S,
    OPTICAL_IMAGING_MAX_DURATION_S,
    SAR_1_SPOTLIGHT_MIN_DURATION_S,
    SAR_1_SPOTLIGHT_MAX_DURATION_S,
    SAR_1_SLIDING_SPOTLIGHT_MIN_DURATION_S,
    SAR_1_SLIDING_SPOTLIGHT_MAX_DURATION_S,
    SAR_1_STRIPMAP_MIN_DURATION_S,
    SAR_1_STRIPMAP_MAX_DURATION_S,
)


def create_optical_payload_config(
    resolution: float,
    swath_width: float,
    data_rate: float,
    imager_fov: Optional[Dict] = None,
    spectral_bands: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    创建光学载荷的payload_config

    Args:
        resolution: 分辨率（米）
        swath_width: 幅宽（米）
        data_rate: 数据率（Mbps）- 作为参考
        imager_fov: imager中的FOV配置
        spectral_bands: 光谱波段列表

    Returns:
        payload_config字典
    """
    # 将swath_width从km转换为m（如果需要）
    if swath_width < 1000:  # 可能是km单位
        swath_width_m = swath_width * 1000
    else:
        swath_width_m = swath_width

    # 计算FOV半角（基于幅宽和典型轨道高度500km）
    import math
    altitude = 500000  # 500km
    half_angle = math.degrees(math.atan2(swath_width_m / 2, altitude))

    # 根据分辨率选择功耗（高分辨率功耗更高）
    if resolution <= 1.0:
        power = OPTICAL_PUSH_BROOM_POWER_W * 1.2  # 高分辨率增加功耗
    elif resolution <= 2.0:
        power = OPTICAL_PUSH_BROOM_POWER_W
    else:
        power = OPTICAL_PUSH_BROOM_POWER_W * 0.8  # 低分辨率降低功耗

    # 构建FOV配置
    fov_config = {
        "cross_track_fov_deg": round(half_angle * 2, 2),
        "along_track_fov_deg": round(half_angle * 0.4, 2),
    }

    # 如果有imager_fov，尝试提取更精确的配置
    if imager_fov:
        if "half_angle" in imager_fov:
            fov_config = {
                "cross_track_fov_deg": imager_fov["half_angle"] * 2,
                "along_track_fov_deg": imager_fov["half_angle"] * 0.4,
            }
        elif "half_angle_x" in imager_fov and "half_angle_y" in imager_fov:
            fov_config = {
                "along_track_fov_deg": imager_fov["half_angle_x"],
                "cross_track_fov_deg": imager_fov["half_angle_y"],
            }

    return {
        "payload_type": "optical",
        "default_mode": "push_broom",
        "description": f"光学载荷，分辨率{resolution}m，幅宽{swath_width_m/1000:.1f}km，被动推扫成像",
        "modes": {
            "push_broom": {
                "resolution_m": resolution,
                "swath_width_m": swath_width_m,
                "power_consumption_w": power,
                "data_rate_mbps": data_rate if data_rate > 0 else OPTICAL_PUSH_BROOM_DATA_RATE_MBPS,
                "min_duration_s": OPTICAL_IMAGING_MIN_DURATION_S,
                "max_duration_s": OPTICAL_IMAGING_MAX_DURATION_S,
                "mode_type": "optical",
                "fov_config": fov_config,
                "characteristics": {
                    "spectral_bands": spectral_bands or ["PAN", "RGB", "NIR"],
                    "description": "被动推扫式成像，适合大范围条带成像",
                }
            }
        },
        "common_fov": imager_fov if imager_fov else None,
    }


def create_sar_payload_config(
    resolution: float,
    swath_width: float,
    data_rate: float,
    imaging_modes: List[str],
    imager_fov: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    创建SAR载荷的payload_config

    Args:
        resolution: 基础分辨率（米）
        swath_width: 基础幅宽（米）
        data_rate: 数据率（Mbps）- 作为参考
        imaging_modes: 成像模式列表
        imager_fov: imager中的FOV配置

    Returns:
        payload_config字典
    """
    # 将swath_width从km转换为m（如果需要）
    if swath_width < 1000:  # 可能是km单位
        swath_width_m = swath_width * 1000
    else:
        swath_width_m = swath_width

    modes = {}

    # 条带模式（默认）
    if "stripmap" in imaging_modes:
        modes["stripmap"] = {
            "resolution_m": resolution,
            "swath_width_m": swath_width_m,
            "power_consumption_w": SAR_STRIPMAP_POWER_W,
            "data_rate_mbps": SAR_STRIPMAP_DATA_RATE_MBPS,
            "min_duration_s": SAR_1_STRIPMAP_MIN_DURATION_S,
            "max_duration_s": SAR_1_STRIPMAP_MAX_DURATION_S,
            "mode_type": "sar",
            "fov_config": imager_fov if imager_fov else {},
            "characteristics": {
                "polarization": "HH+HV",
                "incidence_angle_range": [20, 45],
                "description": "条带模式，标准SAR成像，中等分辨率大范围",
            }
        }

    # 聚束模式
    if "spotlight" in imaging_modes:
        modes["spotlight"] = {
            "resolution_m": max(0.5, resolution * 0.5),  # 聚束分辨率更高
            "swath_width_m": min(10000, swath_width_m * 0.3),  # 聚束幅宽更小
            "power_consumption_w": SAR_SPOTLIGHT_POWER_W,
            "data_rate_mbps": SAR_SPOTLIGHT_DATA_RATE_MBPS,
            "min_duration_s": SAR_1_SPOTLIGHT_MIN_DURATION_S,
            "max_duration_s": SAR_1_SPOTLIGHT_MAX_DURATION_S,
            "mode_type": "sar",
            "fov_config": imager_fov if imager_fov else {},
            "characteristics": {
                "polarization": "HH+HV+VV+VH",
                "incidence_angle_range": [25, 50],
                "description": "聚束模式，高分辨率成像，适合精细目标",
            }
        }

    # 扫描模式
    if "scan" in imaging_modes:
        modes["scan"] = {
            "resolution_m": resolution * 3,  # 扫描模式分辨率较低
            "swath_width_m": swath_width_m * 3,  # 扫描模式幅宽更大
            "power_consumption_w": SAR_SCAN_POWER_W,
            "data_rate_mbps": SAR_SCAN_DATA_RATE_MBPS,
            "min_duration_s": 8.0,
            "max_duration_s": 20.0,
            "mode_type": "sar",
            "fov_config": imager_fov if imager_fov else {},
            "characteristics": {
                "polarization": "HH+HV",
                "incidence_angle_range": [15, 55],
                "burst_duration_s": 0.5,
                "description": "扫描模式，超大幅宽成像，适合大范围快速普查",
            }
        }

    # 滑动聚束模式
    if "sliding_spotlight" in imaging_modes:
        modes["sliding_spotlight"] = {
            "resolution_m": max(1.0, resolution * 0.7),
            "swath_width_m": swath_width_m * 0.7,
            "power_consumption_w": SAR_SLIDING_SPOTLIGHT_POWER_W,
            "data_rate_mbps": SAR_SLIDING_SPOTLIGHT_DATA_RATE_MBPS,
            "min_duration_s": SAR_1_SLIDING_SPOTLIGHT_MIN_DURATION_S,
            "max_duration_s": SAR_1_SLIDING_SPOTLIGHT_MAX_DURATION_S,
            "mode_type": "sar",
            "fov_config": imager_fov if imager_fov else {},
            "characteristics": {
                "polarization": "HH+HV+VV",
                "incidence_angle_range": [22, 48],
                "description": "滑动聚束模式，平衡分辨率和幅宽",
            }
        }

    # 如果没有匹配的模式，创建默认条带模式
    if not modes:
        modes["stripmap"] = {
            "resolution_m": resolution,
            "swath_width_m": swath_width_m,
            "power_consumption_w": SAR_STRIPMAP_POWER_W,
            "data_rate_mbps": data_rate if data_rate > 0 else SAR_STRIPMAP_DATA_RATE_MBPS,
            "min_duration_s": SAR_1_STRIPMAP_MIN_DURATION_S,
            "max_duration_s": SAR_1_STRIPMAP_MAX_DURATION_S,
            "mode_type": "sar",
            "fov_config": imager_fov if imager_fov else {},
            "characteristics": {
                "polarization": "HH+HV",
                "incidence_angle_range": [20, 45],
                "description": "条带模式，标准SAR成像",
            }
        }

    return {
        "payload_type": "sar",
        "default_mode": "stripmap" if "stripmap" in modes else list(modes.keys())[0],
        "description": f"SAR载荷，支持{'/'.join(modes.keys())}模式",
        "modes": modes,
        "common_fov": imager_fov if imager_fov else None,
    }


def migrate_satellite(sat_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    将单个卫星配置从旧格式转换为新格式

    Args:
        sat_config: 卫星配置字典

    Returns:
        更新后的卫星配置字典
    """
    # 深度拷贝以避免修改原始数据
    new_config = copy.deepcopy(sat_config)

    # 检查是否已经有payload_config
    if "payload_config" in new_config.get("capabilities", {}):
        print(f"  卫星 {sat_config.get('id', 'unknown')} 已有payload_config，跳过")
        return new_config

    capabilities = new_config.get("capabilities", {})

    # 获取必要的参数
    sat_type = sat_config.get("sat_type", "optical")
    resolution = capabilities.get("resolution", 1.0)
    swath_width = capabilities.get("swath_width", 15000.0)
    data_rate = capabilities.get("data_rate", 300.0)
    imager = capabilities.get("imager", {})
    imager_fov = imager.get("fov_config") if isinstance(imager, dict) else None

    # 获取成像模式列表
    imaging_modes = capabilities.get("imaging_modes", [])
    if isinstance(imaging_modes, list) and len(imaging_modes) > 0:
        if isinstance(imaging_modes[0], str):
            mode_list = imaging_modes
        else:
            # 可能是对象列表，提取mode_id
            mode_list = [m.get("mode_id", "push_broom") for m in imaging_modes if isinstance(m, dict)]
    else:
        mode_list = ["push_broom"] if "optical" in sat_type else ["stripmap"]

    # 创建payload_config
    if "optical" in sat_type:
        spectral_bands = None
        if isinstance(imager, dict):
            spectral_bands = imager.get("spectral_bands")
        payload_config = create_optical_payload_config(
            resolution=resolution,
            swath_width=swath_width,
            data_rate=data_rate,
            imager_fov=imager_fov,
            spectral_bands=spectral_bands,
        )
    else:  # SAR
        payload_config = create_sar_payload_config(
            resolution=resolution,
            swath_width=swath_width,
            data_rate=data_rate,
            imaging_modes=mode_list,
            imager_fov=imager_fov,
        )

    # 添加payload_config到capabilities
    new_config["capabilities"]["payload_config"] = payload_config

    # 添加迁移注释
    new_config["capabilities"]["_comment_resolution_swath"] = "注意: resolution和swath_width已移至payload_config中"

    return new_config


def migrate_scenario(input_path: Path, output_path: Optional[Path] = None, backup: bool = True) -> bool:
    """
    迁移单个场景文件

    Args:
        input_path: 输入场景文件路径
        output_path: 输出文件路径，None则覆盖原文件
        backup: 是否创建备份

    Returns:
        True if successful
    """
    print(f"\n处理场景: {input_path}")

    # 读取场景文件
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            scenario = json.load(f)
    except Exception as e:
        print(f"  错误: 无法读取场景文件: {e}")
        return False

    # 检查是否已迁移
    satellites = scenario.get("satellites", [])
    if not satellites:
        print("  警告: 场景中没有卫星")
        return False

    # 检查第一个卫星是否已有payload_config
    first_sat = satellites[0]
    if "payload_config" in first_sat.get("capabilities", {}):
        print("  场景已包含payload_config，跳过迁移")
        return True

    # 迁移所有卫星
    migrated_count = 0
    for i, sat in enumerate(satellites):
        sat_id = sat.get("id", f"sat_{i}")
        print(f"  迁移卫星: {sat_id}")
        satellites[i] = migrate_satellite(sat)
        migrated_count += 1

    # 更新场景版本和元数据
    scenario["version"] = scenario.get("version", "1.0") + "+payload_config"
    scenario["migrated_at"] = datetime.now().isoformat()
    scenario["migration_note"] = "已迁移到payload_config格式，支持多成像模式配置"

    # 确定输出路径
    if output_path is None:
        output_path = input_path

    # 创建备份
    if backup and output_path == input_path:
        backup_path = input_path.with_suffix('.json.backup')
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                original = f.read()
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original)
            print(f"  备份已创建: {backup_path}")
        except Exception as e:
            print(f"  警告: 无法创建备份: {e}")

    # 保存迁移后的场景
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(scenario, f, indent=2, ensure_ascii=False)
        print(f"  已保存: {output_path}")
        print(f"  成功迁移 {migrated_count} 颗卫星")
        return True
    except Exception as e:
        print(f"  错误: 无法保存场景文件: {e}")
        return False


def migrate_all_scenarios(scenarios_dir: Path, backup: bool = True) -> Dict[str, bool]:
    """
    批量迁移所有场景文件

    Args:
        scenarios_dir: 场景目录
        backup: 是否创建备份

    Returns:
        迁移结果字典 {filename: success}
    """
    results = {}

    # 查找所有JSON场景文件
    scenario_files = list(scenarios_dir.glob("*.json"))

    print(f"\n找到 {len(scenario_files)} 个场景文件")

    for scenario_file in scenario_files:
        # 跳过备份文件
        if scenario_file.suffixes == ['.json', '.backup']:
            continue

        success = migrate_scenario(scenario_file, backup=backup)
        results[scenario_file.name] = success

    return results


def main():
    parser = argparse.ArgumentParser(
        description="场景自动迁移脚本 - 将旧格式场景转换为新payload_config格式"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="输入场景文件路径"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="输出文件路径（默认覆盖原文件）"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="迁移所有场景文件"
    )
    parser.add_argument(
        "--scenarios-dir",
        type=str,
        default="scenarios",
        help="场景目录路径（默认: scenarios）"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="不创建备份文件"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="试运行模式，不实际保存文件"
    )

    args = parser.parse_args()

    if args.dry_run:
        print("【试运行模式】不实际保存任何更改")

    if args.all:
        # 批量迁移
        scenarios_dir = Path(args.scenarios_dir)
        if not scenarios_dir.exists():
            print(f"错误: 场景目录不存在: {scenarios_dir}")
            sys.exit(1)

        results = migrate_all_scenarios(scenarios_dir, backup=not args.no_backup)

        print("\n" + "=" * 60)
        print("批量迁移完成")
        print("=" * 60)
        success_count = sum(1 for v in results.values() if v)
        print(f"成功: {success_count}/{len(results)}")

        for filename, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {filename}")

    elif args.input:
        # 单个文件迁移
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"错误: 输入文件不存在: {input_path}")
            sys.exit(1)

        output_path = Path(args.output) if args.output else None

        success = migrate_scenario(input_path, output_path, backup=not args.no_backup)

        if success:
            print("\n✓ 迁移成功")
        else:
            print("\n✗ 迁移失败")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
