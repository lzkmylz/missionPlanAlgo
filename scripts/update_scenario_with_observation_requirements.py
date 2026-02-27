#!/usr/bin/env python3
"""
更新场景文件，添加不同观测需求的目标

为每个目标设置不同的required_observations:
- 不限频次: 使用-1表示尽可能多观测
- 指定频次: 如6表示需要6次观测
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def update_scenario():
    scenario_path = 'scenarios/point_group_scenario.json'

    with open(scenario_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 保留第一个目标（北京）
    base_target = data['targets'][0].copy()

    # 创建多个目标，每个有不同的观测需求
    targets = []

    # 1. 北京 - 高频观测需求（6次）
    target1 = base_target.copy()
    target1.update({
        'id': 'TARGET-001',
        'name': '北京-高频观测',
        'longitude': 116.4,
        'latitude': 39.9,
        'priority': 9,
        'required_observations': 6,  # 需要6次观测
        'description': '首都重点区域，需要高频观测6次'
    })
    targets.append(target1)

    # 2. 上海 - 不限频次（尽可能多观测）
    target2 = base_target.copy()
    target2.update({
        'id': 'TARGET-002',
        'name': '上海-不限频次',
        'longitude': 121.5,
        'latitude': 31.2,
        'priority': 7,
        'required_observations': -1,  # -1表示不限，尽可能多观测
        'description': '上海区域，每次可见都进行观测'
    })
    targets.append(target2)

    # 3. 广州 - 中等频次（3次）
    target3 = base_target.copy()
    target3.update({
        'id': 'TARGET-003',
        'name': '广州-中频观测',
        'longitude': 113.3,
        'latitude': 23.1,
        'priority': 6,
        'required_observations': 3,  # 需要3次观测
        'description': '广州区域，需要3次观测'
    })
    targets.append(target3)

    # 4. 成都 - 低频观测（1次）
    target4 = base_target.copy()
    target4.update({
        'id': 'TARGET-004',
        'name': '成都-低频观测',
        'longitude': 104.1,
        'latitude': 30.6,
        'priority': 5,
        'required_observations': 1,  # 只需要1次观测
        'description': '成都区域，只需1次观测'
    })
    targets.append(target4)

    # 5. 西安 - 不限频次
    target5 = base_target.copy()
    target5.update({
        'id': 'TARGET-005',
        'name': '西安-不限频次',
        'longitude': 108.9,
        'latitude': 34.3,
        'priority': 6,
        'required_observations': -1,  # -1表示不限
        'description': '西安区域，每次可见都进行观测'
    })
    targets.append(target5)

    # 6. 武汉 - 高频观测（5次）
    target6 = base_target.copy()
    target6.update({
        'id': 'TARGET-006',
        'name': '武汉-高频观测',
        'longitude': 114.3,
        'latitude': 30.6,
        'priority': 7,
        'required_observations': 5,  # 需要5次观测
        'description': '武汉区域，需要5次观测'
    })
    targets.append(target6)

    # 更新场景数据
    data['targets'] = targets
    data['description'] = '多目标观测场景: 包含不同观测频次需求的目标（6个目标，频次需求1-6次或不限）'

    # 保存更新后的场景
    with open(scenario_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("场景文件已更新")
    print("=" * 60)
    print(f"目标数量: {len(targets)}")
    print("\n各目标观测需求:")
    for t in targets:
        freq = "不限" if t['required_observations'] == -1 else f"{t['required_observations']}次"
        print(f"  {t['id']}: {t['name']}")
        print(f"    位置: ({t['longitude']}, {t['latitude']})")
        print(f"    频次: {freq}")
        print(f"    优先级: {t['priority']}")
    print("=" * 60)


if __name__ == '__main__':
    update_scenario()
