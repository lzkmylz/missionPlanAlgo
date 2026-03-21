"""
Generate scenarios/large_scale_isl.json from scenarios/large_scale_frequency.json.

Adds ISL capabilities to all 60 satellites with topology-aware peer_links:
  1. Same-plane ring laser links (within OPT and SAR planes)
  2. Cross-plane laser links (OPT-OPT and SAR-SAR adjacent planes)
  3. Cross-constellation microwave links (each OPT → nearest SAR, bidirectional)

Usage:
    python3 scripts/generate_isl_scenario.py
"""

import json
import copy
import sys
import os

# Paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(PROJECT_ROOT, 'scenarios', 'large_scale_frequency.json')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'scenarios', 'large_scale_isl.json')


def raan_diff(r1: float, r2: float) -> float:
    d = abs(r1 - r2)
    return min(d, 360.0 - d)


def ma_diff(m1: float, m2: float) -> float:
    d = abs(m1 - m2)
    return min(d, 360.0 - d)


def add_peer_link(peer_links_map: dict, src: str, dst: str,
                  link_type: str, preferred: bool) -> None:
    peer_links_map.setdefault(src, [])
    if not any(p['peer_satellite_id'] == dst for p in peer_links_map[src]):
        peer_links_map[src].append({
            "peer_satellite_id": dst,
            "link_type": link_type,
            "enabled": True,
            "preferred": preferred
        })


def add_ring_links(planes_dict: dict, peer_links_map: dict) -> None:
    """Add same-plane ring topology laser links."""
    for raan, sats in planes_dict.items():
        n = len(sats)
        for i, (sat_id, ma) in enumerate(sats):
            for peer_id in [sats[(i - 1) % n][0], sats[(i + 1) % n][0]]:
                add_peer_link(peer_links_map, sat_id, peer_id, "laser", True)


def add_cross_plane_links(planes_dict: dict, peer_links_map: dict) -> None:
    """Add adjacent-plane cross-links (circular, both directions)."""
    raans = sorted(planes_dict.keys())
    for i, raan_a in enumerate(raans):
        raan_b = raans[(i + 1) % len(raans)]
        for sat_id_a, ma_a in planes_dict[raan_a]:
            sat_id_b = min(planes_dict[raan_b], key=lambda x: ma_diff(x[1], ma_a))[0]
            add_peer_link(peer_links_map, sat_id_a, sat_id_b, "laser", False)
            add_peer_link(peer_links_map, sat_id_b, sat_id_a, "laser", False)


def add_cross_constellation_mw_links(opt_planes: dict, sar_planes: dict,
                                     peer_links_map: dict) -> None:
    """Add OPT-SAR microwave cross-constellation links."""
    sar_raans = sorted(sar_planes.keys())
    for raan_opt, sats_opt in opt_planes.items():
        closest_sar_raan = min(sar_raans, key=lambda r: raan_diff(raan_opt, r))
        for sat_id_opt, ma_opt in sats_opt:
            sat_id_sar = min(
                sar_planes[closest_sar_raan],
                key=lambda x: ma_diff(x[1], ma_opt)
            )[0]
            add_peer_link(peer_links_map, sat_id_opt, sat_id_sar, "microwave", False)
            add_peer_link(peer_links_map, sat_id_sar, sat_id_opt, "microwave", False)


def main() -> None:
    print(f"Reading {INPUT_PATH}")
    with open(INPUT_PATH, encoding='utf-8') as f:
        scenario = json.load(f)

    # Build satellite plane lookup tables
    opt_planes: dict = {}
    sar_planes: dict = {}

    for sat in scenario['satellites']:
        sat_id = sat['id']
        raan = sat['orbit']['raan']
        ma = sat['orbit']['mean_anomaly']
        if sat_id.startswith('OPT'):
            opt_planes.setdefault(raan, []).append((sat_id, ma))
        elif sat_id.startswith('SAR'):
            sar_planes.setdefault(raan, []).append((sat_id, ma))

    for raan in opt_planes:
        opt_planes[raan].sort(key=lambda x: x[1])
    for raan in sar_planes:
        sar_planes[raan].sort(key=lambda x: x[1])

    print(f"  OPT planes: {sorted(opt_planes.keys())}")
    print(f"  SAR planes: {sorted(sar_planes.keys())}")

    # Build peer links
    peer_links_map: dict = {}
    add_ring_links(opt_planes, peer_links_map)
    add_ring_links(sar_planes, peer_links_map)
    add_cross_plane_links(opt_planes, peer_links_map)
    add_cross_plane_links(sar_planes, peer_links_map)
    add_cross_constellation_mw_links(opt_planes, sar_planes, peer_links_map)

    # ISL hardware config
    laser_config = {
        "wavelength_nm": 1550,
        "transmit_power_w": 2.0,
        "transmit_aperture_m": 0.1,
        "receive_aperture_m": 0.1,
        "beam_divergence_urad": 5.0,
        "max_range_km": 7000,
        "acquisition_time_s": 30,
        "coarse_tracking_time_s": 5,
        "fine_tracking_time_s": 2,
        "tracking_accuracy_urad": 2.0,
        "point_ahead_urad": 30.0,
        "min_link_margin_db": 3.0,
        "snr_required_db": 20.0
    }
    microwave_config = {
        "frequency_ghz": 26.0,
        "transmit_power_w": 10.0,
        "antenna_gain_dbi": 30.0,
        "max_beam_count": 4,
        "scan_angle_deg": 60.0,
        "max_range_km": 3500,
        "tdma_slots": 8,
        "gain_rolloff_db_per_deg": 0.067,
        "system_noise_temp_k": 1000.0,
        "snr_required_db": 15.0
    }

    # Build new scenario (deep copy to preserve all original fields)
    new_scenario = copy.deepcopy(scenario)
    new_scenario['name'] = '大规模星座任务规划实验场景（含星间网络）'
    new_scenario['version'] = '2.3+isl'
    new_scenario['description'] = (
        '60颗卫星(30光学+30SAR) vs 1000目标，总观测需求2638次，24小时规划周期，'
        '配置天链式中继卫星，支持混合数传；新增星间链路（ISL）能力，支持多跳中继数传路由'
    )

    # Inject ISL config into each satellite
    for sat in new_scenario['satellites']:
        sat_id = sat['id']
        sat['capabilities']['isl'] = {
            "enabled": True,
            "laser": copy.deepcopy(laser_config),
            "microwave": copy.deepcopy(microwave_config),
            "peer_links": peer_links_map.get(sat_id, []),
            "max_simultaneous_laser": 2,
            "link_selection": "laser_preferred"
        }

    # Write output
    print(f"Writing {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(new_scenario, f, ensure_ascii=False, indent=2)

    # Verify
    with open(OUTPUT_PATH, encoding='utf-8') as f:
        verify = json.load(f)

    total_sats = len(verify['satellites'])
    isl_enabled = sum(
        1 for s in verify['satellites']
        if s['capabilities'].get('isl', {}).get('enabled', False)
    )
    sample = verify['satellites'][0]
    peer_count_sample = len(sample['capabilities']['isl']['peer_links'])

    print(f"\nVerification:")
    print(f"  Total satellites: {total_sats}")
    print(f"  ISL-enabled satellites: {isl_enabled}")
    print(f"  First satellite ({sample['id']}) peer_links: {peer_count_sample}")
    print(f"  Version: {verify['version']}")
    print(f"  Name: {verify['name']}")

    # Show a sample
    print(f"\nOPT-01 peer links:")
    opt01 = next(s for s in verify['satellites'] if s['id'] == 'OPT-01')
    for p in opt01['capabilities']['isl']['peer_links']:
        print(f"  -> {p['peer_satellite_id']} ({p['link_type']}, preferred={p['preferred']})")

    print(f"\nDone! File size: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")


if __name__ == '__main__':
    main()
