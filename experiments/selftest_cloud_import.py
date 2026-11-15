# -*- coding: utf-8 -*-
"""实测点云导入自检：坐标偏移 / Y-up 导出 / 不同格式 的往返测量"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
np.random.seed(42)
import open3d as o3d

from core.tower_model import TowerConfig, build_tower, true_indicators, _resolve_crossarm_levels, _body_width_at
from core.lidar_scanner import LidarConfig, simulate_scan
from core.tilt_solver import SolverConfig, solve
from core.point_io import load_point_cloud, normalize_cloud, suggest_crop

TMP = os.path.join(tempfile.gettempdir(), "tower_cloud_test")
os.makedirs(TMP, exist_ok=True)

CASES = [
    ("原始坐标", dict(tilt_deg_x=1.0, tilt_deg_y=0.5), None),
    ("平移到工地坐标 +156m 高程", dict(tilt_deg_x=1.0, tilt_deg_y=0.5), (1200.0, 3400.0, 156.7)),
    ("Y-up 导出", dict(tilt_deg_x=2.0, tilt_deg_y=0.8), "yup"),
    ("大倾角 2°+1°", dict(tilt_deg_x=2.0, tilt_deg_y=1.0), (8000.0, -2000.0, 45.0)),
]

print(f"{'工况':<26}{'点数':>8}{'测高m':>7}{'真α‰':>8}{'测α‰':>8}{'误差‰':>7}{'真状态':>7}{'测状态':>7}")
print("-" * 84)

for name, kw, shift in CASES:
    cfg = TowerConfig(**kw)
    ta, td, _, _, _ = true_indicators(cfg)
    pcd = simulate_scan(build_tower(cfg), LidarConfig(horizontal_resolution_deg=0.2))

    pts = np.asarray(pcd.points).copy()
    if shift == "yup":                       # 模拟设备导出 Y-up
        R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)  # z -> y
        pts = pts @ R.T
    elif shift is not None:
        pts = pts + np.asarray(shift)

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts)
    path = os.path.join(TMP, f"scan_{abs(hash(name)) % 99999}.ply")
    o3d.io.write_point_cloud(path, out)

    loaded = load_point_cloud(path)
    cloud, height = normalize_cloud(loaded)
    scfg = SolverConfig()
    z_lo, z_hi = suggest_crop(height)
    scfg.z_min, scfg.z_max = z_lo, z_hi
    scfg.tower_height = height

    r, _ = solve(cloud, scfg)
    from core.standards import evaluate
    ev_t = evaluate(ta, td, height, "220kV")
    ev_m = evaluate(r.alpha_standard_permille, r.foundation_tilt_permille, height, "220kV")
    print(f"{name:<26}{len(cloud.points):8d}{height:7.1f}{ta:8.2f}"
          f"{r.alpha_standard_permille:8.2f}{r.alpha_standard_permille-ta:+7.2f}"
          f"{ev_t.grade:>8}{ev_m.grade:>8}")

print("-" * 84)
print(f"临时文件目录：{TMP}")
