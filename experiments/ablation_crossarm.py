# -*- coding: utf-8 -*-
"""横担点云影响实验：不剔除 / 当前剔除 / 不同容差 / 理想剔除"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
np.random.seed(42)
from core.tower_model import TowerConfig, build_tower, true_indicators, _resolve_crossarm_levels, _body_width_at
from core.lidar_scanner import LidarConfig, simulate_scan
import core.tilt_solver as ts

CASES = [(3.0, 1.0), (2.0, 0.8), (1.0, 0.5), (0.5, 0.2), (0.0, 0.0)]


def ideal_strip(pts, cfg, axis):
    """用真值几何精确剔除横担点：接近横担层且伸出塔身轮廓之外"""
    keep = np.ones(len(pts), dtype=bool)
    z = pts[:, 2]
    ax = np.array([axis(zz) for zz in z])
    dx = np.abs(pts[:, 0] - ax[:, 0])
    for h in _resolve_crossarm_levels(cfg):
        band = np.abs(z - h) < 0.28
        half = _body_width_at(h, cfg) / 2
        keep &= ~(band & (dx > half))
    return pts[keep]


def evaluate(pts, cfg, H):
    pcd = ts.crop_height(_to_pcd(pts), 4.0, 27.0)
    axis_pts, conf = ts.extract_axis_weighted(pcd, 2.0, 1.0)
    v0, p00, n, mc = ts.fit_line_robust(axis_pts, conf, ts._auto_threshold(axis_pts))
    ref = ts.refine_axis_prism(np.asarray(pcd.points), p00, v0, H, balance=True, balance_power=0.5)
    if ref is None:
        return np.nan, np.nan
    return ref["alpha"], ref["delta"]


def _to_pcd(arr):
    import open3d as o3d
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(arr)
    return p


print(f"{'工况':<14}{'':4}{'点数':>7}{'剔除%':>7}{'α误差‰':>9}{'δ误差‰':>9}")
print("-" * 56)
summary = {}
for tx, ty in CASES:
    cfg = TowerConfig(tilt_deg_x=tx, tilt_deg_y=ty)
    ta, td, _, _, _ = true_indicators(cfg)
    axis_fn = __import__("core.tower_model", fromlist=["tower_axis"]).tower_axis(cfg)
    raw = simulate_scan(build_tower(cfg), LidarConfig(horizontal_resolution_deg=0.2))
    pts_full = np.asarray(ts.crop_height(ts.preprocess(raw, 0.05), 4.0, 27.0).points)

    variants = []
    variants.append(("不剔除", pts_full))
    for tol in (0.3, 0.6, 1.0):
        p3 = ts.remove_crossarms_pca(_to_pcd(pts_full), tol=tol)
        variants.append((f"包络剔除 tol={tol}", np.asarray(p3.points)))
    variants.append(("真值理想剔除", ideal_strip(pts_full, cfg, axis_fn)))

    for label, pts in variants:
        a, d = evaluate(pts, cfg, cfg.total_height)
        share = 100 * (1 - len(pts) / len(pts_full))
        print(f"({tx},{ty})".ljust(14) + f"{label:<16}{len(pts):7d}{share:7.1f}"
              f"{a-ta:+9.2f}{d-td:+9.2f}")
        summary.setdefault(label, []).append((abs(a - ta), abs(d - td)))
    print()

print("=" * 56)
print(f"{'方案':<18}{'α 平均':>9}{'α 最大':>9}{'δ 平均':>9}{'δ 最大':>9}")
print("-" * 56)
for label, vals in summary.items():
    a = [v[0] for v in vals]; d = [v[1] for v in vals]
    print(f"{label:<18}{np.mean(a):9.2f}{np.max(a):9.2f}{np.mean(d):9.2f}{np.max(d):9.2f}")
