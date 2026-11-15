# -*- coding: utf-8 -*-
"""
可配置仿真实验：支持不同线数雷达和不同切片步长
雷达参数基于速腾聚创真实规格：
  - Helios 16: 30°垂直FOV, 16线均匀2°
  - Helios 32: 26°垂直FOV, 32线, 中间加密~0.5°
  - VLS-64类: 26°垂直FOV, 64线, 0.4°
  - Ruby 128: 40°垂直FOV, 128线, 0.1°
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import open3d as o3d
from dataclasses import dataclass, field
from typing import Tuple

from simulator.tower_builder import TowerParams, build_tower
import algorithms.tilt_estimator as te


# ========== 雷达参数配置（基于真实产品规格） ==========

@dataclass
class LidarConfig:
    name: str
    vertical_fov_deg: float        # 总垂直FOV
    n_channels: int                # 线数
    vert_resolution_deg: float     # 平均垂直角分辨率
    range_sigma: float = 0.01       # 测距噪声1σ
    horizontal_resolution_deg: float = 0.2

    def vertical_angles_deg(self) -> np.ndarray:
        """返回均匀分布的垂直角度列表"""
        half = self.vertical_fov_deg / 2
        return np.linspace(-half, half, self.n_channels)


# 四款主流雷达
LIDAR_PRESETS = {
    "16线 Helios 16": LidarConfig(
        name="Helios16", vertical_fov_deg=30.0, n_channels=16,
        vert_resolution_deg=2.0, range_sigma=0.01),
    "32线 Helios 32": LidarConfig(
        name="Helios32", vertical_fov_deg=26.0, n_channels=32,
        vert_resolution_deg=0.8125, range_sigma=0.01),
    "64线 VLS-64": LidarConfig(
        name="VLS64", vertical_fov_deg=26.0, n_channels=64,
        vert_resolution_deg=0.40625, range_sigma=0.008),
    "128线 Ruby": LidarConfig(
        name="Ruby128", vertical_fov_deg=40.0, n_channels=128,
        vert_resolution_deg=0.3125, range_sigma=0.005),
}


def simulate_scan_with_config(tower_mesh: o3d.geometry.TriangleMesh,
                              lidar: LidarConfig,
                              station_distance: float = 15.0,
                              station_bearings_deg: Tuple[float, ...] = (0, 90, 180, 270),
                              station_heights: Tuple[float, ...] = (6, 13, 20, 27, 34),
                              scan_sector_deg: float = 120.0) -> o3d.geometry.PointCloud:
    """按指定雷达配置扫描塔"""
    from open3d.t.geometry import RaycastingScene
    scene = RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(tower_mesh))

    vertical_angles = lidar.vertical_angles_deg()
    all_points = []

    for bearing_deg in station_bearings_deg:
        br = np.deg2rad(bearing_deg)
        sx = station_distance * np.cos(br)
        sy = station_distance * np.sin(br)
        for sh in station_heights:
            station_pos = np.array([sx, sy, sh], dtype=np.float32)
            yaw = np.arctan2(-sy, -sx)
            rays = []
            h_angles = np.deg2rad(np.arange(-scan_sector_deg/2, scan_sector_deg/2 + lidar.horizontal_resolution_deg, lidar.horizontal_resolution_deg))
            for v_deg in vertical_angles:
                v_rad = np.deg2rad(v_deg)
                for h_rad in h_angles:
                    dx_l = np.cos(v_rad)*np.cos(h_rad)
                    dy_l = np.cos(v_rad)*np.sin(h_rad)
                    dz_l = np.sin(v_rad)
                    cy, sy_ = np.cos(yaw), np.sin(yaw)
                    dx = cy*dx_l - sy_*dy_l
                    dy = sy_*dx_l + cy*dy_l
                    dz = dz_l
                    rays.append([sx, sy, sh, dx, dy, dz])
            rays = np.array(rays, dtype=np.float32)
            result = scene.cast_rays(o3d.core.Tensor(rays))
            hit = result['t_hit'].numpy()
            valid = hit < 1e6
            origins = rays[valid, 0:3]
            dirs = rays[valid, 3:6]
            dists = hit[valid]
            pts = origins + dirs * dists[:, None]
            pts += np.random.normal(0, lidar.range_sigma, pts.shape)
            all_points.append(pts)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    return pcd


def run_experiment(lidar_name: str, step_m: float = 1.0, tilt_x: float = 1.0, tilt_y: float = 0.5):
    """跑一个配置，返回(点数, 测量倾斜率, 真值倾斜率, 误差)"""
    np.random.seed(42)
    lidar = LIDAR_PRESETS[lidar_name]
    tower = build_tower(TowerParams(tilt_deg_x=tilt_x, tilt_deg_y=tilt_y))
    pcd = simulate_scan_with_config(tower, lidar)
    n_pts = len(pcd.points)

    # 解算（用指定切片步长）
    p1 = te.preprocess(pcd)
    p2 = te.crop_height(p1)
    p3 = te.remove_crossarms_pca(p2)
    axis, conf = te.extract_axis_weighted(p3, window_h=step_m*2, step=step_m)
    scatter = np.std(np.sqrt((axis[:,0]-axis[:,0].mean())**2 + (axis[:,1]-axis[:,1].mean())**2))
    thr = max(0.15, scatter*1.5)
    v, p0, nin, mc = te.fit_line_robust(axis, conf, distance_threshold=thr)
    meas_rate = np.tan(np.arccos(np.clip(abs(v[2]),-1,1)))*1000

    true_total = np.arccos(np.cos(np.deg2rad(tilt_x))*np.cos(np.deg2rad(tilt_y)))
    true_rate = np.tan(true_total)*1000
    return n_pts, meas_rate, true_rate, meas_rate - true_rate


if __name__ == "__main__":
    print("=" * 80)
    print("对比实验：不同线数雷达 + 不同切片步长（固定倾斜角 1.0°/0.5° ≈ 19.5‰）")
    print("=" * 80)

    # 实验1：不同线数，固定步长1.0m
    print("\n【实验1】不同线数雷达对比（步长=1.0m）")
    print(f"{'雷达':<18} {'点数':<8} {'测量‰':<10} {'真值‰':<10} {'误差‰':<8}")
    print("-" * 60)
    for name in LIDAR_PRESETS:
        n, meas, true_r, err = run_experiment(name, step_m=1.0)
        print(f"{name:<18} {n:<8} {meas:<10.2f} {true_r:<10.2f} {err:<+8.2f}")

    # 实验2：固定16线，不同切片步长
    print("\n【实验2】16线下不同切片步长对比")
    print(f"{'步长m':<8} {'测量‰':<10} {'误差‰':<8}")
    print("-" * 40)
    for step in [0.5, 0.75, 1.0, 1.5, 2.0]:
        n, meas, true_r, err = run_experiment("16线 Helios 16", step_m=step)
        print(f"{step:<8} {meas:<10.2f} {err:<+8.2f}")
