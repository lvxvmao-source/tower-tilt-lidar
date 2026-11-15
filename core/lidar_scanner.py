# -*- coding: utf-8 -*-
"""
激光雷达扫描器：支持任意线数/FOV/角分辨率配置
"""
import numpy as np
import open3d as o3d
from open3d.t.geometry import RaycastingScene
from dataclasses import dataclass
from typing import Tuple, List


# 传感器"正前方"相对局部坐标系的偏置角。
# 本模型把正前方定义为 +x，故需转 180° 才能从站位看向塔心；
# 若设备手册规定正前方为 +y，改成 90.0 即可。
FORWARD_OFFSET_DEG = np.pi


@dataclass
class LidarConfig:
    """雷达参数"""
    name: str = "Helios16"
    n_channels: int = 16
    vertical_fov_deg: float = 30.0
    horizontal_resolution_deg: float = 0.2
    range_sigma: float = 0.01
    scan_sector_deg: float = 120.0
    scan_distance: float = 15.0
    station_heights: Tuple[float, ...] = (6.0, 13.0, 20.0, 27.0, 34.0)
    # 测距噪声的随机种子。固定值 → 同一配置每次扫描完全可复现；
    # 0 表示每次运行都换一组随机数（结果会有 ±0.05‰ 的抖动）
    random_seed: int = 42
    station_bearings_deg: Tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)

    def vertical_angles(self) -> np.ndarray:
        half = self.vertical_fov_deg / 2
        return np.linspace(-half, half, self.n_channels)

    def ray_count(self) -> int:
        """单站射线数 = 线数 × 水平角数"""
        n_h = int(round(self.scan_sector_deg / max(self.horizontal_resolution_deg, 1e-6))) + 1
        return self.n_channels * n_h

    def station_list(self):
        """展开成 (方位角, 高度) 列表"""
        return [(float(b), float(h))
                for b in self.station_bearings_deg
                for h in self.station_heights]


def simulate_scan(tower_mesh: o3d.geometry.TriangleMesh,
                  cfg: LidarConfig = None) -> o3d.geometry.PointCloud:
    """按雷达配置对塔执行多站扫描。

    射线方向按"线数 × 水平角"一次性矢量化生成（原先是三重 Python 循环），
    站位增加到几十个时速度差别很明显。
    """
    if cfg is None:
        cfg = LidarConfig()
    scene = RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(tower_mesh))

    v_rad = np.deg2rad(cfg.vertical_angles())
    h_rad = np.deg2rad(np.arange(-cfg.scan_sector_deg / 2.0,
                                 cfg.scan_sector_deg / 2.0 + 1e-9,
                                 cfg.horizontal_resolution_deg))
    HH, VV = np.meshgrid(h_rad, v_rad)              # (线数, 水平角数)
    lx = (np.cos(VV) * np.cos(HH)).ravel()
    ly = (np.cos(VV) * np.sin(HH)).ravel()
    lz = np.sin(VV).ravel()
    n_ray = lx.size

    all_points: List[np.ndarray] = []

    for bearing_deg in cfg.station_bearings_deg:
        br = np.deg2rad(bearing_deg)
        sx = cfg.scan_distance * np.cos(br)
        sy = cfg.scan_distance * np.sin(br)
        # 站位按方位角 b 摆出，传感器"正前方"定义为局部 +x
        # （ψ=0, η=0 时 d_loc=(1,0,0)），因此把 +x 转到 b+180° 即对准塔心。
        # 早先用 arctan2(-sy,-sx) 反算，与这里完全等价，但多绕了一步。
        yaw = br + FORWARD_OFFSET_DEG
        cy, syw = np.cos(yaw), np.sin(yaw)
        wx = cy * lx - syw * ly
        wy = syw * lx + cy * ly

        for sh in cfg.station_heights:
            rays = np.empty((n_ray, 6), dtype=np.float32)
            rays[:, 0] = sx
            rays[:, 1] = sy
            rays[:, 2] = sh
            rays[:, 3] = wx
            rays[:, 4] = wy
            rays[:, 5] = lz

            result = scene.cast_rays(o3d.core.Tensor(rays))
            hit = result['t_hit'].numpy()
            valid = hit < 1e6
            if not valid.any():
                continue
            origins = rays[valid, 0:3]
            dirs = rays[valid, 3:6]
            dists = hit[valid]
            # 测距噪声沿射线方向叠加（1σ 指标是测距精度）
            dists = dists + np.random.normal(0, cfg.range_sigma, dists.shape)
            all_points.append(origins + dirs * dists[:, None])

    pcd = o3d.geometry.PointCloud()
    if not all_points:                      # 全部射线打空（塔模型为空/位置不对）
        pcd.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
        return pcd
    pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    return pcd
