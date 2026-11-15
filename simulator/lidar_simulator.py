# -*- coding: utf-8 -*-
"""
激光雷达扫描模拟器（实验脚本入口）

原先是与 core/lidar_scanner.py 并列的第二份实现（射线生成、噪声、站位逻辑
几乎逐行重复）。现在统一为对 core.lidar_scanner 的薄封装，
参数名保持兼容：ScannerParams = LidarConfig。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.lidar_scanner import (        # noqa: F401
    LidarConfig,
    simulate_scan,
)

# 兼容旧名字
ScannerParams = LidarConfig


if __name__ == "__main__":
    import numpy as np
    import open3d as o3d

    from core.tower_model import TowerConfig, build_tower

    np.random.seed(42)
    tower = build_tower(TowerConfig(tilt_deg_x=0.5, tilt_deg_y=0.2))
    print(f"塔三角面数：{len(tower.triangles)}")

    pcd = simulate_scan(tower, ScannerParams())
    print(f"扫描得到点云点数：{len(pcd.points)}")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_pcd = os.path.join(out_dir, "demo_scan.ply")
    o3d.io.write_point_cloud(out_pcd, pcd)
    print(f"已保存点云：{out_pcd}")
