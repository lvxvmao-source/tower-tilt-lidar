# -*- coding: utf-8 -*-
"""
杆塔倾斜测量算法（实验脚本入口）

说明：本模块原先是一份与 core/tilt_solver.py 并列的独立实现，
两边长期各改各的，导致 GUI 与论文实验跑的不是同一套代码。
现在统一为对 core.tilt_solver 的薄封装，保证实验结果与界面结果完全一致。

链路：
  预处理 → 高度截断 → PCA定横担方位+逐高度自适应包络剔除
      → 逐窗口对称中心提取中轴点(带置信度) → RANSAC + Huber 选权迭代直线拟合
      → 倾斜解算
"""
import numpy as np
import open3d as o3d

from core.tilt_solver import (          # noqa: F401  (对外保留原有名字)
    SolverConfig,
    TiltResult,
    preprocess,
    crop_height,
    remove_crossarms_pca,
    extract_axis_weighted,
    fit_line_robust,
    solve,
)

DEFAULT_CONFIG = SolverConfig()


def estimate_tilt(pcd: o3d.geometry.PointCloud) -> TiltResult:
    """完整倾斜解算，返回 TiltResult（不返回中间几何体）"""
    result, _ = solve(pcd, DEFAULT_CONFIG)
    return result


if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    np.random.seed(42)

    pcd = o3d.io.read_point_cloud(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "demo_scan.ply")
    )
    print(f"输入点云点数：{len(pcd.points)}")

    true_x, true_y = 0.5, 0.2
    true_total = np.rad2deg(np.arccos(
        np.cos(np.deg2rad(true_x)) * np.cos(np.deg2rad(true_y))
    ))

    result = estimate_tilt(pcd)
    print("\n===== 测量结果 vs 真值 =====")
    print(f"X方向倾斜角：测量 {result.tilt_x_deg:.4f}°  vs  真值 {true_x:.4f}°")
    print(f"Y方向倾斜角：测量 {result.tilt_y_deg:.4f}°  vs  真值 {true_y:.4f}°")
    print(f"总倾斜角：测量 {result.total_tilt_deg:.4f}°  vs  真值 {true_total:.4f}°")
    print(f"倾斜率：{result.tilt_rate_permille:.2f}‰")
    print(f"中轴点数：{result.num_axis_points}，内点数：{result.num_inliers}，"
          f"置信度：{result.confidence:.2f}")
