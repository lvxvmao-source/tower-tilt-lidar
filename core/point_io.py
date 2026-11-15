# -*- coding: utf-8 -*-
"""
实测点云导入与规范化

实测点云和仿真点云最大的差别不在数据本身，而在**坐标系**：
  1) 很多设备/软件导出的是 Y-up 或 X-up，不是 Z-up；
  2) 原点通常不在塔基，z 可能是 123.4m 这样的绝对高程；
  3) 点云里往往还包含地面、植被、导线等非塔身点。
算法只要求"Z 向上、塔基在 z=0"，所以导入后先做这两步规范化，
其余交给裁剪和抗差拟合。
"""
import os

import numpy as np
import open3d as o3d

SUPPORTED_EXT = (".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts", ".txt")
EXT_FILTER = ("点云 (*.ply *.pcd *.xyz *.xyzn *.xyzrgb *.pts *.txt);;"
              "PLY (*.ply);;PCD (*.pcd);;文本 (*.xyz *.txt);;所有文件 (*.*)")


def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    """读取点云文件；格式不支持或为空时抛出明确错误"""
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXT:
        raise ValueError(f"不支持的点云格式 {ext}，支持：{'、'.join(SUPPORTED_EXT)}")
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise ValueError("点云为空或读取失败（文件可能不是标准点云格式）")
    return pcd


def _extents(pts: np.ndarray) -> np.ndarray:
    return pts.max(axis=0) - pts.min(axis=0)


def align_z_up(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """把点云最长方向旋到 +Z。

    塔是细高结构，最长方向就是塔高方向。设备导出 Y-up 时如果不转，
    算法会把塔"横过来"算，结果完全错。
    """
    pts = np.asarray(pcd.points)
    axis = int(np.argmax(_extents(pts)))
    if axis == 2:
        return pcd
    if axis == 1:                       # Y-up -> Z-up
        R = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0]])
    else:                               # X-up -> Z-up
        R = np.array([[0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0]])
    out = o3d.geometry.PointCloud(pcd)
    out.rotate(R, center=np.zeros(3))
    return out


def normalize_cloud(pcd: o3d.geometry.PointCloud,
                    max_points: int = 1_200_000):
    """规范化：预降采样 → Z 轴朝上 → 塔基移到 z=0。

    返回 (pcd, height)。height 是规范化后的总高度（m），
    用作标准判定里的杆塔全高初值。
    """
    if len(pcd.points) > max_points:
        pts = np.asarray(pcd.points)
        span = float(np.max(_extents(pts)))
        voxel = max(span / 600.0, 0.01)     # 目标约 60 万~100 万点
        pcd = pcd.voxel_down_sample(voxel)

    pcd = align_z_up(pcd)

    pts = np.asarray(pcd.points).copy()
    z0 = float(pts[:, 2].min())
    pts[:, 2] -= z0
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts)
    height = float(pts[:, 2].max())
    return out, height


def suggest_crop(height: float, head_ratio: float = 0.9, base_ratio: float = 0.13):
    """按塔高给出截断区间的建议值（去掉塔腿段与塔头段）"""
    h = max(float(height), 1.0)
    return round(h * base_ratio, 1), round(h * head_ratio, 1)
