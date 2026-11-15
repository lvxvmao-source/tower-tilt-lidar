# -*- coding: utf-8 -*-
"""
塔模型模块：参数化生成 + 倾斜变形 + PLY导入

倾斜生成有两种模式（由 TowerConfig 决定）：
  1) 刚体旋转（默认，bow_ratio=0 且沉降为 0）：
     整塔绕塔基中心 O 刚性旋转，塔轴严格为直线。
  2) 差异沉降 + 塔身弯曲（进阶）：
     四根塔腿各给一个独立沉降量，由四个塔脚拟合出基础倾斜平面；
     塔身在此基础上再叠加一条沿倾斜方位的弯曲挠曲线，塔轴由直线变成曲线。
     弯曲形状取悬臂梁挠曲线 φ(t) = (3t² − t³)/2，t = z/H，
     保证塔基地面处位移和转角都为 0，塔顶位移最大。
"""
import os
from dataclasses import dataclass
from typing import List, Tuple, Callable

import numpy as np
import open3d as o3d

from core.frame import (CORNER_SIGNS, corner_settlement_from_tilt,
                        tilt_from_corner_settlement)


@dataclass
class TowerConfig:
    """塔几何参数"""
    total_height: float = 30.0
    body_bottom_width: float = 5.0
    body_top_width: float = 2.0
    body_start_height: float = 3.0
    leg_outreach: float = 1.5
    main_member_section: float = 0.2
    horizontal_spacing: float = 2.0
    crossarm_levels: tuple = (18.0, 22.0, 26.0)
    crossarm_length: float = 4.0
    crossarm_section: float = 0.4
    brace_thickness: float = 0.08

    # ---- 倾斜（真值）----
    # 真值倾斜角（度）。仅在四个沉降量全为 0 时生效。
    tilt_deg_x: float = 0.0
    tilt_deg_y: float = 0.0

    # 进阶模型：四腿差异沉降（米，下沉为正）
    # NW=(-X,+Y)  NE=(+X,+Y)  SE=(+X,-Y)  SW=(-X,-Y)
    settlement_nw: float = 0.0
    settlement_ne: float = 0.0
    settlement_se: float = 0.0
    settlement_sw: float = 0.0
    # 强制使用沉降模型（即使四个沉降量都是 0，也不回退到 tilt_deg_x/y）
    use_settlement: bool = False

    # 塔身附加弯曲比例：0=纯刚体，0.2=塔顶在刚体倾斜基础上再多偏 20%
    bow_ratio: float = 0.0

    # 局部折角（模拟某一高度处塔材失稳/损伤导致的局部弯曲）
    # local_bend_height=0 表示不启用；local_bend_slope 单位 ‰（该高度以上轴线斜率突增量）
    local_bend_height: float = 0.0
    local_bend_slope: float = 0.0

    # 横担端部下垂（m，正值=右侧 +X 横担端部下沉），用于模拟横担挠曲/塔材锈蚀工况
    crossarm_droop: float = 0.0
    # 横担缺失（模拟塔材丢失/断裂）：0=完整，+1=右侧缺失，-1=左侧缺失
    crossarm_missing_side: int = 0

    def has_settlement(self) -> bool:
        if self.use_settlement:
            return True
        return max(abs(self.settlement_nw), abs(self.settlement_ne),
                   abs(self.settlement_se), abs(self.settlement_sw)) > 1e-9


# ---------------------------------------------------------------- 构件生成

def _cylinder(p1, p2, radius):
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return o3d.geometry.TriangleMesh()
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=8)
    z_axis = np.array([0.0, 0.0, 1.0])
    direction_unit = direction / length
    v = np.cross(z_axis, direction_unit)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, direction_unit)
    if s < 1e-8:
        if c < 0:
            cyl.rotate(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), center=np.array([0, 0, 0]))
    else:
        vx, vy, vz = v
        skew = np.array([[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]])
        R = np.eye(3) + skew + skew @ skew * ((1 - c) / (s * s))
        cyl.rotate(R, center=np.array([0, 0, 0]))
    mid = (p1 + p2) / 2
    cyl.translate(mid)
    cyl.compute_vertex_normals()
    return cyl


def _body_width_at(height, p: TowerConfig):
    if height < p.body_start_height:
        return p.body_bottom_width
    if height >= p.total_height:
        return p.body_top_width
    t = (height - p.body_start_height) / (p.total_height - p.body_start_height)
    return p.body_bottom_width + t * (p.body_top_width - p.body_bottom_width)


def _body_corners_at(height, p: TowerConfig):
    w = _body_width_at(height, p)
    hw = w / 2
    return [
        np.array([-hw, -hw, height]),
        np.array([hw, -hw, height]),
        np.array([hw, hw, height]),
        np.array([-hw, hw, height]),
    ]


def _leg_corners_at(height, p: TowerConfig):
    """塔腿角点：0~body_start_height 从基础外伸点线性收到塔身底宽"""
    if height <= 1e-9:
        c = p.body_bottom_width / 2 + p.leg_outreach
        return [
            np.array([-c, -c, 0.0]),
            np.array([c, -c, 0.0]),
            np.array([c, c, 0.0]),
            np.array([-c, c, 0.0]),
        ]
    if height < p.body_start_height:
        t = height / p.body_start_height
        base = _leg_corners_at(0.0, p)
        top = _body_corners_at(p.body_start_height, p)
        return [base[i] * (1 - t) + top[i] * t for i in range(4)]
    return _body_corners_at(height, p)


def _level_heights(cfg: TowerConfig) -> List[float]:
    """塔身分层高度：0、塔腿顶、以后按 horizontal_spacing 等分到塔顶"""
    spacing = max(0.5, cfg.horizontal_spacing)
    levels = [0.0, cfg.body_start_height]
    z = cfg.body_start_height
    while z + spacing < cfg.total_height - 1e-6:
        z += spacing
        levels.append(z)
    if levels[-1] < cfg.total_height - 1e-6:
        levels.append(cfg.total_height)
    return levels


def _resolve_crossarm_levels(cfg: TowerConfig) -> tuple:
    """横担高度按总高等比缩放，并剔除超出塔高的层。

    crossarm_levels 的默认值 (18, 22, 26) 是按 30m 标准塔给的；
    用户把总高改成 20m 或 45m 时，横担必须跟着缩放，否则会悬在塔顶外面。
    """
    scale = cfg.total_height / 30.0
    levels = tuple(h * scale for h in cfg.crossarm_levels)
    return tuple(h for h in levels if 0.0 < h < cfg.total_height - 0.5)


# ---------------------------------------------------------------- 倾斜变换

def base_corner_offset(cfg: TowerConfig) -> float:
    """塔脚相对塔位中心的水平距离（米）"""
    return cfg.body_bottom_width / 2 + cfg.leg_outreach


def base_tilt_deg(cfg: TowerConfig) -> Tuple[float, float]:
    """塔的总倾角（度）= 塔身相对基础的附加倾斜 + 基础沉降造成的倾斜。

    - `tilt_deg_x/y` 是**塔身相对基础的附加倾斜**（施工偏差、外力等）
    - 四角沉降反算出的是**基础倾角**
    - 两者矢量叠加，小角度下直接逐分量相加（20‰ 量级误差约 0.02‰）

    坐标约定见 core/frame.py：+x = 南、+y = 东、+z = 上。
    四角坐标：NW(−c,−c)  NE(−c,+c)  SE(+c,+c)  SW(+c,−c)。
    """
    tx = float(cfg.tilt_deg_x)
    ty = float(cfg.tilt_deg_y)
    if not cfg.has_settlement():
        return tx, ty

    c = base_corner_offset(cfg)
    d = {"NW": cfg.settlement_nw, "NE": cfg.settlement_ne,
         "SE": cfg.settlement_se, "SW": cfg.settlement_sw}
    tan_x, tan_y = tilt_from_corner_settlement(c, d)
    tx += float(np.rad2deg(np.arctan(tan_x)))
    ty += float(np.rad2deg(np.arctan(tan_y)))
    return tx, ty


def settlement_tilt_deg(cfg: TowerConfig) -> Tuple[float, float]:
    """仅由四角沉降反算出的基础倾角（度），用于界面分项显示"""
    if not cfg.has_settlement():
        return 0.0, 0.0
    c = base_corner_offset(cfg)
    d = {"NW": cfg.settlement_nw, "NE": cfg.settlement_ne,
         "SE": cfg.settlement_se, "SW": cfg.settlement_sw}
    tan_x, tan_y = tilt_from_corner_settlement(c, d)
    return float(np.rad2deg(np.arctan(tan_x))), float(np.rad2deg(np.arctan(tan_y)))


def settlements_from_tilt(cfg: TowerConfig,
                          tilt_x_deg: float = None,
                          tilt_y_deg: float = None) -> Tuple[float, float, float, float]:
    """由倾斜角反算四腿沉降量（NE, SE, SW, NW），用于界面上一键换算"""
    tx = cfg.tilt_deg_x if tilt_x_deg is None else tilt_x_deg
    ty = cfg.tilt_deg_y if tilt_y_deg is None else tilt_y_deg
    c = base_corner_offset(cfg)
    d = corner_settlement_from_tilt(c, np.tan(np.deg2rad(tx)), np.tan(np.deg2rad(ty)))
    return d["NE"], d["SE"], d["SW"], d["NW"]


def _tilt_matrix(cfg: TowerConfig) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (旋转矩阵 R, 塔轴方向 v)。

    tilt_deg_x 表示"塔顶沿 +X 方向偏移"，对应绕 Y 轴旋转；
    tilt_deg_y 表示"塔顶沿 +Y 方向偏移"，对应绕 X 轴负向旋转。
    塔轴 v = (sinθx, sinθy·cosθx, cosθx·cosθy)，
    与解算端 tilt_x=atan2(|v_x|,|v_z|)、tilt_y=atan2(|v_y|,|v_z|) 自洽。
    """
    tx_deg, ty_deg = base_tilt_deg(cfg)
    theta_x = np.deg2rad(tx_deg)
    theta_y = np.deg2rad(ty_deg)
    Ry = np.array([[np.cos(theta_x), 0, np.sin(theta_x)],
                   [0, 1, 0],
                   [-np.sin(theta_x), 0, np.cos(theta_x)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_y), np.sin(theta_y)],
                   [0, -np.sin(theta_y), np.cos(theta_y)]])
    R = Rx @ Ry
    v = R @ np.array([0.0, 0.0, 1.0])
    return R, v


def _axial_offset(cfg: TowerConfig, v: np.ndarray, z: np.ndarray,
                  trans_width: float = 1.6):
    """沿倾斜方位的附加水平位移（整体弯曲 + 局部折角），单位 m。

    返回 (n, 2)，对应 x/y 增量。塔基地面处位移与斜率都为 0。
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    horiz = np.array([v[0], v[1]], dtype=float)
    nh = float(np.linalg.norm(horiz))
    if nh < 1e-12:
        direction = np.array([1.0, 0.0])
        tan_total = 0.0
    else:
        direction = horiz / nh
        tan_total = nh / max(abs(v[2]), 1e-9)

    u = np.zeros_like(z)

    # 整体弯曲：悬臂挠曲线 φ(t)=(3t²−t³)/2
    if cfg.bow_ratio > 0.0:
        t = np.clip(z / max(cfg.total_height, 1e-6), 0.0, 1.0)
        u += cfg.bow_ratio * cfg.total_height * tan_total * 0.5 * (3 * t ** 2 - t ** 3)

    # 局部折角：h 处斜率突增 tanβ，用宽度 trans_width 的平滑过渡保证 C1 连续
    if cfg.local_bend_height > 0.0 and abs(cfg.local_bend_slope) > 0.0:
        beta = cfg.local_bend_slope / 1000.0          # ‰ -> tan
        s = z - cfg.local_bend_height
        w = max(trans_width, 1e-3)
        ramp = np.where(s <= 0.0, 0.0,
                        np.where(s >= w, s - w / 2.0, s * s / (2.0 * w)))
        u += beta * ramp

    return np.outer(u, direction)


def _bow_displacement(cfg: TowerConfig, v: np.ndarray, z: np.ndarray):
    """兼容旧名：塔身附加水平位移"""
    return _axial_offset(cfg, v, z)


def tower_axis(cfg: TowerConfig) -> Callable[[float], np.ndarray]:
    """返回塔轴（各高度截面中心）的解析函数 z -> (x, y, z)"""
    R, v = _tilt_matrix(cfg)
    v = v / np.linalg.norm(v)

    def axis(z: float) -> np.ndarray:
        z = float(z)
        t = z / max(v[2], 1e-9)          # 沿 v 的参数，使世界 z 坐标等于 z
        p = v * t
        off = _axial_offset(cfg, v, np.array([z]))[0]
        return np.array([p[0] + off[0], p[1] + off[1], p[2]])

    return axis


def true_tilt(cfg: TowerConfig,
              z_min: float = None, z_max: float = None,
              n_samples: int = 60) -> Tuple[float, float, float, float]:
    """解析真值：在解算高度区间内对塔轴做最小二乘直线拟合，返回
    (tilt_x_deg, tilt_y_deg, total_deg, tilt_rate_permille)。

    弯曲塔的"倾斜角"本身没有唯一答案，必须和算法采用同一口径
    （同一高度区间、同一拟合方式）才能比较误差。
    """
    z0 = 0.13 * cfg.total_height if z_min is None else float(z_min)
    z1 = 0.9 * cfg.total_height if z_max is None else float(z_max)
    if z1 <= z0:
        z0, z1 = sorted((z0, z1))

    axis = tower_axis(cfg)
    zs = np.linspace(z0, z1, n_samples)
    pts = np.array([axis(z) for z in zs])

    center = pts.mean(axis=0)
    try:
        _, _, vt = np.linalg.svd(pts - center, full_matrices=False)
        v = vt[0]
    except np.linalg.LinAlgError:
        v = np.array([0.0, 0.0, 1.0])
    if v[2] < 0:
        v = -v
    v = v / np.linalg.norm(v)

    total_rad = np.arccos(np.clip(abs(v[2]), 0.0, 1.0))
    tx = np.rad2deg(np.arctan2(abs(v[0]), abs(v[2])))
    ty = np.rad2deg(np.arctan2(abs(v[1]), abs(v[2])))
    return float(tx), float(ty), float(np.rad2deg(total_rad)), float(np.tan(total_rad) * 1000.0)


# ---------------------------------------------------------------- 主生成

def _arm(root, side: int, arm_len: float, arm_h: float, droop: float = 0.0):
    """一根横担：从 root 出发沿 side 方向伸出 arm_len，端部可下垂 droop"""
    box = o3d.geometry.TriangleMesh.create_box(width=arm_len, height=arm_h, depth=arm_h)
    box.translate(np.array([-arm_len / 2, -arm_h / 2, -arm_h / 2]))
    box.translate(np.array([side * arm_len / 2, 0.0, 0.0]))     # 相对根部摆放
    if abs(droop) > 1e-9:
        phi = np.arctan2(droop, arm_len)
        c, s = np.cos(side * phi), np.sin(side * phi)
        R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        box.rotate(R, center=np.zeros(3))                       # 绕根部转动
    box.translate(np.asarray(root, dtype=float))
    box.compute_vertex_normals()
    return box


def build_tower(cfg: TowerConfig) -> o3d.geometry.TriangleMesh:
    """生成参数化角钢塔网格（含基础差异沉降与塔身弯曲）"""
    meshes: List[o3d.geometry.TriangleMesh] = []
    s = cfg.main_member_section
    levels = _level_heights(cfg)

    # ---- 1. 塔腿 + 塔身主角钢：按分层高度拆成多段 ----
    # 必须拆段，否则整根主材只有上下两个顶点，弯曲变形时无法随中间横隔一起变形，
    # 会在视觉上穿出横隔、也会让弯曲挠曲线失效。
    for k in range(len(levels) - 1):
        z0, z1 = levels[k], levels[k + 1]
        c0 = _leg_corners_at(z0, cfg)
        c1 = _leg_corners_at(z1, cfg)
        for i in range(4):
            meshes.append(_cylinder(c0[i], c1[i], radius=s / 2))

    # ---- 2. 水平横隔材 ----
    for h in levels[1:]:
        corners = _body_corners_at(h, cfg)
        for i in range(4):
            meshes.append(_cylinder(corners[i], corners[(i + 1) % 4],
                                    radius=cfg.brace_thickness))

    # ---- 3. 斜腹材：相邻两层之间 X 形交叉 ----
    for k in range(len(levels) - 2):
        c0 = _body_corners_at(levels[k + 1], cfg)
        c1 = _body_corners_at(levels[k + 2], cfg)
        for face in range(4):
            meshes.append(_cylinder(c0[face], c1[(face + 1) % 4],
                                    radius=cfg.brace_thickness * 0.7))
            meshes.append(_cylinder(c0[(face + 1) % 4], c1[face],
                                    radius=cfg.brace_thickness * 0.7))

    # ---- 4. 横担 ----
    for h_cross in _resolve_crossarm_levels(cfg):
        w_body = _body_width_at(h_cross, cfg)
        arm_len = cfg.crossarm_length
        arm_h = cfg.crossarm_section
        root_l = np.array([-w_body / 2, 0.0, h_cross])
        root_r = np.array([w_body / 2, 0.0, h_cross])
        if cfg.crossarm_missing_side != -1:
            meshes.append(_arm(root_l, -1, arm_len, arm_h))
        if cfg.crossarm_missing_side != 1:
            meshes.append(_arm(root_r, +1, arm_len, arm_h, droop=cfg.crossarm_droop))
        # 端部挂点（绝缘子串位置）
        for side, drp in ((-1, 0.0), (1, cfg.crossarm_droop)):
            if cfg.crossarm_missing_side == side:
                continue
            tip_x = side * (w_body / 2 + arm_len)
            tip_z = h_cross - drp
            meshes.append(_cylinder(np.array([tip_x, 0.0, tip_z]),
                                    np.array([tip_x, 0.0, tip_z - 1.2]), radius=0.05))

    tower_mesh = meshes[0]
    for m in meshes[1:]:
        tower_mesh += m

    # ---- 5. 变形：先整体刚体旋转，再叠加塔身弯曲位移 ----
    R, v = _tilt_matrix(cfg)
    verts = np.asarray(tower_mesh.vertices)
    verts = verts @ R.T
    bow = _bow_displacement(cfg, v, verts[:, 2])
    verts[:, :2] += bow
    tower_mesh.vertices = o3d.utility.Vector3dVector(verts)
    tower_mesh.compute_vertex_normals()
    return tower_mesh


def true_indicators(cfg: TowerConfig) -> Tuple[float, float, float, float, float]:
    """解析真值的**标准口径**指标。

    返回 (alpha_permille, delta_permille, tip_offset_mm, tilt_x_deg, tilt_y_deg)
      alpha = 塔顶水平偏移 / 塔高 ×1000   —— DL/T 741 的杆塔倾斜率
      delta = 塔基处轴线斜率 ×1000        —— DL/T 5219 的基础倾斜率
    """
    H = max(cfg.total_height, 1e-6)
    axis = tower_axis(cfg)
    p0 = axis(0.0)
    pH = axis(H)
    tip = float(np.hypot(pH[0] - p0[0], pH[1] - p0[1]))
    alpha = tip / H * 1000.0

    tx, ty = base_tilt_deg(cfg)
    delta = float(np.hypot(np.tan(np.deg2rad(tx)), np.tan(np.deg2rad(ty)))) * 1000.0
    return alpha, delta, tip * 1000.0, float(tx), float(ty)


def load_tower_from_ply(filepath: str) -> o3d.geometry.TriangleMesh:
    """从PLY文件导入塔网格"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    return o3d.io.read_triangle_mesh(filepath)
