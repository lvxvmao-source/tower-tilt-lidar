# -*- coding: utf-8 -*-
"""
横担几何健康分析

原先横担点云在解算前被整块剔除，只当作"干扰"。实际上它是塔上受力最集中、
最容易出现塔材丢失/断裂/偏斜的部件，可以反演出多项健康指标。

坐标约定：高度一律用**沿塔轴**的坐标
    t = (z + dx·vx + dy·vy) / |v|
而不是世界 z。否则塔体一倾斜，4m 长的横担两端天然就有上百毫米高差。

指标可信度（受扫描几何约束，务必如实标注）：
  ✔ 臂长 / 臂长对称性 / 端部横向偏移 —— 水平量，任意姿态下都可靠
  ✘ 端部垂直下垂 —— 需要看到横担上表面或下表面，当前站位（6/13/20/27/34m，
     80% 横担高度上没有正上或正下视点）会造成大面积遮挡，实测不可靠。
     若要测下垂，需在横担高度附近增设扫描站位。
"""
from dataclasses import dataclass, field
from typing import List, Optional, Callable

import numpy as np


@dataclass
class CrossarmSide:
    arm_len: float = 0.0          # 臂长 m
    tip_offset: float = 0.0       # 端部横向（y 向）偏移 mm
    n_points: int = 0
    status: str = "未知"          # 正常 / 偏短 / 无有效点


@dataclass
class CrossarmReport:
    height: float
    design_len: float
    left: CrossarmSide = field(default_factory=CrossarmSide)
    right: CrossarmSide = field(default_factory=CrossarmSide)

    @property
    def asymmetry_mm(self) -> float:
        return abs(self.left.arm_len - self.right.arm_len) * 1000.0

    @property
    def ok(self) -> bool:
        return self.left.status == "正常" and self.right.status == "正常"


def _axis_frame(pts: np.ndarray, params: np.ndarray):
    px, py, vx, vy, h0, k, ax, ay = params
    z = pts[:, 2]
    dx = pts[:, 0] - (px + vx * z + ax * z * z)
    dy = pts[:, 1] - (py + vy * z + ay * z * z)
    nv = np.sqrt(vx * vx + vy * vy + 1.0)
    t = (z + dx * vx + dy * vy) / nv
    return dx, dy, t


def analyze_crossarms(pts: np.ndarray,
                      params: Optional[np.ndarray],
                      crossarm_levels,
                      body_half_width_at: Callable[[float], float],
                      design_len: float,
                      band: float = 0.5,
                      tol_len: float = 0.20) -> List[CrossarmReport]:
    """逐层提取横担健康指标。

    pts               含横担的塔身点云（截断后、未剔除横担）
    params            精化后的塔轴参数 [px,py,vx,vy,h0,k,ax,ay]
    crossarm_levels   各横担设计高度（沿塔轴）
    design_len        设计臂长
    tol_len           判定"偏短"的容差（m）
    """
    reports: List[CrossarmReport] = []
    if params is None:
        return reports
    pts = np.asarray(pts, dtype=float)
    if len(pts) == 0:
        return reports

    dx, dy, t = _axis_frame(pts, params)
    adx = np.abs(dx)

    for h in crossarm_levels:
        half_w = float(body_half_width_at(h))
        rep = CrossarmReport(height=float(h), design_len=float(design_len))
        for side, attr in ((-1, "left"), (1, "right")):
            m = ((np.abs(t - h) < band) &
                 (np.sign(dx) == side) &
                 (adx > half_w + 0.25) &
                 (np.abs(dy) < 0.45))
            n = int(m.sum())
            s = CrossarmSide(n_points=n)
            # 判定"缺失"看的是**有没有伸出点**，不是点数多少：
            # 采样稀疏时点数会很少但臂仍在，用点数阈值会造成误报。
            exists = False
            reach = 0.0
            if n >= 3:
                reach = (float(np.percentile(adx[m], 99)) if n >= 12
                         else float(adx[m].max()))
                exists = reach > half_w + 0.5
            if exists:
                s.arm_len = max(reach - half_w, 0.0)
                tip = m & (adx > reach - 0.5)
                if int(tip.sum()) >= 5:
                    s.tip_offset = float(np.median(dy[tip])) * 1000.0
                if s.arm_len < design_len * 0.6:
                    s.status = "无有效点"
                elif s.arm_len < design_len - tol_len:
                    s.status = "偏短"
                else:
                    s.status = "正常"
            else:
                s.arm_len = 0.0
                s.status = "无有效点"
            setattr(rep, attr, s)
        reports.append(rep)
    return reports
