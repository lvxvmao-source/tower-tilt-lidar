# -*- coding: utf-8 -*-
"""核心算法自检：刚体 / 弯曲 / 差异沉降 三种倾斜模型下的精度回归

注意：这里比较的是**标准口径**的 α（塔顶偏移/塔高），
与界面、DL/T 741 判定所用口径一致。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
np.random.seed(42)
from core.tower_model import TowerConfig, build_tower, true_indicators
from core.lidar_scanner import LidarConfig, simulate_scan
from core.tilt_solver import SolverConfig, solve


def run(cfg):
    ta, td, _tip, _tx, _ty = true_indicators(cfg)
    scfg = SolverConfig()
    scfg.tower_height = cfg.total_height
    pcd = simulate_scan(build_tower(cfg), LidarConfig(horizontal_resolution_deg=0.2))
    r, _ = solve(pcd, scfg)
    return ta, td, r


print("【1】刚体旋转：10 个标准工况（α 单位 ‰）")
CASES = [(3.0, 1.0), (2.0, 0.8), (1.0, 0.5), (0.5, 0.2), (0.2, 0.05),
         (0.1, 0.0), (0.0, 0.0), (0.0, 3.0), (3.0, 0.0), (-2.0, -0.8)]
print(f"{'输入X':<7}{'输入Y':<7}{'真α':>8}{'测α':>8}{'误差':>7}   {'真δ':>7}{'测δ':>7}{'误差':>7}")
print("-" * 66)
ea, ed = [], []
for tx, ty in CASES:
    cfg = TowerConfig(tilt_deg_x=tx, tilt_deg_y=ty)
    ta, td, r = run(cfg)
    err = r.alpha_standard_permille - ta
    ea.append(abs(err)); ed.append(abs(r.foundation_tilt_permille - td))
    print(f"{tx:<7.2f}{ty:<7.2f}{ta:8.2f}{r.alpha_standard_permille:8.2f}{err:+7.2f}   "
          f"{td:7.2f}{r.foundation_tilt_permille:7.2f}"
          f"{r.foundation_tilt_permille - td:+7.2f}")
print("-" * 66)
print(f"α 平均绝对误差 {np.mean(ea):.2f}‰  最大 {np.max(ea):.2f}‰")
print(f"δ 平均绝对误差 {np.mean(ed):.2f}‰  最大 {np.max(ed):.2f}‰")

print("\n【2】进阶模型：塔身弯曲 + 四腿差异沉降")
BENT = [
    ("纯刚体 (1.0, 0.5)", dict(tilt_deg_x=1.0, tilt_deg_y=0.5)),
    ("弯曲 15%", dict(tilt_deg_x=1.0, tilt_deg_y=0.5, bow_ratio=0.15)),
    ("弯曲 30%", dict(tilt_deg_x=1.0, tilt_deg_y=0.5, bow_ratio=0.30)),
    ("弯曲 15% @ (3.0, 1.0)", dict(tilt_deg_x=3.0, tilt_deg_y=1.0, bow_ratio=0.15)),
    ("沉降 NE=100 / SW=-60 mm",
     dict(use_settlement=True, settlement_ne=0.10, settlement_nw=0.02,
          settlement_se=0.02, settlement_sw=-0.06)),
    ("沉降 + 弯曲 20%",
     dict(use_settlement=True, settlement_ne=0.10, settlement_nw=0.02,
          settlement_se=0.02, settlement_sw=-0.06, bow_ratio=0.20)),
]
print(f"{'工况':<26}{'真α':>8}{'测α':>8}{'误差':>7}{'置信':>7}")
print("-" * 58)
e2 = []
for name, kw in BENT:
    cfg = TowerConfig(**kw)
    ta, td, r = run(cfg)
    err = r.alpha_standard_permille - ta
    e2.append(abs(err))
    print(f"{name:<26}{ta:8.2f}{r.alpha_standard_permille:8.2f}{err:+7.2f}{r.confidence:7.2f}")
print("-" * 58)
print(f"平均绝对误差 {np.mean(e2):.2f}‰  最大 {np.max(e2):.2f}‰")
