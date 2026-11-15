# -*- coding: utf-8 -*-
"""消融实验：两阶段 → 全局精化 → 分面分层均衡，各步骤对精度的贡献"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
np.random.seed(42)
from core.tower_model import TowerConfig, build_tower, true_indicators
from core.lidar_scanner import LidarConfig, simulate_scan
from core.tilt_solver import SolverConfig, solve

CASES = [(3.0, 1.0), (2.0, 0.8), (1.5, 0.6), (1.0, 0.5), (0.8, 0.3),
         (0.5, 0.2), (0.3, 0.1), (0.2, 0.05), (0.1, 0.0), (0.0, 0.0)]

VARIANTS = [
    ("① 两阶段（基线）", dict(use_global_refine=False)),
    ("② + 全局精化", dict(use_global_refine=True, use_balance_weight=False)),
    ("③ + 分面分层均衡", dict(use_global_refine=True, use_balance_weight=True)),
]

LIDARS = [
    ("16线 0.2°", dict(horizontal_resolution_deg=0.2)),
    ("16线 0.8°（稀疏）", dict(horizontal_resolution_deg=0.8)),
]

print("平均绝对误差 / 最大绝对误差（单位 ‰，10 个工况）")
print("=" * 78)
for lidar_name, lk in LIDARS:
    print(f"\n【{lidar_name}】")
    print(f"{'方案':<22}{'α 平均':>9}{'α 最大':>9}{'δ 平均':>9}{'δ 最大':>9}")
    print("-" * 58)
    for vname, vk in VARIANTS:
        ea, ed = [], []
        for tx, ty in CASES:
            cfg = TowerConfig(tilt_deg_x=tx, tilt_deg_y=ty)
            ta, td, _, _, _ = true_indicators(cfg)
            scfg = SolverConfig(**vk)
            scfg.tower_height = cfg.total_height
            pcd = simulate_scan(build_tower(cfg), LidarConfig(**lk))
            r, _ = solve(pcd, scfg)
            ea.append(abs(r.alpha_standard_permille - ta))
            ed.append(abs(r.foundation_tilt_permille - td))
        print(f"{vname:<22}{np.mean(ea):9.3f}{np.max(ea):9.3f}{np.mean(ed):9.3f}{np.max(ed):9.3f}")
