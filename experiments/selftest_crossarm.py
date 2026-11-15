# -*- coding: utf-8 -*-
"""横担健康分析自检：完整 / 单侧缺失 / 不同倾斜姿态下的臂长与对称性"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
np.random.seed(42)
from core.tower_model import TowerConfig, build_tower, _resolve_crossarm_levels, _body_width_at
from core.lidar_scanner import LidarConfig, simulate_scan
from core.tilt_solver import SolverConfig, solve
from core.crossarm_analysis import analyze_crossarms

CASES = [
    ("完整", dict()),
    ("右侧横担缺失", dict(crossarm_missing_side=1)),
    ("左侧横担缺失", dict(crossarm_missing_side=-1)),
    ("完整 + 倾斜 1°", dict(tilt_deg_x=1.0, tilt_deg_y=0.5)),
    ("完整 + 倾斜 2° + 弯曲", dict(tilt_deg_x=2.0, tilt_deg_y=0.8, bow_ratio=0.2)),
]

print(f"{'工况':<22}{'层高':>6}{'左臂长':>8}{'右臂长':>8}{'差(mm)':>8}{'左状态':>8}{'右状态':>8}")
print("-" * 74)
for name, kw in CASES:
    cfg = TowerConfig(**kw)
    scfg = SolverConfig()
    scfg.tower_height = cfg.total_height
    _, inter = solve(simulate_scan(build_tower(cfg), LidarConfig(horizontal_resolution_deg=0.2)), scfg)
    reps = analyze_crossarms(np.asarray(inter["crossarm_cloud"].points),
                             inter["axis_params"],
                             _resolve_crossarm_levels(cfg),
                             lambda h: _body_width_at(h, cfg) / 2.0,
                             cfg.crossarm_length)
    for r in reps:
        print(f"{name:<22}{r.height:6.1f}{r.left.arm_len:8.2f}{r.right.arm_len:8.2f}"
              f"{r.asymmetry_mm:8.0f}{r.left.status:>9}{r.right.status:>9}")
    print()
print("设计臂长 = 4.00 m")
