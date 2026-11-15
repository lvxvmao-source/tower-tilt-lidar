# -*- coding: utf-8 -*-
"""标准指标测算自检：α（杆塔倾斜率）/ δ（基础倾斜率）/ 塔顶偏移 / 状态等级"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
np.random.seed(42)
from core.tower_model import TowerConfig, build_tower, true_indicators
from core.lidar_scanner import LidarConfig, simulate_scan
from core.tilt_solver import SolverConfig, solve
from core.standards import evaluate

CASES = [
    ("直立", TowerConfig(), "110kV"),
    ("(0.5,0.2) 刚体", TowerConfig(tilt_deg_x=0.5, tilt_deg_y=0.2), "110kV"),
    ("(1.0,0.5) 刚体", TowerConfig(tilt_deg_x=1.0, tilt_deg_y=0.5), "220kV"),
    ("(1.0,0.5) 弯曲20%", TowerConfig(tilt_deg_x=1.0, tilt_deg_y=0.5, bow_ratio=0.2), "500kV"),
    ("(3.0,1.0) 弯曲15%", TowerConfig(tilt_deg_x=3.0, tilt_deg_y=1.0, bow_ratio=0.15), "750kV"),
    ("沉降 NE100/SW-60", TowerConfig(use_settlement=True, settlement_ne=0.10,
                                     settlement_nw=0.02, settlement_se=0.02,
                                     settlement_sw=-0.06), "±800kV"),
]

print(f"{'工况':<20}{'真α‰':>7}{'测α‰':>7}{'Δα':>6}{'真δ‰':>7}{'测δ‰':>7}{'Δδ':>6}  {'真状态':<6}{'测状态':<6}一致 隐患")
print("-" * 100)
ea, ed = [], []
for name, cfg, volt in CASES:
    ta, td, tip, tx, ty = true_indicators(cfg)
    scfg = SolverConfig()
    scfg.tower_height = cfg.total_height
    pcd = simulate_scan(build_tower(cfg), LidarConfig(horizontal_resolution_deg=0.2))
    r, _ = solve(pcd, scfg)
    ev_m = evaluate(r.alpha_standard_permille, r.foundation_tilt_permille,
                    cfg.total_height, volt)
    ev_t = evaluate(ta, td, cfg.total_height, volt)
    ea.append(abs(r.alpha_standard_permille - ta))
    ed.append(abs(r.foundation_tilt_permille - td))
    print(f"{name:<20}{ta:7.2f}{r.alpha_standard_permille:7.2f}"
          f"{r.alpha_standard_permille-ta:+6.2f}{td:7.2f}{r.foundation_tilt_permille:7.2f}"
          f"{r.foundation_tilt_permille-td:+6.2f}  {ev_t.grade:<6}{ev_m.grade:<6}"
          f"{'✓' if ev_t.grade==ev_m.grade else '✗':<3} "
          f"{'是' if ev_m.major_hazard else '否'}")
print("-" * 100)
print(f"α 平均绝对误差 {np.mean(ea):.2f}‰  最大 {np.max(ea):.2f}‰")
print(f"δ 平均绝对误差 {np.mean(ed):.2f}‰  最大 {np.max(ed):.2f}‰（δ 为轴线外推反演量）")

print("\n【限值查询示例】塔高 30m / 60m，电压等级 ±800kV")
from core.standards import (operation_limit_permille, foundation_tilt_limit,
                            major_hazard_limit_permille, state_grade)
for h in (30.0, 60.0):
    print(f"  H={h:.0f}m: DL/T741 限值 {operation_limit_permille(h):.0f}‰ | "
          f"DL/T5219 δ允许 {foundation_tilt_limit(h)*1000:.1f}‰ | "
          f"41号令红线 {major_hazard_limit_permille(h, 800):.0f}‰")
