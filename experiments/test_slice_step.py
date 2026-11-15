# -*- coding: utf-8 -*-
"""测试切片间距对精度的影响"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from simulator.tower_builder import TowerParams, build_tower
from simulator.lidar_simulator import simulate_scan
import algorithms.tilt_estimator as te

for tx, ty, true_rate in [(0.5, 0.2, 9.4), (3.0, 1.0, 55.25)]:
    np.random.seed(42)
    params = TowerParams(tilt_deg_x=tx, tilt_deg_y=ty)
    tower = build_tower(params)
    pcd = simulate_scan(tower)
    p1 = te.preprocess(pcd)
    p2 = te.crop_height(p1)
    p3 = te.remove_crossarms_pca(p2)
    print(f"=== case: tilt=({tx},{ty}), true_rate={true_rate} permille ===")
    print(f"{'step':<6} {'win_h':<6} {'n_axis':<8} {'meas_rate':<10} {'err':<8}")
    for step, win in [(0.3,0.6),(0.5,1.0),(0.75,1.5),(1.0,2.0),(1.5,3.0),(2.0,4.0)]:
        axis, conf = te.extract_axis_weighted(p3, window_h=win, step=step)
        if len(axis) < 5:
            print(f"{step:<6} {win:<6} too few points")
            continue
        scatter = np.std(np.sqrt((axis[:,0]-axis[:,0].mean())**2 + (axis[:,1]-axis[:,1].mean())**2))
        thr = max(0.15, scatter*1.5)
        v, p0, nin, mc = te.fit_line_robust(axis, conf, distance_threshold=thr)
        rate = np.tan(np.arccos(np.clip(abs(v[2]),-1,1)))*1000
        print(f"{step:<6} {win:<6} {len(axis):<8} {rate:<10.2f} {rate-true_rate:+.2f}")
    print()
