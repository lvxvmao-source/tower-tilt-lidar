# -*- coding: utf-8 -*-
"""
批量实验：生成10座不同倾斜角的虚拟塔，扫描→解算→对比真值
输出结果保存到 results/batch_experiment.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import open3d as o3d

from simulator.tower_builder import TowerParams, build_tower
from simulator.lidar_simulator import simulate_scan
from algorithms.tilt_estimator import estimate_tilt
from core.tower_model import true_tilt

# 10组真值倾斜角（X方向, Y方向），由大到小
# 总倾斜率从约52‰ 到 0
TEST_CASES = [
    (3.0, 1.0, "case01_严重倾斜"),
    (2.0, 0.8, "case02_大倾斜"),
    (1.5, 0.6, "case03_较大倾斜"),
    (1.0, 0.5, "case04_中倾斜"),
    (0.8, 0.3, "case05_中轻倾斜"),
    (0.5, 0.2, "case06_小倾斜"),
    (0.3, 0.1, "case07_轻微倾斜"),
    (0.2, 0.05, "case08_微小倾斜"),
    (0.1, 0.0, "case09_近直立"),
    (0.0, 0.0, "case10_完全直立"),
]

def main():
    np.random.seed(42)
    results = []

    print(f"{'编号':<22} {'真值总角°':<10} {'测量总角°':<10} {'总误差°':<10} {'真值‰':<8} {'测量‰':<8} {'‰误差':<8} {'点数':<6}")
    print("-" * 90)

    for tx, ty, name in TEST_CASES:
        # 造塔
        params = TowerParams(tilt_deg_x=tx, tilt_deg_y=ty)
        tower = build_tower(params)
        # 扫描
        pcd = simulate_scan(tower)
        n_pts = len(pcd.points)
        # 解算
        result = estimate_tilt(pcd)

        # 真值取与算法同口径的解析最小二乘方向（4~27m），
        # 这样即使以后换成带弯曲的倾斜模型，这里的对比依然成立
        _, _, true_total, true_rate = true_tilt(params, 4.0, 27.0)

        err_total = result.total_tilt_deg - true_total
        err_rate = result.tilt_rate_permille - true_rate

        print(f"{name:<22} {true_total:<10.4f} {result.total_tilt_deg:<10.4f} {err_total:<+10.4f} {true_rate:<8.2f} {result.tilt_rate_permille:<8.2f} {err_rate:<+8.2f} {n_pts:<6}")

        results.append({
            "name": name,
            "true_x_deg": tx,
            "true_y_deg": ty,
            "true_total_deg": round(true_total, 4),
            "meas_total_deg": round(result.total_tilt_deg, 4),
            "err_total_deg": round(err_total, 4),
            "true_rate_permille": round(true_rate, 2),
            "meas_rate_permille": round(result.tilt_rate_permille, 2),
            "err_rate_permille": round(err_rate, 2),
            "n_points": n_pts,
            "n_axis": result.num_axis_points,
            "n_inliers": result.num_inliers,
            "confidence": round(result.confidence, 3),
        })

    # 保存CSV
    import csv
    out_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "batch_experiment.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n结果已保存到: {out_csv}")

    # 统计
    abs_errs = [abs(r["err_total_deg"]) for r in results]
    print(f"\n总角度绝对误差：平均 {np.mean(abs_errs):.4f}°，最大 {np.max(abs_errs):.4f}°")
    abs_rate_errs = [abs(r["err_rate_permille"]) for r in results]
    print(f"倾斜率绝对误差：平均 {np.mean(abs_rate_errs):.2f}‰，最大 {np.max(abs_rate_errs):.2f}‰")

if __name__ == "__main__":
    main()
