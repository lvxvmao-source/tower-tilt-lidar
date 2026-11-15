# -*- coding: utf-8 -*-
"""
完整实验矩阵：4种线数 × 5种切片步长 × 10个倾斜case
输出：精度热力图 + CSV数据
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import csv
from experiments.configurable_test import LIDAR_PRESETS, simulate_scan_with_config
from simulator.tower_builder import TowerParams, build_tower
import algorithms.tilt_estimator as te
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 10个倾斜case（和batch_test一致）
CASES = [
    ("case01", 3.0, 1.0),
    ("case02", 2.0, 0.8),
    ("case03", 1.5, 0.6),
    ("case04", 1.0, 0.5),
    ("case05", 0.8, 0.3),
    ("case06", 0.5, 0.2),
    ("case07", 0.3, 0.1),
    ("case08", 0.2, 0.05),
    ("case09", 0.1, 0.0),
    ("case10", 0.0, 0.0),
]

STEPS = [0.5, 0.75, 1.0, 1.5, 2.0]
LIDAR_NAMES = list(LIDAR_PRESETS.keys())


def run_one(lidar_name, step_m, tx, ty):
    np.random.seed(42)
    lidar = LIDAR_PRESETS[lidar_name]
    tower = build_tower(TowerParams(tilt_deg_x=tx, tilt_deg_y=ty))
    pcd = simulate_scan_with_config(tower, lidar)
    p1 = te.preprocess(pcd)
    p2 = te.crop_height(p1)
    p3 = te.remove_crossarms_pca(p2)
    axis, conf = te.extract_axis_weighted(p3, window_h=step_m*2, step=step_m)
    if len(axis) < 5:
        return np.nan
    scatter = np.std(np.sqrt((axis[:,0]-axis[:,0].mean())**2 + (axis[:,1]-axis[:,1].mean())**2))
    thr = max(0.15, scatter*1.5)
    v, p0, nin, mc = te.fit_line_robust(axis, conf, distance_threshold=thr)
    meas = np.tan(np.arccos(np.clip(abs(v[2]),-1,1)))*1000
    true_total = np.arccos(np.cos(np.deg2rad(tx))*np.cos(np.deg2rad(ty)))
    true = np.tan(true_total)*1000
    return abs(meas - true)


def load_matrix(csv_path):
    """从已有 CSV 读回误差矩阵，用于只重绘不重算"""
    rows = []
    with open(csv_path, encoding="utf-8-sig") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            rows.append([float(x) for x in row[1:]])
    return np.array(rows)


def main(plot_only=False):
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "step_lidar_matrix.csv")
    matrix = np.zeros((len(LIDAR_NAMES), len(STEPS)))

    if plot_only and os.path.exists(csv_path):
        matrix = load_matrix(csv_path)
        print(f"已从 {csv_path} 读取现成结果，仅重绘图表")
    else:
        print(f"开始实验矩阵：{len(LIDAR_NAMES)}种雷达 × {len(STEPS)}种步长 × {len(CASES)}个case")
        print("-" * 70)

        for i, lidar_name in enumerate(LIDAR_NAMES):
            for j, step in enumerate(STEPS):
                errors = []
                for case_name, tx, ty in CASES:
                    err = run_one(lidar_name, step, tx, ty)
                    if not np.isnan(err):
                        errors.append(err)
                avg_err = np.mean(errors) if errors else np.nan
                matrix[i, j] = avg_err
                print(f"  {lidar_name:<18} 步长{step:<5}m  平均误差={avg_err:.3f}‰  ({len(errors)}个有效case)")

        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["雷达型号"] + [f"步长{s}m" for s in STEPS])
            for i, name in enumerate(LIDAR_NAMES):
                w.writerow([name] + [f"{matrix[i,j]:.3f}" for j in range(len(STEPS))])
        print(f"\nCSV已保存: {csv_path}")

    # 色标上限按实际误差自适应：误差整体降到 1‰ 以内后，
    # 固定 vmax=3 会让整张图糊成一片绿，看不出差异
    vmax = max(0.5, float(np.nanmax(matrix)) * 1.15)
    vmax = np.ceil(vmax * 4) / 4

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=vmax)

    ax.set_xticks(range(len(STEPS)))
    ax.set_xticklabels([f"{s} m" for s in STEPS], fontsize=12)
    ax.set_yticks(range(len(LIDAR_NAMES)))
    ax.set_yticklabels([n.replace("线 ", "线\n") for n in LIDAR_NAMES], fontsize=11)
    ax.set_xlabel("切片步长", fontsize=13)
    ax.set_ylabel("激光雷达型号", fontsize=13)
    ax.set_title("不同线数雷达 × 不同切片步长下的平均绝对误差 (‰)\n(10个倾斜case平均，值越小越好)", fontsize=14)

    # 格子里写数字
    for i in range(len(LIDAR_NAMES)):
        for j in range(len(STEPS)):
            text = ax.text(j, i, f"{matrix[i,j]:.2f}",
                           ha="center", va="center", color="black", fontsize=12, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("平均绝对误差 (‰)", fontsize=12)

    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "results", "step_lidar_heatmap.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"热力图已保存: {fig_path}")

    # 画折线图
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    markers = ['o', 's', '^', 'D']
    for i, name in enumerate(LIDAR_NAMES):
        ax2.plot(STEPS, matrix[i], marker=markers[i], label=name, linewidth=2, markersize=8)
    ax2.set_xlabel("切片步长 (m)", fontsize=13)
    ax2.set_ylabel("平均绝对误差 (‰)", fontsize=13)
    ax2.set_title("不同线数雷达的切片步长-精度曲线", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(STEPS)
    plt.tight_layout()
    line_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "results", "step_lidar_line.png")
    plt.savefig(line_path, dpi=150, bbox_inches='tight')
    print(f"折线图已保存: {line_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="雷达线数 × 切片步长 实验矩阵")
    parser.add_argument("--plot-only", action="store_true",
                        help="不重算，直接读 results/step_lidar_matrix.csv 重绘图表")
    args = parser.parse_args()
    main(plot_only=args.plot_only)
