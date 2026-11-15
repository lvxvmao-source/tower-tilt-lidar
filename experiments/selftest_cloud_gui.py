# -*- coding: utf-8 -*-
"""GUI 离屏回归：实测点云模式端到端"""
import sys, os, tempfile, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import numpy as np
np.random.seed(42)
import open3d as o3d
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from core.tower_model import TowerConfig, build_tower, true_indicators
from core.lidar_scanner import LidarConfig, simulate_scan

# 造一份"实测点云"：已知倾斜 1°/0.5°，并平移到工地坐标
cfg = TowerConfig(tilt_deg_x=1.0, tilt_deg_y=0.5)
ta, td, _, _, _ = true_indicators(cfg)
pcd = simulate_scan(build_tower(cfg), LidarConfig(horizontal_resolution_deg=0.2))
pts = np.asarray(pcd.points) + np.array([1200.0, 3400.0, 156.7])
out = o3d.geometry.PointCloud()
out.points = o3d.utility.Vector3dVector(pts)
PLY = os.path.join(tempfile.gettempdir(), "field_scan_demo.ply")
o3d.io.write_point_cloud(PLY, out)
print(f"生成测试点云: {PLY}  ({len(pts)} 点, 真值 α={ta:.2f}‰)")

app = QApplication(sys.argv)
from ui.main_window import MainWindow

w = MainWindow()
state = {"step": 0}


def step():
    s = state["step"]; state["step"] += 1
    try:
        tp = w.tower_panel
        if s == 0:
            print("[1] 默认建模模式, actors =", list(w.viewer._actors.keys()))
            tp.pcd_path = PLY
            tp.rb_cloud.setChecked(True)
            print("    切到点云模式并载入文件")

        elif s == 1:
            print("[2] 点云预览 actors =", list(w.viewer._actors.keys()))
            print("    日志尾:", w.result_panel.log.toPlainText().strip().splitlines()[-1])
            w.run_simulation()

        elif s == 2:
            rp = w.result_panel
            print("[3] 解算完成, actors =", list(w.viewer._actors.keys()))
            print("    α   :", rp.val_rate.text(), " (真值 %.2f‰, 误差 %s)" % (ta, rp.val_err_alpha.text()))
            print("    δ   :", rp.val_delta.text())
            print("    塔顶偏移:", rp.val_tip.text())
            print("    状态:", rp.val_grade.text(), "| 结论:", rp.alert_label.text())
            print("    真值对比:", rp.val_true_alpha.text(), "（应为 -- ，导入点云无真值）")
            print("    精化:", rp.val_refine.text().replace(chr(10), " "))
            print("    横担:", rp.val_crossarm.text()[:30])
            QTimer.singleShot(300, app.quit)
    except Exception:
        traceback.print_exc()
        app.quit()


t = QTimer(); t.timeout.connect(step); t.start(2000)
QTimer.singleShot(30000, app.quit)
app.exec()
print("\n=== 正常退出 ===")
