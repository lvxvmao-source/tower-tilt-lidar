# -*- coding: utf-8 -*-
"""GUI 离屏回归：线程安全 / 图层勾选 / 标准指标显示"""
import sys, os, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
app = QApplication(sys.argv)
from ui.main_window import MainWindow

w = MainWindow()
w.lidar_panel.horiz_res.setValue(0.4)
w.tower_panel.tilt_x.setValue(1.0)
w.tower_panel.tilt_y.setValue(0.5)
w.tower_panel.voltage.setCurrentText("±800kV")

state = {"step": 0}
events = []


def rows():
    return w.layer_panel._box_layout.count() - 1


def step():
    s = state["step"]; state["step"] += 1
    try:
        if s == 0:
            print("[1] 连续点击两次运行")
            w.run_simulation()
            w.run_simulation()
            print("    第二次被拦截，按钮:", w.act_run.isEnabled(), w.act_run.text())

        elif s == 1:
            print(f"[2] 第一次结束，图层行数={rows()}")
            w.run_simulation()

        elif s == 2:
            print(f"[3] 第二次结束，图层行数={rows()}（应为5），actor={len(w.viewer._actors)}")
            print("[4] 勾选框双向切换")
            w.layer_panel.layer_visible_changed.connect(lambda n, v: events.append((n, v)))
            cb = w.layer_panel._checks["原始点云"]
            cb.setChecked(False); a = events[-1] if events else None
            cb.setChecked(True); b = events[-1] if events else None
            print("    取消->", a, "| 重新勾选->", b)

            print("[5] 标准指标显示")
            rp = w.result_panel
            print("    倾斜率 α      :", rp.val_rate.text())
            print("    塔顶水平偏移  :", rp.val_tip.text())
            print("    α 运行限值    :", rp.val_alpha_std.text())
            print("    状态等级      :", rp.val_grade.text())
            print("    基础倾斜率 δ  :", rp.val_delta.text())
            print("    δ 设计允许值  :", rp.val_delta_limit.text())
            print("    重大隐患      :", rp.val_hazard.text())
            print("    结论          :", rp.alert_label.text())
            print("    真值 α        :", rp.val_true_alpha.text())
            print("    α 误差        :", rp.val_err_alpha.text())
            print("    真值 δ        :", rp.val_true_delta.text())
            print("    δ 误差        :", rp.val_err_delta.text())
            print("    真值状态      :", rp.val_true_grade.text())
            print("    轴线弯曲量    :", rp.val_curve.text())
            QTimer.singleShot(300, app.quit)
    except Exception:
        traceback.print_exc()
        app.quit()


t = QTimer(); t.timeout.connect(step); t.start(1600)
QTimer.singleShot(40000, app.quit)
app.exec()
print("\n=== 正常退出 ===")
