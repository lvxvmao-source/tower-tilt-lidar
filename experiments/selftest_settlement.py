# -*- coding: utf-8 -*-
"""GUI 离屏回归：进阶倾斜模型（差异沉降 + 塔身弯曲）"""
import sys, os, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
app = QApplication(sys.argv)
from ui.main_window import MainWindow

w = MainWindow()
w.lidar_panel.horiz_res.setValue(0.4)
state = {"step": 0}


def step():
    s = state["step"]; state["step"] += 1
    try:
        if s == 0:
            print("[1] 勾选前：沉降输入框可用状态 =",
                  [sp.isEnabled() for sp in (w.tower_panel.sp_nw, w.tower_panel.sp_ne,
                                             w.tower_panel.sp_se, w.tower_panel.sp_sw)])
            w.tower_panel.tilt_x.setValue(1.0)
            w.tower_panel.tilt_y.setValue(0.5)
            w.tower_panel._convert_settlements()
            print("[2] 换算后（mm） NW/NE/SE/SW =",
                  [round(sp.value()) for sp in (w.tower_panel.sp_nw, w.tower_panel.sp_ne,
                                                w.tower_panel.sp_se, w.tower_panel.sp_sw)])
            print("    勾选状态 =", w.tower_panel.use_settlement.isChecked(),
                  "| 输入框可用 =", w.tower_panel.sp_ne.isEnabled())

        elif s == 1:
            cfg = w.tower_panel.get_config()
            print(f"[3] get_config: use_settlement={cfg.use_settlement} "
                  f"NE={cfg.settlement_ne:.3f} SW={cfg.settlement_sw:.3f} bow={cfg.bow_ratio}")
            w.tower_panel.bow_ratio.setValue(0.20)
            print("[4] 设置弯曲比例 0.20，开始仿真")
            w.run_simulation()

        elif s == 2:
            print("[5] 沉降+弯曲 结果：")
            rp = w.result_panel
            print("    测量 X/Y :", rp.val_x.text(), "/", rp.val_y.text())
            print("    真值 X/Y :", rp.val_true_tilt.text())
            print("    α 测量/真值/误差:", rp.val_rate.text(),
                  rp.val_true_alpha.text(), rp.val_err_alpha.text())
            print("    δ 测量/真值/误差:", rp.val_delta.text(),
                  rp.val_true_delta.text(), rp.val_err_delta.text())
            print("    状态等级 :", rp.val_grade.text(), "| 结论:", rp.alert_label.text())
            cfg = w.tower_panel.get_config()
            from core.tower_model import true_indicators
            print("    解析真值复核:", tuple(round(x, 3) for x in true_indicators(cfg)))
            QTimer.singleShot(300, app.quit)
    except Exception:
        traceback.print_exc()
        app.quit()


t = QTimer(); t.timeout.connect(step); t.start(1500)
QTimer.singleShot(40000, app.quit)
app.exec()
print("\n=== 正常退出 ===")
