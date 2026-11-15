# -*- coding: utf-8 -*-
"""
杆塔倾斜测量系统 - 程序入口
"""
import sys
import os

# 把项目根目录加入path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QFontDatabase
from ui.main_window import MainWindow


def _pick_cjk_font(app):
    """显式指定中文字体：部分系统默认字体缺 CJK 字形，界面会显示成方框"""
    try:
        families = set(QFontDatabase.families())
    except Exception:
        return
    for name in ("Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC",
                 "Noto Sans CJK SC", "SimHei"):
        if name in families:
            app.setFont(QFont(name, 9))
            return


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("杆塔倾斜测量系统")
    _pick_cjk_font(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
