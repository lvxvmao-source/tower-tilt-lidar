# -*- coding: utf-8 -*-
"""最小PyVista显示测试"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from pyvistaqt import QtInteractor
import pyvista as pv

app = QApplication(sys.argv)

window = QMainWindow()
central = QWidget()
layout = QVBoxLayout(central)

plotter = QtInteractor(central)
layout.addWidget(plotter)
window.setCentralWidget(central)

# 加一个测试点云
pts = np.random.rand(1000, 3) * 10
pdata = pv.PolyData(pts)
plotter.add_points(pdata, color='red', point_size=5)

# 加一个测试立方体
cube = pv.Cube()
plotter.add_mesh(cube, color='blue', opacity=0.5)

plotter.set_background('#2b2b2b')
plotter.show_axes()
plotter.reset_camera()

window.resize(1000, 800)
window.show()

sys.exit(app.exec())
