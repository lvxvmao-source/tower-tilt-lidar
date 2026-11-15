# -*- coding: utf-8 -*-
"""
3D视图组件：PyVista + pyvistaqt嵌入
"""
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                                  QSlider, QLabel)
from PySide6.QtCore import Qt
from typing import Dict


class PointCloudViewer(QWidget):
    """嵌入式3D查看器"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._actors: Dict[str, object] = {}
        self._kinds: Dict[str, str] = {}
        self._bounds = None            # 只统计业务图层的包围盒，不含地面网格
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        # 顶部工具栏
        btn_bar = QHBoxLayout()
        for label, view in [("等轴测", "isometric"), ("前视", "front"),
                            ("侧视", "side"), ("顶视", "top")]:
            b = QPushButton(label)
            b.clicked.connect(lambda checked=False, v=view: self._set_view(v))
            btn_bar.addWidget(b)

        btn_bar.addWidget(QLabel("点大小:"))
        self.point_size = QSlider(Qt.Horizontal)
        self.point_size.setRange(1, 12)
        self.point_size.setValue(4)
        self.point_size.setFixedWidth(80)
        self.point_size.valueChanged.connect(self._update_point_size)
        btn_bar.addWidget(self.point_size)
        layout.addLayout(btn_bar)

        # PyVista 3D视图
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter)

        # 基础设置
        self.plotter.set_background("#2b2b2b")
        self.plotter.show_axes()
        self._add_ground_grid()

    def _add_ground_grid(self):
        grid = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1),
                        i_size=40, j_size=40, i_resolution=40, j_resolution=40)
        self.plotter.add_mesh(grid, color="#555555", style="wireframe", opacity=0.4)

    # ---------------------------------------------------------------- 视角

    def _set_view(self, view_name):
        """view_xy=俯视(-Z)，view_xz=正视(-Y)，view_yz=侧视(-X)"""
        if view_name == "isometric":
            self.plotter.view_isometric()
        elif view_name == "front":
            self.plotter.view_xz()
        elif view_name == "side":
            self.plotter.view_yz()
        elif view_name == "top":
            self.plotter.view_xy()
        self.fit_view()

    # ---------------------------------------------------------------- 图层

    def _register(self, name, actor, kind):
        self._actors[name] = actor
        self._kinds[name] = kind
        self._update_bounds()

    def _drop_actor(self, name):
        actor = self._actors.pop(name, None)
        self._kinds.pop(name, None)
        if actor is not None:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self._update_bounds()

    def _update_bounds(self):
        """合并所有业务图层的包围盒（不含地面网格，否则相机被拉到 40m 外）"""
        bounds = None
        for name, actor in self._actors.items():
            try:
                b = actor.GetBounds()
            except Exception:
                continue
            if not b or len(b) != 6:
                continue
            bounds = list(b) if bounds is None else [
                min(bounds[0], b[0]), max(bounds[1], b[1]),
                min(bounds[2], b[2]), max(bounds[3], b[3]),
                min(bounds[4], b[4]), max(bounds[5], b[5]),
            ]
        self._bounds = bounds

    def _update_point_size(self):
        size = self.point_size.value()
        for name, actor in self._actors.items():
            if self._kinds.get(name) != "points":
                continue
            try:
                actor.prop.point_size = size
            except Exception:
                pass
        self.plotter.render()

    def add_point_cloud(self, name: str, points: np.ndarray,
                        color=(0.8, 0.8, 0.8), visible=True):
        """添加点云图层"""
        if points is None or len(points) == 0:
            return
        self._drop_actor(name)

        pdata = pv.PolyData(np.asarray(points, dtype=float))
        actor = self.plotter.add_points(
            pdata, color=color, point_size=self.point_size.value(),
            opacity=1.0, render_points_as_spheres=True,
        )
        actor.visibility = visible
        self._register(name, actor, "points")
        self.plotter.render()

    def add_lines(self, name: str, points: np.ndarray, color=(0.0, 0.8, 0.2),
                  width: float = 4.0, visible=True):
        """添加折线图层（拟合中心线用这个，之前当成点云画只看到两个点）"""
        if points is None or len(points) < 2:
            return
        self._drop_actor(name)

        pts = np.asarray(points, dtype=float)
        actor = self.plotter.add_lines(pts, color=color, width=width, connected=True)
        actor.visibility = visible
        self._register(name, actor, "lines")
        self.plotter.render()

    def add_mesh(self, name: str, verts: np.ndarray, faces: np.ndarray,
                 color=(0.6, 0.6, 0.6), visible=True, opacity=0.7):
        """添加网格图层"""
        if verts is None or len(verts) == 0 or len(faces) == 0:
            return
        self._drop_actor(name)

        verts = np.asarray(verts, dtype=float)
        faces = np.asarray(faces, dtype=np.int64)
        faces_with_n = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces])
        pdata = pv.PolyData(verts, faces=faces_with_n)
        actor = self.plotter.add_mesh(pdata, color=color, opacity=opacity)
        actor.visibility = visible
        self._register(name, actor, "mesh")
        self.plotter.render()

    def set_layer_visible(self, name: str, visible: bool):
        """设置图层显示/隐藏"""
        if name in self._actors:
            self._actors[name].visibility = bool(visible)
            self.plotter.render()

    def set_layer_color(self, name: str, color):
        """设置图层颜色，color为(r,g,b) 0-1"""
        if name in self._actors:
            try:
                self._actors[name].prop.color = color
            except Exception:
                pass
            self.plotter.render()

    def clear_all(self):
        """清空所有业务图层"""
        for name in list(self._actors.keys()):
            self._drop_actor(name)
        self._actors.clear()
        self._kinds.clear()
        self._bounds = None
        self.plotter.render()

    def fit_view(self):
        """相机对准业务图层；没有数据时用默认视角"""
        if self._bounds is not None:
            try:
                self.plotter.reset_camera(bounds=self._bounds)
                self.plotter.render()
                return
            except Exception:
                pass
        self.plotter.reset_camera()
