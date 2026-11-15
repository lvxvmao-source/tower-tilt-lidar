# -*- coding: utf-8 -*-
"""
图层面板：管理各图层的显示/隐藏/颜色
"""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QCheckBox,
                                  QPushButton, QColorDialog, QScrollArea, QHBoxLayout)
from PySide6.QtCore import Signal
from PySide6.QtGui import QColor


class LayerPanel(QWidget):
    """图层控制面板"""
    layer_visible_changed = Signal(str, bool)
    layer_color_changed = Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        box = QGroupBox("图层管理")
        self._box_layout = QVBoxLayout(box)
        self._box_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(box)
        layout.addWidget(scroll)

        self._checks = {}
        self._rows = {}

    # ---------------------------------------------------------------- 增删

    def add_layer(self, name: str, visible: bool = True, color=(0.7, 0.7, 0.7)):
        """添加（或更新）一个图层。

        同名图层不会重复添加 —— 之前每跑一次仿真就往面板里再塞一行，
        跑两次就从 5 行变 10 行。
        """
        if name in self._checks:
            cb = self._checks[name]
            cb.blockSignals(True)
            cb.setChecked(visible)
            cb.blockSignals(False)
            return

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        cb = QCheckBox(name)
        cb.setChecked(visible)
        # 用 toggled(bool)：stateChanged 在 PySide6 6.x 传的是 int，
        # 而 int 与 Qt.CheckState.Checked 直接比较恒为 False，会导致勾选失效。
        cb.toggled.connect(lambda checked, n=name: self.layer_visible_changed.emit(n, checked))
        row_layout.addWidget(cb)
        self._checks[name] = cb

        color_btn = QPushButton("颜色")
        color_btn.setFixedWidth(50)
        color_btn.setStyleSheet(
            f"background-color: rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)});"
        )
        color_btn.clicked.connect(lambda checked=False, n=name: self._pick_color(n))
        row_layout.addWidget(color_btn)
        self._rows[name] = (row, color_btn)

        # 插到 stretch 之前，保持顺序稳定
        self._box_layout.insertWidget(self._box_layout.count() - 1, row)

    def clear_layers(self):
        """清空所有图层条目（配合 3D 视图 clear_all 使用）"""
        for name, (row, _) in list(self._rows.items()):
            self._box_layout.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        self._checks.clear()
        self._rows.clear()

    # ---------------------------------------------------------------- 交互

    def _pick_color(self, name):
        btn = self._rows.get(name, (None, None))[1]
        initial = QColor(180, 180, 180)
        c = QColorDialog.getColor(initial, self, f"选择「{name}」颜色")
        if c.isValid():
            rgb = (c.redF(), c.greenF(), c.blueF())
            if btn is not None:
                btn.setStyleSheet(f"background-color: rgb({c.red()},{c.green()},{c.blue()});")
            self.layer_color_changed.emit(name, rgb)

    def set_layer_visible(self, name: str, visible: bool):
        if name in self._checks:
            cb = self._checks[name]
            cb.blockSignals(True)
            cb.setChecked(visible)
            cb.blockSignals(False)
