# -*- coding: utf-8 -*-
"""
两期点云对比面板

思路：两期各自重建塔轴，直接比较两条轴线（以各自塔基为原点）。
免配准、与点云密度无关，输出的是"哪个高度位移了多少"。
"""
import os

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QPushButton,
                                QDoubleSpinBox, QGroupBox, QFileDialog, QLabel)
from PySide6.QtCore import Signal


class DeformPanel(QWidget):
    compare_requested = Signal()
    clear_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        inp = QGroupBox("两期对比输入")
        form = QFormLayout(inp)

        self.epoch1_label = QLabel("未解算（先跑一次解算作为第一期）")
        self.epoch1_label.setWordWrap(True)
        self.epoch1_label.setStyleSheet("color: #666;")
        form.addRow("第一期：", self.epoch1_label)

        self.btn_epoch2 = QPushButton("导入第二期点云")
        self.btn_epoch2.clicked.connect(self._pick_epoch2)
        form.addRow(self.btn_epoch2)

        self.epoch2_label = QLabel("未导入")
        self.epoch2_label.setWordWrap(True)
        self.epoch2_label.setStyleSheet("color: #666;")
        form.addRow("第二期：", self.epoch2_label)

        self.days = QDoubleSpinBox()
        self.days.setRange(1.0, 3650.0)
        self.days.setDecimals(0)
        self.days.setSingleStep(30.0)
        self.days.setValue(90.0)
        self.days.setSuffix(" 天")
        form.addRow("时间间隔：", self.days)

        self.btn_run = QPushButton("计算两期变化")
        self.btn_run.clicked.connect(self.compare_requested.emit)
        form.addRow(self.btn_run)

        self.btn_clear = QPushButton("清除第二期")
        self.btn_clear.clicked.connect(self._clear)
        form.addRow(self.btn_clear)
        layout.addWidget(inp)

        res = QGroupBox("变化指标")
        rf = QFormLayout(res)
        self.val_alpha = QLabel("--")
        self.val_alpha.setStyleSheet("font-weight: bold; font-size: 14px; color: #1565C0;")
        rf.addRow("倾斜率增量 Δα：", self.val_alpha)
        self.val_delta = QLabel("--")
        rf.addRow("基础倾斜增量 Δδ：", self.val_delta)
        self.val_tip = QLabel("--")
        rf.addRow("塔顶增量位移：", self.val_tip)
        self.val_max = QLabel("--")
        rf.addRow("最大增量位移：", self.val_max)
        self.val_rate = QLabel("--")
        self.val_rate.setStyleSheet("font-weight: bold; font-size: 14px; color: #C62828;")
        rf.addRow("年化变形速率：", self.val_rate)
        self.val_base = QLabel("--")
        rf.addRow("基础水平位移：", self.val_base)
        layout.addWidget(res)

        self.note = QLabel(
            "说明：两期都以各自塔基中心为原点比较，得到的是「塔身相对基础的变化」；\n"
            "基础自身的水平位移单独列出。全程不需要点云配准。")
        self.note.setWordWrap(True)
        self.note.setStyleSheet("font-size: 11px; color: #888;")
        layout.addWidget(self.note)

        layout.addStretch()
        self.epoch2_path = None

    def _pick_epoch2(self):
        from core.point_io import EXT_FILTER
        path, _ = QFileDialog.getOpenFileName(self, "选择第二期点云", "", EXT_FILTER)
        if path:
            self.epoch2_path = path
            self.epoch2_label.setText(os.path.basename(path))
            self.epoch2_label.setStyleSheet("color: #1565C0;")

    def _clear(self):
        self.epoch2_path = None
        self.epoch2_label.setText("未导入")
        self.epoch2_label.setStyleSheet("color: #666;")
        for lab in (self.val_alpha, self.val_delta, self.val_tip,
                    self.val_max, self.val_rate, self.val_base):
            lab.setText("--")

    def set_epoch1(self, text: str):
        self.epoch1_label.setText(text)
        self.epoch1_label.setStyleSheet("color: #1565C0;")

    def show_result(self, cp):
        self.val_alpha.setText(f"{cp.delta_alpha:+.2f}‰"
                               f"   （{cp.alpha1:.2f} → {cp.alpha2:.2f}‰）")
        self.val_delta.setText(f"{cp.delta_delta:+.2f}‰"
                               f"   （{cp.delta1:.2f} → {cp.delta2:.2f}‰）")
        self.val_tip.setText(f"{cp.tip_shift_mm:+.1f} mm")
        self.val_max.setText(f"{cp.max_shift_mm:.1f} mm  @ 高度 {cp.z_max_shift:.1f} m")
        if cp.rate_mm_per_year:
            self.val_rate.setText(f"{cp.rate_mm_per_year:.1f} mm/年")
        else:
            self.val_rate.setText("--")
        self.val_base.setText(f"{cp.base_shift_mm:.1f} mm")
