# -*- coding: utf-8 -*-
"""
左侧参数面板：塔模型/雷达配置/算法参数三个Tab
"""
import json
import os
import sys

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QPushButton,
                                QComboBox, QDoubleSpinBox, QSpinBox, QGroupBox,
                                QFileDialog, QLabel, QCheckBox, QRadioButton, QGridLayout)
from PySide6.QtCore import Signal, Qt

from core.tower_model import TowerConfig, settlements_from_tilt
from core.standards import VOLTAGE_OPTIONS
from core.lidar_scanner import LidarConfig


def _resource_root():
    """源码运行 → 项目根目录；PyInstaller 打包运行 → 解包临时目录"""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return meipass
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


CONFIG_DIR = os.path.join(_resource_root(), "config")


class TowerPanel(QWidget):
    """塔模型参数面板"""
    tower_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # ---- 数据源：两种模式二选一，互不干扰 ----
        src_box = QGroupBox("数据源")
        src_layout = QVBoxLayout(src_box)
        self.rb_sim = QRadioButton("建模仿真（生成塔 → 模拟扫描）")
        self.rb_cloud = QRadioButton("实测点云（直接解算，不建模）")
        self.rb_sim.setChecked(True)
        self.rb_sim.setToolTip("用参数化/导入的塔模型生成仿真点云，可与真值比对")
        self.rb_cloud.setToolTip("直接导入实际扫描的点云做解算，不需要塔模型")
        src_layout.addWidget(self.rb_sim)
        src_layout.addWidget(self.rb_cloud)
        layout.addWidget(src_box)

        # 建模模式下的所有参数放进一个容器，切换模式时整体禁用
        self.sim_panel = QWidget()
        sim_layout = QVBoxLayout(self.sim_panel)
        sim_layout.setContentsMargins(0, 0, 0, 0)

        # 倾斜角设置
        tilt_box = QGroupBox("塔身附加倾斜（仿真用）")
        tilt_form = QFormLayout(tilt_box)
        self.tilt_x = QDoubleSpinBox()
        self.tilt_x.setRange(-5.0, 5.0)
        self.tilt_x.setDecimals(2)
        self.tilt_x.setSingleStep(0.1)
        self.tilt_x.setValue(0.5)
        self.tilt_x.setToolTip(
            "塔身相对基础的附加倾斜（施工偏差、外力等）。\n"
            "若同时启用了四角沉降，最终总倾角 = 本值 + 沉降反算的基础倾角。")
        tilt_form.addRow("θx 附加倾斜·南北(°):", self.tilt_x)

        self.tilt_y = QDoubleSpinBox()
        self.tilt_y.setRange(-5.0, 5.0)
        self.tilt_y.setDecimals(2)
        self.tilt_y.setSingleStep(0.1)
        self.tilt_y.setValue(0.2)
        self.tilt_y.setToolTip(
            "塔身相对基础的附加倾斜（施工偏差、外力等）。\n"
            "若同时启用了四角沉降，最终总倾角 = 本值 + 沉降反算的基础倾角。")
        tilt_form.addRow("θy 附加倾斜·东西(°):", self.tilt_y)
        sim_layout.addWidget(tilt_box)

        # 进阶：基础差异沉降 + 塔身弯曲
        sett_box = QGroupBox("倾斜模型（进阶）")
        sett_form = QFormLayout(sett_box)

        self.use_settlement = QCheckBox("启用四腿差异沉降")
        self.use_settlement.setToolTip(
            "勾选后忽略上面的 X/Y 倾斜角，改由四个塔脚的独立沉降量决定基础倾斜；\n"
            "塔身还会叠加一条沿倾斜方位的弯曲挠曲线（由下面的弯曲比例控制）。"
        )
        self.use_settlement.toggled.connect(self._on_settlement_toggled)
        sett_form.addRow(self.use_settlement)

        def _mm_spin():
            sp = QDoubleSpinBox()
            sp.setRange(-300.0, 300.0)
            sp.setDecimals(0)
            sp.setSingleStep(5.0)
            sp.setSuffix(" mm")
            sp.setEnabled(False)
            return sp

        self.sp_nw = _mm_spin()
        self.sp_ne = _mm_spin()
        self.sp_se = _mm_spin()
        self.sp_sw = _mm_spin()
        sett_form.addRow("西北角 NW 沉降:", self.sp_nw)
        sett_form.addRow("东北角 NE 沉降:", self.sp_ne)
        sett_form.addRow("东南角 SE 沉降:", self.sp_se)
        sett_form.addRow("西南角 SW 沉降:", self.sp_sw)
        for _sp in (self.sp_nw, self.sp_ne, self.sp_se, self.sp_sw):
            _sp.valueChanged.connect(self._update_sett_warn)

        self.bow_ratio = QDoubleSpinBox()
        self.bow_ratio.setRange(0.0, 0.50)
        self.bow_ratio.setDecimals(2)
        self.bow_ratio.setSingleStep(0.05)
        self.bow_ratio.setValue(0.0)
        self.bow_ratio.setToolTip(
            "0 = 纯刚体旋转，塔轴为直线；\n"
            "0.20 = 塔顶在刚体倾斜的基础上再额外偏移 20%，塔轴变成曲线。"
        )
        sett_form.addRow("塔身弯曲附加比例:", self.bow_ratio)

        # 四角全零时的醒目提示（勾选沉降却没填值，最容易误判）
        self.sett_warn = QLabel("")
        self.sett_warn.setWordWrap(True)
        self.sett_warn.setStyleSheet(
            "background:#FFF3E0; color:#E65100; padding:4px; border-radius:3px;")
        self.sett_warn.hide()
        sett_form.addRow(self.sett_warn)

        self.convert_btn = QPushButton("把塔身倾斜转为基础沉降（倾斜清零）")
        self.convert_btn.clicked.connect(self._convert_settlements)
        sett_form.addRow(self.convert_btn)

        sim_layout.addWidget(sett_box)

        # 几何参数
        geom_box = QGroupBox("塔几何参数")
        geom_form = QFormLayout(geom_box)
        self.height = QDoubleSpinBox()
        self.height.setRange(10, 60)
        self.height.setValue(30)
        self.height.valueChanged.connect(self._on_height_changed)
        geom_form.addRow("总高(m):", self.height)

        self.bottom_w = QDoubleSpinBox()
        self.bottom_w.setRange(2, 10)
        self.bottom_w.setValue(5.0)
        geom_form.addRow("底宽(m):", self.bottom_w)

        self.top_w = QDoubleSpinBox()
        self.top_w.setRange(1, 5)
        self.top_w.setValue(2.0)
        geom_form.addRow("顶宽(m):", self.top_w)

        self.voltage = QComboBox()
        self.voltage.addItems(list(VOLTAGE_OPTIONS))
        self.voltage.setToolTip(
            "用于判定发改委 41 号令的重大隐患红线：\n"
            "±800kV / 1000kV 与 ±660kV / 750kV 才在该条款覆盖范围内。"
        )
        geom_form.addRow("电压等级:", self.voltage)
        sim_layout.addWidget(geom_box)

        # 导入 PLY
        self.import_btn = QPushButton("导入外部 PLY 塔模型")
        self.import_btn.clicked.connect(self._import_ply)
        sim_layout.addWidget(self.import_btn)

        self.ply_label = QLabel("当前使用：参数化建模")
        self.ply_label.setWordWrap(True)
        self.ply_label.setStyleSheet("color: #666;")
        sim_layout.addWidget(self.ply_label)

        self.clear_btn = QPushButton("恢复参数化模型")
        self.clear_btn.clicked.connect(self._clear_ply)
        self.clear_btn.setEnabled(False)
        sim_layout.addWidget(self.clear_btn)

        layout.addWidget(self.sim_panel)

        # 实测点云：跳过建模与仿真，直接对实扫点云做解算
        cloud_box = QGroupBox("实测点云（直接测量）")
        cloud_layout = QVBoxLayout(cloud_box)
        self.pcd_btn = QPushButton("导入实测点云 PLY / PCD / XYZ")
        self.pcd_btn.clicked.connect(self._import_pcd)
        cloud_layout.addWidget(self.pcd_btn)

        self.pcd_label = QLabel("当前使用：未导入（走建模仿真）")
        self.pcd_label.setWordWrap(True)
        self.pcd_label.setStyleSheet("color: #666;")
        cloud_layout.addWidget(self.pcd_label)

        self.pcd_clear_btn = QPushButton("清除实测点云")
        self.pcd_clear_btn.clicked.connect(self._clear_pcd)
        self.pcd_clear_btn.setEnabled(False)
        cloud_layout.addWidget(self.pcd_clear_btn)
        layout.addWidget(cloud_box)

        self.cloud_panel = cloud_box
        self.ply_path = None
        self.pcd_path = None
        layout.addStretch()

        self.rb_sim.toggled.connect(self._apply_mode)
        self._apply_mode()

    # ---------------------------------------------------------------- 模式

    def use_cloud_mode(self) -> bool:
        return self.rb_cloud.isChecked()

    def _apply_mode(self, *args):
        """两种模式互斥：建模模式下禁用点云组，点云模式下禁用建模参数"""
        sim = self.rb_sim.isChecked()
        self.sim_panel.setEnabled(sim)
        self.cloud_panel.setEnabled(not sim)

    def _import_pcd(self):
        from core.point_io import EXT_FILTER
        path, _ = QFileDialog.getOpenFileName(self, "选择实测点云", "", EXT_FILTER)
        if path:
            self._clear_ply()                     # 两种数据源互斥
            self.pcd_path = path
            self.pcd_label.setText(f"当前使用：{os.path.basename(path)}")
            self.pcd_label.setStyleSheet("color: #1565C0;")
            self.pcd_clear_btn.setEnabled(True)
            self.rb_cloud.setChecked(True)        # 导入点云即切到点云模式
            self.tower_changed.emit()

    def _clear_pcd(self):
        self.pcd_path = None
        self.pcd_label.setText("当前使用：未导入（走建模仿真）")
        self.pcd_label.setStyleSheet("color: #666;")
        self.pcd_clear_btn.setEnabled(False)

    def _on_settlement_toggled(self, enabled):
        for sp in (self.sp_nw, self.sp_ne, self.sp_se, self.sp_sw):
            sp.setEnabled(enabled)
        # 叠加语义：沉降反算的基础倾角 + 上面的塔身附加倾斜，
        # 所以倾斜输入框始终可编辑，不再灰掉
        if enabled:
            zeros = all(abs(sp.value()) < 1e-9 for sp in
                        (self.sp_nw, self.sp_ne, self.sp_se, self.sp_sw))
            tip = ("已启用沉降：倾斜角改为由四角沉降反算。\n"
                   + ("当前四角全为 0，塔将按【竖直】处理，倾斜率为 0。"
                      if zeros else ""))
            self.use_settlement.setToolTip(tip)
        else:
            self.use_settlement.setToolTip(
                "勾选后忽略上面的 θx/θy 倾斜角，改由四个塔脚的独立沉降量决定基础倾斜。")
        if hasattr(self, "sett_warn"):
            self._update_sett_warn()

    def _update_sett_warn(self, *args):
        """勾选了沉降但四角全为 0 → 塔会被当成竖直的，必须提示"""
        if not self.use_settlement.isChecked():
            self.sett_warn.hide()
            return
        zeros = all(abs(sp.value()) < 1e-9 for sp in
                    (self.sp_nw, self.sp_ne, self.sp_se, self.sp_sw))
        if zeros:
            self.sett_warn.setText(
                "四角沉降全为 0：本次总倾角 = 上面的塔身附加倾斜（沉降不贡献）。")
            self.sett_warn.setStyleSheet(
                "background:#ECEFF1; color:#546E7A; padding:4px; border-radius:3px;")
            self.sett_warn.show()
        else:
            self.sett_warn.hide()

    def _convert_settlements(self):
        """把当前的塔身附加倾斜「归因」给四角沉降：填好沉降后把倾斜清零

        叠加语义下，若不把倾斜清零，换算出来的沉降会再叠一次，总倾角翻倍。
        """
        cfg = self.get_config()
        ne, se, sw, nw = settlements_from_tilt(cfg, self.tilt_x.value(), self.tilt_y.value())
        self.sp_ne.setValue(round(ne * 1000.0))
        self.sp_se.setValue(round(se * 1000.0))
        self.sp_sw.setValue(round(sw * 1000.0))
        self.sp_nw.setValue(round(nw * 1000.0))
        self.use_settlement.setChecked(True)
        # 倾斜已由沉降承担，清零避免重复叠加
        self.tilt_x.blockSignals(True); self.tilt_y.blockSignals(True)
        self.tilt_x.setValue(0.0); self.tilt_y.setValue(0.0)
        self.tilt_x.blockSignals(False); self.tilt_y.blockSignals(False)
        self._update_sett_warn()

    def _on_height_changed(self, value):
        """塔高变化时给出提示：截断上限需要跟着调整，否则中轴点会跑出塔身"""
        self.tower_changed.emit()

    def _import_ply(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 PLY 塔模型", "", "PLY Files (*.ply)")
        if path:
            self._clear_pcd()                     # 两种数据源互斥
            self.ply_path = path
            self.ply_label.setText(f"当前使用：{os.path.basename(path)}")
            self.ply_label.setStyleSheet("color: #1565C0;")
            self.clear_btn.setEnabled(True)
            self.rb_sim.setChecked(True)          # 导入塔模型即切回建模模式
            self.tower_changed.emit()

    def _clear_ply(self):
        self.ply_path = None
        self.ply_label.setText("当前使用：参数化建模")
        self.ply_label.setStyleSheet("color: #666;")
        self.clear_btn.setEnabled(False)

    def get_config(self) -> TowerConfig:
        use_sett = self.use_settlement.isChecked()
        return TowerConfig(
            total_height=self.height.value(),
            body_bottom_width=self.bottom_w.value(),
            body_top_width=self.top_w.value(),
            tilt_deg_x=self.tilt_x.value(),
            tilt_deg_y=self.tilt_y.value(),
            use_settlement=use_sett,
            settlement_nw=self.sp_nw.value() / 1000.0 if use_sett else 0.0,
            settlement_ne=self.sp_ne.value() / 1000.0 if use_sett else 0.0,
            settlement_se=self.sp_se.value() / 1000.0 if use_sett else 0.0,
            settlement_sw=self.sp_sw.value() / 1000.0 if use_sett else 0.0,
            bow_ratio=self.bow_ratio.value(),
        )


class LidarPanel(QWidget):
    """雷达配置面板"""
    lidar_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # 预设选择
        preset_box = QGroupBox("雷达预设")
        preset_form = QFormLayout(preset_box)
        self.preset_combo = QComboBox()
        self._load_presets()
        preset_form.addRow("型号:", self.preset_combo)
        layout.addWidget(preset_box)

        # 自定义参数
        custom_box = QGroupBox("自定义参数")
        custom_form = QFormLayout(custom_box)

        self.n_channels = QSpinBox()
        self.n_channels.setRange(4, 256)
        self.n_channels.setValue(16)
        custom_form.addRow("线数:", self.n_channels)

        self.fov = QDoubleSpinBox()
        self.fov.setRange(5, 90)
        self.fov.setValue(30.0)
        custom_form.addRow("垂直FOV(°):", self.fov)

        self.horiz_res = QDoubleSpinBox()
        self.horiz_res.setRange(0.05, 1.0)
        self.horiz_res.setDecimals(2)
        self.horiz_res.setValue(0.2)
        custom_form.addRow("水平分辨率(°):", self.horiz_res)

        self.noise = QDoubleSpinBox()
        self.noise.setRange(0.001, 0.05)
        self.noise.setDecimals(3)
        self.noise.setValue(0.01)
        custom_form.addRow("测距噪声(m):", self.noise)

        self.distance = QDoubleSpinBox()
        self.distance.setRange(5, 50)
        self.distance.setValue(15.0)
        custom_form.addRow("扫描距离(m):", self.distance)

        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        self.seed.setValue(42)
        self.seed.setSpecialValueText("每次随机")
        self.seed.setToolTip(
            "测距噪声的随机种子。\n"
            "固定值（如 42）= 同一组参数每次扫描结果完全一致，便于对比调试；\n"
            "0 = 每次运行换一组噪声，结果会有 ±0.05‰ 左右的抖动。")
        custom_form.addRow("随机种子:", self.seed)

        layout.addWidget(custom_box)

        # ---- 扫描站位：方位 + 层数 都可选 ----
        st_box = QGroupBox("扫描站位")
        st_layout = QVBoxLayout(st_box)

        grid = QGridLayout()
        grid.setSpacing(4)
        # (行, 列, 名称, 数学角 b)。坐标约定：+x=南、+y=东（见 core/frame.py）
        # b 从 +x（南）起逆时针：南0 东南45 东90 东北135 北180 西北225 西270 西南315
        spots = [(0, 0, "西北", 225), (0, 1, "北", 180), (0, 2, "东北", 135),
                 (1, 0, "西", 270), (1, 2, "东", 90),
                 (2, 0, "西南", 315), (2, 1, "南", 0), (2, 2, "东南", 45)]
        self.bearing_checks = {}
        for r, c, name, bdeg in spots:
            cb = QCheckBox(f"{name} {bdeg}°")
            cb.setChecked(bdeg in (0, 90, 180, 270))     # 默认四正方向
            cb.toggled.connect(self._update_station_info)
            self.bearing_checks[bdeg] = cb
            grid.addWidget(cb, r, c)
        hub = QLabel("塔位")
        hub.setAlignment(Qt.AlignCenter)
        hub.setStyleSheet("color: #888;")
        grid.addWidget(hub, 1, 1)
        st_layout.addLayout(grid)

        lay = QFormLayout()
        self.n_layers = QSpinBox()
        self.n_layers.setRange(1, 12)
        self.n_layers.setValue(5)
        self.n_layers.valueChanged.connect(self._update_station_info)
        lay.addRow("层数:", self.n_layers)

        self.z_first = QDoubleSpinBox()
        self.z_first.setRange(0.0, 80.0)
        self.z_first.setValue(6.0)
        self.z_first.setSuffix(" m")
        self.z_first.valueChanged.connect(self._update_station_info)
        lay.addRow("首层高度:", self.z_first)

        self.z_step = QDoubleSpinBox()
        self.z_step.setRange(1.0, 30.0)
        self.z_step.setValue(7.0)
        self.z_step.setSuffix(" m")
        self.z_step.valueChanged.connect(self._update_station_info)
        lay.addRow("层间距:", self.z_step)
        st_layout.addLayout(lay)

        self.station_info = QLabel("")
        self.station_info.setWordWrap(True)
        self.station_info.setStyleSheet("font-size: 11px; color: #555;")
        st_layout.addWidget(self.station_info)
        layout.addWidget(st_box)

        layout.addStretch()

        # 信号必须在填充预设之后连接：否则 addItems 触发的 currentTextChanged
        # 会在控件还没建好时回调，第一个预设也就不会生效
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        if self.preset_combo.count() > 0:
            self._on_preset_changed(self.preset_combo.currentText())

    def _load_presets(self):
        path = os.path.join(CONFIG_DIR, "radar_presets.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._presets = data["presets"]
        except (OSError, KeyError, json.JSONDecodeError) as e:
            self._presets = {"Helios 16": {"n_channels": 16, "vertical_fov_deg": 30.0,
                                           "horizontal_resolution_deg": 0.2, "range_sigma": 0.01}}
            print(f"读取雷达预设失败({e})，使用内置默认值")
        self.preset_combo.addItems(list(self._presets.keys()))

    def _on_preset_changed(self, name):
        p = self._presets.get(name)
        if not p:
            return
        self.n_channels.setValue(int(p.get("n_channels", 16)))
        self.fov.setValue(float(p.get("vertical_fov_deg", 30.0)))
        self.horiz_res.setValue(float(p.get("horizontal_resolution_deg", 0.2)))
        self.noise.setValue(float(p.get("range_sigma", 0.01)))
        self._update_station_info()

    def selected_bearings(self):
        bs = sorted(b for b, cb in self.bearing_checks.items() if cb.isChecked())
        return tuple(bs) if bs else (0.0, 90.0, 180.0, 270.0)

    def station_heights(self):
        z0 = self.z_first.value()
        dz = self.z_step.value()
        return tuple(round(z0 + i * dz, 2) for i in range(self.n_layers.value()))

    def _update_station_info(self, *args):
        nb = len([1 for cb in self.bearing_checks.values() if cb.isChecked()])
        nl = self.n_layers.value()
        heights = self.station_heights()
        n_station = nb * nl
        cfg = LidarConfig(n_channels=self.n_channels.value(),
                          horizontal_resolution_deg=self.horiz_res.value())
        rays = cfg.ray_count() * n_station
        warn = ""
        if not nb:
            warn = "  ← 未选方位，将按四正方向计算"
        elif rays > 2_000_000:
            warn = "  ← 射线过多，扫描会较慢"
        self.station_info.setText(
            f"{nb} 方位 × {nl} 层 = {n_station} 站位，"
            f"约 {rays/10000:.0f} 万条射线{warn}\n"
            f"层高：{heights[0]:.2f} ~ {heights[-1]:.2f} m")

    def get_config(self) -> LidarConfig:
        return LidarConfig(
            name=self.preset_combo.currentText(),
            n_channels=self.n_channels.value(),
            vertical_fov_deg=self.fov.value(),
            horizontal_resolution_deg=self.horiz_res.value(),
            range_sigma=self.noise.value(),
            scan_distance=self.distance.value(),
            station_bearings_deg=self.selected_bearings(),
            station_heights=self.station_heights(),
            random_seed=self.seed.value(),
        )


class SolverPanel(QWidget):
    """算法参数面板"""
    solver_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        param_box = QGroupBox("解算参数")
        form = QFormLayout(param_box)

        self.step = QDoubleSpinBox()
        self.step.setRange(0.3, 3.0)
        self.step.setDecimals(2)
        self.step.setSingleStep(0.05)
        self.step.setValue(1.0)
        form.addRow("切片步长(m):", self.step)

        self.window_h = QDoubleSpinBox()
        self.window_h.setRange(1.0, 5.0)
        self.window_h.setValue(2.0)
        form.addRow("窗口高度(m):", self.window_h)

        self.z_min = QDoubleSpinBox()
        self.z_min.setRange(0, 20)
        self.z_min.setValue(4.0)
        form.addRow("截断下限(m):", self.z_min)

        self.z_max = QDoubleSpinBox()
        self.z_max.setRange(10, 80)
        self.z_max.setValue(27.0)
        form.addRow("截断上限(m):", self.z_max)

        self.voxel = QDoubleSpinBox()
        self.voxel.setRange(0.01, 0.2)
        self.voxel.setDecimals(3)
        self.voxel.setValue(0.05)
        form.addRow("体素大小(m):", self.voxel)

        layout.addWidget(param_box)

        # ---- 核心算法：方形棱柱几何约束的全局抗差精化 ----
        refine_box = QGroupBox("全局精化（创新算法）")
        rform = QFormLayout(refine_box)
        self.use_refine = QCheckBox("启用棱柱几何约束精化")
        self.use_refine.setChecked(True)
        self.use_refine.setToolTip(
            "把「塔身是线性收窄的正方形棱柱」这一几何先验写进目标函数，\n"
            "对全部塔身点做一次鲁棒 Gauss-Newton 求解，替代逐层提点→拟合直线的两阶段流程。\n"
            "取消勾选可退回两阶段算法，用于对比实验。"
        )
        rform.addRow(self.use_refine)

        self.balance_power = QDoubleSpinBox()
        self.balance_power.setRange(0.0, 1.0)
        self.balance_power.setDecimals(2)
        self.balance_power.setSingleStep(0.1)
        self.balance_power.setValue(0.5)
        self.balance_power.setToolTip(
            "分面 × 分层的均衡加权强度：\n"
            "0 = 不均衡（充分利用过采样信息）\n"
            "1 = 完全均衡（最强抗密度偏斜）\n"
            "0.5 = 平方根折中（推荐）"
        )
        rform.addRow("分面均衡强度:", self.balance_power)
        layout.addWidget(refine_box)

        layout.addStretch()

    def get_config(self):
        from core.tilt_solver import SolverConfig
        return SolverConfig(
            step=self.step.value(),
            window_h=self.window_h.value(),
            z_min=self.z_min.value(),
            z_max=self.z_max.value(),
            voxel_size=self.voxel.value(),
            use_global_refine=self.use_refine.isChecked(),
            use_balance_weight=self.balance_power.value() > 0.0,
            balance_power=self.balance_power.value(),
        )
