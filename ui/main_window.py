# -*- coding: utf-8 -*-
"""
主窗口：PyVista嵌入式3D视图版
"""
import time
import traceback
import dataclasses
import os

import numpy as np
from PySide6.QtWidgets import (QMainWindow, QSplitter, QTabWidget, QToolBar,
                               QWidget, QVBoxLayout, QHBoxLayout, QLabel)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QAction

from core.tower_model import (build_tower, load_tower_from_ply, true_tilt,
                              true_indicators, _resolve_crossarm_levels,
                              _body_width_at, base_tilt_deg, settlement_tilt_deg)
from core.crossarm_analysis import analyze_crossarms
from core.standards import evaluate
from core.lidar_scanner import simulate_scan
from core.tilt_solver import solve
from ui.panel_params import TowerPanel, LidarPanel, SolverPanel
from ui.panel_result import ResultPanel
from ui.view_3d import PointCloudViewer
from ui.layer_panel import LayerPanel
from ui.panel_deform import DeformPanel


class WorkerThread(QThread):
    """后台仿真线程

    注意：这里用 done 而不是 finished —— finished 是 QThread 自带的信号（无参），
    覆盖成带参信号会让线程结束时的行为变得不可预测。
    """
    done = Signal(object, dict, object)
    failed = Signal(str)
    log = Signal(str)

    def __init__(self, tower_cfg, lidar_cfg, solver_cfg, ply_path=None, pcd_path=None):
        super().__init__()
        self.tower_cfg = tower_cfg
        self.lidar_cfg = lidar_cfg
        self.solver_cfg = solver_cfg
        self.ply_path = ply_path
        self.pcd_path = pcd_path
        self.cloud_height = None

    def run(self):
        try:
            if self.pcd_path:
                self._run_imported_cloud()
                return

            self.log.emit("正在生成塔模型...")
            if self.ply_path:
                tower = load_tower_from_ply(self.ply_path)
            else:
                tower = build_tower(self.tower_cfg)
            self.log.emit(f"塔网格: {len(tower.triangles)} 三角面")

            self.log.emit("正在执行激光扫描...")
            t0 = time.time()
            # 固定种子 → 同一配置的扫描结果可复现，便于对比"改了什么"
            seed = int(getattr(self.lidar_cfg, "random_seed", 0) or 0)
            if seed:
                np.random.seed(seed)
                self.log.emit(f"随机种子 {seed}（结果可复现）")
            else:
                self.log.emit("随机种子 0（每次噪声不同）")
            pcd = simulate_scan(tower, self.lidar_cfg)
            self.log.emit(f"扫描完成: {len(pcd.points)} 点, 耗时 {time.time()-t0:.1f}s")

            self.log.emit("正在解算倾斜...")
            t1 = time.time()
            result, intermediates = solve(pcd, self.solver_cfg)
            self.log.emit(f"解算耗时 {time.time()-t1:.1f}s")
            self.log.emit(f"解算完成: 总倾斜角 {result.total_tilt_deg:.3f}°, "
                          f"倾斜率 {result.tilt_rate_permille:.2f}‰")
            self.done.emit(result, intermediates, tower)
        except Exception as e:
            self.log.emit(f"错误: {e}")
            self.log.emit(traceback.format_exc())
            self.failed.emit(str(e))

    def _run_imported_cloud(self):
        """实测点云：跳过建模与扫描，直接解算"""
        from core.point_io import load_point_cloud, normalize_cloud, suggest_crop

        self.log.emit("正在读取实测点云...")
        raw = load_point_cloud(self.pcd_path)
        n_raw = len(raw.points)
        pcd, height = normalize_cloud(raw)
        self.cloud_height = height
        self.log.emit(f"点云 {n_raw} 点 → 规范化后 {len(pcd.points)} 点，"
                      f"塔高约 {height:.1f} m（已对齐 Z 轴、塔基归零）")

        cfg = dataclasses.replace(self.solver_cfg)
        z_lo, z_hi = suggest_crop(height)
        if not (0.0 <= cfg.z_min < cfg.z_max <= height * 1.05):
            cfg.z_min, cfg.z_max = z_lo, z_hi
            self.log.emit(f"截断区间与点云不匹配，已自动设为 {z_lo}~{z_hi} m")
        cfg.tower_height = height
        self.solver_cfg = cfg

        self.log.emit("正在解算倾斜...")
        t0 = time.time()
        result, intermediates = solve(pcd, cfg)
        self.log.emit(f"解算耗时 {time.time()-t0:.1f}s")
        self.log.emit(f"解算完成: 倾斜率 α {result.alpha_standard_permille:.2f}‰")
        self.done.emit(result, intermediates, None)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("杆塔倾斜激光测量系统 v2.1　作者：白浪淘沙")
        self.resize(1600, 900)
        self._worker = None
        self._true_pack = None
        self._eval_true = None
        self._tower_height = 30.0
        self._voltage = "110kV"
        self._tower_cfg = None
        self._has_result = False
        self._preview_first = True
        self._spline1 = None          # 第一期塔轴（样条），用于两期对比
        self._alpha1 = 0.0
        self._delta1 = 0.0
        # 塔模型预览的防抖定时器：改参数时不至于每敲一下就重建一次网格
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(250)
        self._preview_timer.timeout.connect(self.refresh_tower_preview)
        self._init_ui()

    def _init_ui(self):
        # 左侧参数面板
        self.tower_panel = TowerPanel()
        self.lidar_panel = LidarPanel()
        self.solver_panel = SolverPanel()
        left_tabs = QTabWidget()
        left_tabs.addTab(self.tower_panel, "塔模型")
        left_tabs.addTab(self.lidar_panel, "雷达")
        left_tabs.addTab(self.solver_panel, "算法")
        self.deform_panel = DeformPanel()
        left_tabs.addTab(self.deform_panel, "变形对比")

        # 中央3D视图
        self.viewer = PointCloudViewer()

        # 右侧：结果面板 + 图层面板
        right_splitter = QSplitter(Qt.Vertical)
        self.result_panel = ResultPanel()
        self.layer_panel = LayerPanel()
        right_splitter.addWidget(self.result_panel)
        right_splitter.addWidget(self.layer_panel)
        right_splitter.setSizes([460, 300])

        # 主分割
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_tabs)
        splitter.addWidget(self.viewer)
        splitter.addWidget(right_splitter)
        splitter.setSizes([280, 900, 340])

        # 顶部署名条
        header = QWidget()
        header.setObjectName("headerBar")
        hb = QHBoxLayout(header)
        hb.setContentsMargins(18, 9, 18, 9)
        title = QLabel("杆塔倾斜激光测量系统")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #1A237E;")
        subtitle = QLabel("无人机激光点云 · 倾斜率与变形分析")
        subtitle.setStyleSheet("font-size: 12px; color: #5C6BC0;")
        author = QLabel("作者：白浪淘沙")
        author.setStyleSheet("font-size: 13px; font-weight: bold; color: #B71C1C;")
        hb.addWidget(title)
        hb.addSpacing(12)
        hb.addWidget(subtitle)
        hb.addStretch()
        hb.addWidget(author)
        header.setStyleSheet(
            "#headerBar { background: #E8EAF6; border-bottom: 1px solid #C5CAE9; }")

        central = QWidget()
        vbox = QVBoxLayout(central)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        vbox.addWidget(header)
        vbox.addWidget(splitter, 1)
        self.setCentralWidget(central)

        # 工具栏
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        self.act_run = QAction("运行仿真", self)
        self.act_run.triggered.connect(self.run_simulation)
        toolbar.addAction(self.act_run)

        # 图层信号连接
        self.layer_panel.layer_visible_changed.connect(self.viewer.set_layer_visible)
        self.layer_panel.layer_color_changed.connect(self.viewer.set_layer_color)

        self.deform_panel.compare_requested.connect(self.run_epoch_compare)

        # 塔高变了就把截断上限跟着挪（默认 0.9 倍塔高），否则高塔只会算到腰部
        self.tower_panel.height.valueChanged.connect(self._on_tower_height_changed)

        # 塔模型实时预览：进软件就能看到塔，改参数也会跟着变
        self._wire_tower_preview()

        self.setStyleSheet("""
            QMainWindow { background: #f5f5f5; }
            QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 4px; margin-top: 8px; padding-top: 8px; }
        """)

        # 等窗口先画出来，再生成塔模型，避免启动时白屏
        QTimer.singleShot(0, self.refresh_tower_preview)

    # ---------------------------------------------------------------- 塔模型预览

    def _wire_tower_preview(self):
        tp = self.tower_panel
        for w in (tp.height, tp.bottom_w, tp.top_w, tp.tilt_x, tp.tilt_y,
                  tp.bow_ratio, tp.sp_nw, tp.sp_ne, tp.sp_se, tp.sp_sw):
            w.valueChanged.connect(self._schedule_preview)
        tp.use_settlement.toggled.connect(self._schedule_preview)
        tp.import_btn.clicked.connect(self._schedule_preview)
        tp.clear_btn.clicked.connect(self._schedule_preview)
        tp.pcd_btn.clicked.connect(self._schedule_preview)
        tp.pcd_clear_btn.clicked.connect(self._schedule_preview)
        tp.rb_sim.toggled.connect(self._schedule_preview)
        tp.rb_sim.toggled.connect(self._update_run_label)
        tp.rb_cloud.toggled.connect(self._schedule_preview)
        tp.rb_cloud.toggled.connect(self._update_run_label)
        tp.tower_changed.connect(self._schedule_preview)

    def _run_label(self) -> str:
        return "运行解算（实测点云）" if self.tower_panel.use_cloud_mode() else "运行仿真"

    def _update_run_label(self, *args):
        if self._worker is None or not self._worker.isRunning():
            self.act_run.setText(self._run_label())

    def _schedule_preview(self, *args):
        self._preview_timer.start()

    def refresh_tower_preview(self):
        """按当前参数重建塔模型并显示（不需要先跑仿真）"""
        if self._worker is not None and self._worker.isRunning():
            return                      # 仿真进行中不要动场景
        if self.tower_panel.use_cloud_mode():
            if self.tower_panel.pcd_path:
                self._show_cloud_preview()
            else:
                self.viewer.clear_all()
                self.layer_panel.clear_layers()
                self._has_result = False
                self.viewer.fit_view()
            return
        try:
            if self.tower_panel.ply_path:
                mesh = load_tower_from_ply(self.tower_panel.ply_path)
            else:
                mesh = build_tower(self.tower_panel.get_config())
        except Exception as e:
            self.result_panel.append_log(f"塔模型预览失败: {e}")
            return
        if mesh is None or len(mesh.triangles) == 0:
            return

        # 参数一改，上一次的仿真结果就失效了，清掉避免出现"新塔 + 旧点云"的错配
        if self._has_result:
            self.viewer.clear_all()
            self.layer_panel.clear_layers()
            self._has_result = False
            self._spline1 = None
            self.deform_panel.set_epoch1("未解算（先跑一次解算作为第一期）")
            self.result_panel.append_log("塔模型/数据源已变更，请重新运行")

        verts = np.asarray(mesh.vertices)
        tris = np.asarray(mesh.triangles)
        color = (0.62, 0.68, 0.78)
        self.viewer.add_mesh("塔网格", verts, tris, color=color,
                             visible=True, opacity=0.85)
        self.layer_panel.add_layer("塔网格", visible=True, color=color)
        self.viewer.fit_view()

        if self._preview_first:
            self._preview_first = False
            self.result_panel.append_log(
                f"已载入默认塔模型（{len(tris)} 面），可直接查看；"
                "点击「运行仿真」开始点云测量")

    def _show_cloud_preview(self):
        """导入实测点云后直接预览（不必先跑解算）"""
        from core.point_io import load_point_cloud, normalize_cloud
        try:
            pcd = load_point_cloud(self.tower_panel.pcd_path)
            pcd, height = normalize_cloud(pcd, max_points=400_000)
        except Exception as e:
            self.result_panel.append_log(f"点云预览失败: {e}")
            return

        self.viewer.clear_all()
        self.layer_panel.clear_layers()
        self._has_result = False
        self._spline1 = None
        self.deform_panel.set_epoch1("未解算（先跑一次解算作为第一期）")

        pts = np.asarray(pcd.points)
        self.viewer.add_point_cloud("实测点云", pts, color=(0.55, 0.72, 0.92),
                                    visible=True)
        self.layer_panel.add_layer("实测点云", visible=True, color=(0.55, 0.72, 0.92))
        self.viewer.fit_view()
        self.result_panel.append_log(
            f"已载入实测点云：预览 {len(pts)} 点，塔高约 {height:.1f} m；"
            "点击「运行解算」开始解算")

    # ---------------------------------------------------------------- 运行

    def _on_tower_height_changed(self, height: float):
        """塔高变化时同步截断上限与下限，避免只解算到塔腰"""
        self.solver_panel.z_max.setValue(round(height * 0.9, 1))
        self.solver_panel.z_min.setValue(round(max(1.0, height * 0.13), 1))

    def run_simulation(self):
        # 关键：上一次线程还在跑时直接返回。
        # 之前这里是"覆盖 self._worker"，旧 QThread 引用计数归零被销毁，
        # 而它的 C++ 对象还在运行 -> 进程直接崩溃退出（无任何报错）。
        if self._worker is not None and self._worker.isRunning():
            self.result_panel.append_log("上一次仿真仍在运行，请稍候…")
            return

        # 停掉还没触发的预览刷新，否则它可能在仿真结束后才执行、把结果清掉
        self._preview_timer.stop()

        tower_cfg = self.tower_panel.get_config()
        lidar_cfg = self.lidar_panel.get_config()
        solver_cfg = self.solver_panel.get_config()
        ply_path = self.tower_panel.ply_path
        cloud_mode = self.tower_panel.use_cloud_mode()
        pcd_path = self.tower_panel.pcd_path if cloud_mode else None
        if cloud_mode and not pcd_path:
            self.result_panel.append_log("当前是实测点云模式，但还没有导入点云文件")
            return
        if cloud_mode:
            ply_path = None

        # 把塔高交给解算器，用于把倾斜角折算成标准口径的塔顶偏移
        solver_cfg.tower_height = tower_cfg.total_height
        self._tower_height = tower_cfg.total_height
        self._voltage = self.tower_panel.voltage.currentText()

        if ply_path or pcd_path:
            self._true_pack = None
            self._eval_true = None
            self._tower_cfg = None
        else:
            # 启用沉降时，把「塔身附加倾斜 + 沉降基础倾斜 = 合成」写进日志，
            # 便于核对两者是否按预期叠加
            if tower_cfg.has_settlement():
                sx, sy = settlement_tilt_deg(tower_cfg)
                gx, gy = base_tilt_deg(tower_cfg)
                self.result_panel.append_log(
                    f"倾角合成：塔身附加 ({tower_cfg.tilt_deg_x:+.3f}°, "
                    f"{tower_cfg.tilt_deg_y:+.3f}°) + 沉降基础 ({sx:+.3f}°, {sy:+.3f}°)"
                    f" = 总 ({gx:+.3f}°, {gy:+.3f}°)")
            alpha, delta, tip, _tx, _ty = true_indicators(tower_cfg)
            # X/Y 方向角取与算法同口径的最小二乘轴线方向（带弯曲时与塔基倾角不同）
            lx, ly, _tot, _rate = true_tilt(tower_cfg, solver_cfg.z_min, solver_cfg.z_max)
            self._true_pack = {"alpha": alpha, "delta": delta, "tip": tip,
                               "tilt_x": lx, "tilt_y": ly}
            self._eval_true = evaluate(alpha, delta, tower_cfg.total_height, self._voltage)
            self._tower_cfg = tower_cfg

        self.result_panel.append_log("=" * 40)
        if pcd_path:
            self.result_panel.append_log(
                f"开始解算实测点云: {os.path.basename(pcd_path)}")
        else:
            self.result_panel.append_log(
                f"开始运行: {lidar_cfg.name}, {lidar_cfg.n_channels}线")

        self.act_run.setEnabled(False)
        self.act_run.setText("运行中…")

        self._worker = WorkerThread(tower_cfg, lidar_cfg, solver_cfg, ply_path,
                                    self.tower_panel.pcd_path)
        self._worker.log.connect(self.result_panel.append_log)
        self._worker.done.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_failed(self, msg):
        self.result_panel.append_log(f"仿真失败: {msg}")
        self._set_running(False)

    def _set_running(self, running: bool):
        self.act_run.setEnabled(not running)
        self.act_run.setText("运行中…" if running else self._run_label())

    def _on_finished(self, result, intermediates, tower_mesh):
        # 实测点云的实际高度回填，标准判定要用
        w = getattr(self, "_worker", None)
        if w is not None and getattr(w, "cloud_height", None):
            self._tower_height = w.cloud_height

        eval_meas = evaluate(result.alpha_standard_permille,
                             result.foundation_tilt_permille,
                             getattr(self, "_tower_height", 30.0),
                             getattr(self, "_voltage", "110kV"))

        # 横担健康：用原本要剔除的横担点云反演几何指标
        arm_reports = None
        cfg = getattr(self, "_tower_cfg", None)
        if cfg is not None and intermediates.get("axis_params") is not None:
            try:
                arm_reports = analyze_crossarms(
                    np.asarray(intermediates["crossarm_cloud"].points),
                    intermediates["axis_params"],
                    _resolve_crossarm_levels(cfg),
                    lambda h: _body_width_at(h, cfg) / 2.0,
                    cfg.crossarm_length,
                )
                bad = [r for r in arm_reports if not r.ok]
                if bad:
                    self.result_panel.append_log(
                        "横担异常: " + "；".join(
                            f"{r.height:.0f}m " +
                            "、".join(s for s, st in ((("左侧", r.left.status),
                                                       ("右侧", r.right.status)))
                                     if st != "正常")
                            for r in bad))
            except Exception as e:
                self.result_panel.append_log(f"横担分析失败: {e}")
                arm_reports = None

        self.result_panel.update_result(result, eval_meas,
                                        getattr(self, "_eval_true", None),
                                        getattr(self, "_true_pack", None),
                                        arm_reports)
        self._has_result = True
        self._spline1 = intermediates.get("axis_spline")
        self._alpha1 = result.alpha_standard_permille
        self._delta1 = result.foundation_tilt_permille
        if self._spline1 is not None:
            self.deform_panel.set_epoch1(
                f"当前点云（α={self._alpha1:.2f}‰，塔高 {self._tower_height:.1f} m）")
        self.result_panel.append_log(
            f"标准判定: α={result.alpha_standard_permille:.2f}‰ "
            f"(限值{eval_meas.operation_limit:.0f}‰) → {eval_meas.grade}；"
            f"δ={result.foundation_tilt_permille:.2f}‰ "
            f"(允许{eval_meas.foundation_limit:.1f}‰)"
        )
        self._set_running(False)

        self.viewer.clear_all()
        self.layer_panel.clear_layers()

        # 塔网格
        if tower_mesh is not None and len(tower_mesh.triangles) > 0:
            verts = np.asarray(tower_mesh.vertices)
            tris = np.asarray(tower_mesh.triangles)
            self.viewer.add_mesh("塔网格", verts, tris, color=(0.6, 0.6, 0.6),
                                 visible=True, opacity=0.35)
            self.layer_panel.add_layer("塔网格", visible=True, color=(0.6, 0.6, 0.6))

        # 原始点云
        pts = np.asarray(intermediates["preprocessed"].points)
        self.viewer.add_point_cloud("原始点云", pts, color=(0.75, 0.75, 0.75), visible=True)
        self.layer_panel.add_layer("原始点云", visible=True, color=(0.75, 0.75, 0.75))

        # 去横担后点云
        pts = np.asarray(intermediates["crossarm_removed"].points)
        self.viewer.add_point_cloud("去横担点云", pts, color=(0.2, 0.5, 0.9), visible=False)
        self.layer_panel.add_layer("去横担点云", visible=False, color=(0.2, 0.5, 0.9))

        # 中轴点
        axis_pts = np.asarray(intermediates["axis_points"].points)
        if len(axis_pts) > 0:
            self.viewer.add_point_cloud("中轴点", axis_pts, color=(1.0, 0.15, 0.15), visible=True)
            self.layer_panel.add_layer("中轴点", visible=True, color=(1.0, 0.15, 0.15))

        # 拟合中心线（折线，不是两个点）
        line_pts = np.asarray(intermediates["fitted_line"].points)
        if len(line_pts) >= 2:
            self.viewer.add_lines("拟合中心线", line_pts, color=(0.0, 0.85, 0.25), width=5.0,
                                  visible=True)
            self.layer_panel.add_layer("拟合中心线", visible=True, color=(0.0, 0.85, 0.25))

        self.viewer.fit_view()

    # ---------------------------------------------------------------- 两期对比

    def run_epoch_compare(self):
        """导入第二期点云 → 独立重建塔轴 → 轴对轴比较（免配准）"""
        from core.point_io import load_point_cloud, normalize_cloud, suggest_crop
        from core.tilt_solver import SolverConfig, solve as solve_cloud
        from core.deform_analysis import compare_epochs

        if self._spline1 is None:
            self.result_panel.append_log("请先运行一次解算，作为第一期数据")
            return
        path = self.deform_panel.epoch2_path
        if not path:
            self.result_panel.append_log("请先导入第二期点云")
            return

        try:
            raw = load_point_cloud(path)
            cloud2, h2 = normalize_cloud(raw)
            cfg2 = SolverConfig()
            cfg2.z_min, cfg2.z_max = suggest_crop(h2)
            cfg2.tower_height = h2
            cfg2.use_spline = True
            r2, inter2 = solve_cloud(cloud2, cfg2)
            sp2 = inter2.get("axis_spline")
            if sp2 is None:
                self.result_panel.append_log("第二期点云不足以重建塔轴")
                return

            days = float(self.deform_panel.days.value())
            cp = compare_epochs(self._spline1, sp2, self._tower_height, days=days)
            self.deform_panel.show_result(cp)
            self.result_panel.append_log(
                f"两期对比（间隔 {days:.0f} 天）：Δα {cp.delta_alpha:+.2f}‰，"
                f"塔顶增量 {cp.tip_shift_mm:+.1f} mm，"
                f"年化 {cp.rate_mm_per_year:.0f} mm/年")

            # 第二期的塔轴画出来，便于目视对比
            self.viewer.add_point_cloud("第二期点云",
                                        np.asarray(inter2["crossarm_removed"].points),
                                        color=(0.95, 0.55, 0.2), visible=True)
            self.layer_panel.add_layer("第二期点云", visible=True, color=(0.95, 0.55, 0.2))
            self.viewer.add_lines("第二期塔轴",
                                  sp2.world(np.linspace(sp2.z0, sp2.z1, 30)),
                                  color=(1.0, 0.45, 0.0), width=5.0, visible=True)
            self.layer_panel.add_layer("第二期塔轴", visible=True, color=(1.0, 0.45, 0.0))
            self.viewer.fit_view()
        except Exception as e:
            traceback.print_exc()
            self.result_panel.append_log(f"两期对比失败: {e}")

    # ---------------------------------------------------------------- 生命周期

    def closeEvent(self, event):
        """窗口关闭时等后台线程收尾，避免线程运行中被析构导致崩溃"""
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait(5000)
        super().closeEvent(event)
