# -*- coding: utf-8 -*-
"""
右侧结果面板：测量结果 + 标准符合性判定 + 真值对比 + 日志
"""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QLabel,
                                  QGroupBox, QTextEdit, QScrollArea)
from PySide6.QtCore import Qt

from core.standards import GRADE_COLORS


def _bold(size=14):
    lab = QLabel("--")
    lab.setStyleSheet(f"font-size: {size}px; font-weight: bold;")
    return lab


class ResultPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ---------------- 测量结果 ----------------
        res_box = QGroupBox("倾斜测量结果")
        form = QFormLayout(res_box)
        self.val_x = _bold(13)
        form.addRow("南北向倾斜角:", self.val_x)
        self.val_y = _bold(13)
        form.addRow("东西向倾斜角:", self.val_y)
        self.val_total = _bold(14)
        self.val_total.setStyleSheet("font-size: 14px; font-weight: bold; color: #1565C0;")
        form.addRow("总倾斜角:", self.val_total)
        self.val_rate = _bold(16)
        self.val_rate.setStyleSheet("font-size: 16px; font-weight: bold; color: #1565C0;")
        form.addRow("倾斜率 α:", self.val_rate)
        self.val_tip = _bold(13)
        form.addRow("塔顶水平偏移:", self.val_tip)
        layout.addWidget(res_box)

        # ---------------- 标准符合性 ----------------
        std_box = QGroupBox("标准符合性判定")
        sf = QFormLayout(std_box)
        self.val_alpha_std = _bold(13)
        sf.addRow("α 运行限值:", self.val_alpha_std)
        self.val_grade = _bold(14)
        sf.addRow("状态等级:", self.val_grade)
        self.val_delta = _bold(13)
        sf.addRow("基础倾斜率 δ:", self.val_delta)
        self.val_delta_limit = QLabel("--")
        sf.addRow("δ 设计允许值:", self.val_delta_limit)
        self.val_hazard = _bold(13)
        sf.addRow("重大隐患:", self.val_hazard)
        layout.addWidget(std_box)

        self.alert_label = QLabel("")
        self.alert_label.setAlignment(Qt.AlignCenter)
        self.alert_label.setWordWrap(True)
        self.alert_label.setStyleSheet("font-size: 12px; font-weight: bold; padding: 6px;")
        layout.addWidget(self.alert_label)

        # ---------------- 真值对比 ----------------
        cmp_box = QGroupBox("真值对比（仿真模式）")
        cf = QFormLayout(cmp_box)
        self.val_true_tilt = QLabel("--")
        cf.addRow("真值 南-北 / 东-西:", self.val_true_tilt)
        self.val_true_alpha = QLabel("--")
        cf.addRow("真值 α:", self.val_true_alpha)
        self.val_err_alpha = QLabel("--")
        self.val_err_alpha.setStyleSheet("font-weight: bold;")
        cf.addRow("α 误差:", self.val_err_alpha)
        self.val_true_delta = QLabel("--")
        cf.addRow("真值 δ:", self.val_true_delta)
        self.val_err_delta = QLabel("--")
        self.val_err_delta.setStyleSheet("font-weight: bold;")
        cf.addRow("δ 误差:", self.val_err_delta)
        self.val_true_grade = QLabel("--")
        cf.addRow("真值状态等级:", self.val_true_grade)
        layout.addWidget(cmp_box)

        # ---------------- 算法统计 ----------------
        stat_box = QGroupBox("算法统计")
        sf2 = QFormLayout(stat_box)
        self.val_axis = QLabel("--")
        sf2.addRow("中轴点 / 内点:", self.val_axis)
        self.val_conf = QLabel("--")
        sf2.addRow("置信度:", self.val_conf)
        self.val_curve = QLabel("--")
        sf2.addRow("轴线弯曲量:", self.val_curve)
        self.val_refine = QLabel("--")
        self.val_refine.setWordWrap(True)
        sf2.addRow("全局精化:", self.val_refine)
        self.val_quality = QLabel("--")
        sf2.addRow("数据质量:", self.val_quality)
        self.val_deflect = QLabel("--")
        sf2.addRow("弯曲矢高:", self.val_deflect)
        layout.addWidget(stat_box)

        # 数据质量告警条（不合格时醒目提示）
        self.quality_banner = QLabel("")
        self.quality_banner.setWordWrap(True)
        self.quality_banner.setAlignment(Qt.AlignCenter)
        self.quality_banner.hide()
        layout.addWidget(self.quality_banner)

        # ---------------- 弯曲定位 ----------------
        bend_box = QGroupBox("弯曲定位（基元拟合 + 变点检测）")
        bf = QFormLayout(bend_box)
        self.val_bend = QLabel("--")
        self.val_bend.setStyleSheet("font-weight: bold; font-size: 14px;")
        bf.addRow("判定:", self.val_bend)
        self.val_bend_pos = QLabel("--")
        bf.addRow("折角高度:", self.val_bend_pos)
        self.val_bend_delta = QLabel("--")
        bf.addRow("折点斜率增量:", self.val_bend_delta)
        self.val_spline = QLabel("--")
        bf.addRow("样条轴线残差:", self.val_spline)
        layout.addWidget(bend_box)

        # ---------------- 横担健康 ----------------
        arm_box = QGroupBox("横担健康（利用原本剔除的横担点云）")
        arm_layout = QVBoxLayout(arm_box)
        self.val_crossarm = QLabel("--")
        self.val_crossarm.setTextFormat(Qt.RichText)
        self.val_crossarm.setStyleSheet("font-size: 12px;")
        self.val_crossarm.setWordWrap(True)
        arm_layout.addWidget(self.val_crossarm)
        self.arm_note = QLabel("※ 臂长/对称性为水平量，可靠；「无有效点」可能是塔材缺失，也可能是遮挡，需复核；端部垂直下垂受遮挡影响，不作判定")
        self.arm_note.setStyleSheet("font-size: 11px; color: #888;")
        self.arm_note.setWordWrap(True)
        arm_layout.addWidget(self.arm_note)
        layout.addWidget(arm_box)

        layout.addStretch()
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        # ---------------- 日志 ----------------
        log_box = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_box)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(110)
        log_layout.addWidget(self.log)
        outer.addWidget(log_box, 0)

    # ---------------------------------------------------------------- 更新

    def update_result(self, result, eval_meas=None, eval_true=None, true_pack=None,
                      crossarm_reports=None):
        self.val_x.setText(f"{result.tilt_x_deg:.3f}°")
        self.val_y.setText(f"{result.tilt_y_deg:.3f}°")
        self.val_total.setText(f"{result.total_tilt_deg:.3f}°")
        self.val_rate.setText(f"{result.alpha_standard_permille:.2f}‰")
        self.val_tip.setText(f"{result.tip_offset_mm:.1f} mm")

        self.val_axis.setText(f"{result.num_axis_points} / {result.num_inliers}")
        self.val_conf.setText(f"{result.confidence:.2f}")
        self.val_curve.setText(f"{result.axis_curve_permille:+.2f}‰")
        if result.refine_used:
            self.val_refine.setText(
                f"已启用，{result.refine_points} 点参与\n"
                f"内符合精度 ±{result.alpha_sigma_permille:.2f}‰")
            self.val_refine.setStyleSheet("color: #2E7D32; font-size: 12px;")
        else:
            self.val_refine.setText("未启用（两阶段算法）")
            self.val_refine.setStyleSheet("color: #888; font-size: 12px;")

        # ---- 标准符合性 ----
        if eval_meas is not None:
            self.val_alpha_std.setText(
                f"{eval_meas.alpha_permille:.2f}‰  /  限值 {eval_meas.operation_limit:.0f}‰"
            )
            bg, fg = GRADE_COLORS[eval_meas.grade_index]
            self.val_grade.setText(f"{eval_meas.grade}（{eval_meas.alert_color}）")
            self.val_grade.setStyleSheet(
                f"font-size: 14px; font-weight: bold; color: {fg}; "
                f"background: {bg}; padding: 3px 6px; border-radius: 4px;")

            self.val_delta.setText(f"{eval_meas.foundation_permille:.2f}‰")
            self.val_delta_limit.setText(
                f"≤ {eval_meas.foundation_limit:.1f}‰"
                f"{'' if eval_meas.foundation_ok else '   ← 超限'}")

            if eval_meas.major_hazard_limit is None:
                self.val_hazard.setText("不适用（该电压等级不在 41 号令范围）")
                self.val_hazard.setStyleSheet("font-size: 12px; color: #666;")
            else:
                self.val_hazard.setText(
                    f"{'是 ← 重大事故隐患' if eval_meas.major_hazard else '否'}"
                    f"  （红线 {eval_meas.major_hazard_limit:.0f}‰）")
                color = "#C62828" if eval_meas.major_hazard else "#2E7D32"
                self.val_hazard.setStyleSheet(
                    f"font-size: 13px; font-weight: bold; color: {color};")

            self.alert_label.setText(eval_meas.summary())
            self.alert_label.setStyleSheet(
                f"font-size: 12px; font-weight: bold; padding: 6px; "
                f"background: {bg}; color: {fg};")
        else:
            for lab in (self.val_alpha_std, self.val_grade, self.val_delta,
                        self.val_delta_limit, self.val_hazard):
                lab.setText("--")
            self.alert_label.setText("")

        # ---- 真值对比 ----
        if true_pack is not None and eval_true is not None:
            self.val_true_tilt.setText(f"{true_pack['tilt_x']:.2f}° / {true_pack['tilt_y']:.2f}°")
            self.val_true_alpha.setText(f"{true_pack['alpha']:.2f}‰")
            err_a = result.alpha_standard_permille - true_pack["alpha"]
            self.val_err_alpha.setText(f"{err_a:+.2f}‰")
            self.val_err_alpha.setStyleSheet(
                f"font-weight: bold; color: {self._err_color(err_a)};")

            self.val_true_delta.setText(f"{true_pack['delta']:.2f}‰")
            err_d = result.foundation_tilt_permille - true_pack["delta"]
            self.val_err_delta.setText(f"{err_d:+.2f}‰")
            self.val_err_delta.setStyleSheet(
                f"font-weight: bold; color: {self._err_color(err_d)};")

            same = (eval_true.grade == eval_meas.grade) if eval_meas is not None else False
            self.val_true_grade.setText(
                f"{eval_true.grade}（判定{'一致 ✓' if same else '不一致 ✗'}）")
            self.val_true_grade.setStyleSheet(
                "color: #2E7D32;" if same else "color: #C62828;")
        else:
            self.val_true_tilt.setText("--（导入模型无真值）")
            for lab in (self.val_true_alpha, self.val_err_alpha, self.val_true_delta,
                        self.val_err_delta, self.val_true_grade):
                lab.setText("--")

        # ---- 弯曲定位 ----
        if result.bend_found:
            self.val_bend.setText(f"检出局部弯曲（{result.bend_grade}）")
            self.val_bend.setStyleSheet("font-weight: bold; font-size: 14px; color: #C62828;")
            self.val_bend_pos.setText(f"{result.bend_height:.1f} m 处")
            self.val_bend_delta.setText(f"{result.bend_delta_permille:+.2f}‰")
        else:
            self.val_bend.setText("未检出明显局部弯曲")
            self.val_bend.setStyleSheet("font-weight: bold; font-size: 14px; color: #2E7D32;")
            self.val_bend_pos.setText("--")
            self.val_bend_delta.setText("--")
        self.val_spline.setText(f"{result.spline_rms_mm:.0f} mm" if result.spline_rms_mm else "--")

        # ---- 数据质量 ----
        q = getattr(result, "quality", "未知")
        qr = getattr(result, "quality_ratio", 0.0)
        self.val_quality.setText(f"{q}（残差/半宽 = {qr:.3f}）")
        color = {"合格": "#2E7D32", "可疑": "#E65100", "不合格": "#C62828"}.get(q, "#666")
        self.val_quality.setStyleSheet(f"color: {color}; font-weight: bold;")
        if q == "不合格":
            self.quality_banner.setText(
                "⚠ 点云不满足杆塔棱柱特征，测量结果不可信\n"
                "（常见的点云坐标错位、截断区间不对、或导入的不是杆塔）")
            self.quality_banner.setStyleSheet(
                "background:#FFEBEE; color:#C62828; font-weight:bold; "
                "padding:8px; border:1px solid #C62828;")
            self.quality_banner.show()
        elif q == "可疑":
            self.quality_banner.setText("点云质量可疑，结果仅供参考")
            self.quality_banner.setStyleSheet(
                "background:#FFF3E0; color:#E65100; font-weight:bold; padding:6px;")
            self.quality_banner.show()
        else:
            self.quality_banner.hide()

        dmm = getattr(result, "deflection_mm", 0.0)
        dr = getattr(result, "deflection_ratio_permille", 0.0)
        self.val_deflect.setText(f"{dmm:.1f} mm（矢跨比 {dr:.2f}‰）" if dmm else "--")

        # ---- 横担健康 ----
        if crossarm_reports:
            rows = []
            for r in crossarm_reports:
                cl = "#2E7D32" if r.left.status == "正常" else ("#E65100" if r.left.status == "偏短" else "#C62828")
                cr = "#2E7D32" if r.right.status == "正常" else ("#E65100" if r.right.status == "偏短" else "#C62828")
                rows.append(
                    f'<span style="color:#444">{r.height:5.1f}m</span>　'
                    f'L <b>{r.left.arm_len:.2f}</b> / R <b>{r.right.arm_len:.2f}</b> m　'
                    f'<span style="color:{cl}">{r.left.status}</span>'
                    f'<span style="color:#888"> / </span>'
                    f'<span style="color:{cr}">{r.right.status}</span>'
                )
            self.val_crossarm.setText("<br>".join(rows))
        else:
            self.val_crossarm.setText("--（导入模型无法解析横担）")

    @staticmethod
    def _err_color(err):
        a = abs(err)
        if a < 0.5:
            return "#2E7D32"
        if a < 1.5:
            return "#E65100"
        return "#C62828"

    def append_log(self, text):
        self.log.append(text)
        bar = self.log.verticalScrollBar()
        bar.setValue(bar.maximum())
