# -*- coding: utf-8 -*-
"""
输电杆塔倾斜 / 基础沉降的行业标准判定

引用的现行标准：
  DL/T 741-2019   《架空输电线路运行规程》
                  直线杆塔倾斜率：全高 ≥50m ≤0.5%(5‰)；全高 <50m ≤1%(10‰)
  Q/GDW 173-2008  《架空输电线路杆塔状态评价标准导则》
                  杆塔倾斜四級状态：正常 / 注意 / 异常 / 严重
  DL/T 5219       《架空输电线路基础设计规程》
                  表 3.1.15 地基变形允许值（基础倾斜率 δ = 两端点沉降差/距离）
  发改委第 41 号令 《电力重大事故隐患判定标准及治理监督管理规定》(2026-07-01 施行)
                  第五条(九)：±800kV/1000kV 全高≥50m >10‰、<50m >15‰；
                              ±660kV/750kV 全高≥50m >15‰、<50m >20‰

术语：
  α 杆塔倾斜率 = 塔顶水平偏移 S / 地面以上塔高 H（≈tanθ），单位 ‰
  δ 基础倾斜率 = 基础倾斜方向两端点沉降差 / 两点距离，无量纲
  两者在刚性塔体下数值接近，但检查对象不同（α 查塔身，δ 查地基）。
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ---------------------------------------------------------------- 电压等级

VOLTAGE_OPTIONS = ("110kV", "220kV", "330kV", "500kV", "±660kV", "750kV", "±800kV", "1000kV")


def parse_voltage(label: str) -> float:
    """把 '±800kV' / '750kV' 解析成数值 kV"""
    s = str(label).replace("±", "").replace("kV", "").replace("KV", "").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


# ---------------------------------------------------------------- 限值查询

def operation_limit_permille(tower_height: float) -> float:
    """DL/T 741-2019 直线杆塔倾斜率运行限值（‰）"""
    return 5.0 if tower_height >= 50.0 else 10.0


def foundation_tilt_limit(tower_height: float) -> float:
    """DL/T 5219 表 3.1.15 地基变形允许值（基础倾斜率，无量纲）"""
    if tower_height <= 50.0:
        return 0.006
    if tower_height <= 100.0:
        return 0.005
    if tower_height <= 150.0:
        return 0.004
    if tower_height <= 200.0:
        return 0.003
    if tower_height <= 250.0:
        return 0.002
    return 0.0015


def major_hazard_limit_permille(tower_height: float, voltage_kv: float) -> Optional[float]:
    """发改委 41 号令 第五条(九) 重大隐患红线（‰）。

    该条款只覆盖 ±660kV 及以上、750kV 及以上线路，500kV 及以下返回 None。
    """
    if voltage_kv >= 800.0:          # 直流 ±800kV / 交流 1000kV
        return 10.0 if tower_height >= 50.0 else 15.0
    if voltage_kv >= 660.0:          # ±660kV 直流 / 750kV 交流
        return 15.0 if tower_height >= 50.0 else 20.0
    return None


def state_grade(alpha_permille: float, tower_height: float) -> "tuple[str, int]":
    """Q/GDW 173-2008 杆塔倾斜状态等级，返回 (等级名, 序号 1~4，4=正常)"""
    # NaN 与任何数比较都为 False，会一路掉到"严重"分支 —— 必须先拦住
    if alpha_permille is None or not np.isfinite(alpha_permille):
        return "未知", 4
    if tower_height >= 50.0:
        if alpha_permille < 5.0:
            return "正常", 4
        if alpha_permille < 10.0:
            return "注意", 3
        if alpha_permille < 15.0:
            return "异常", 2
        return "严重", 1
    if alpha_permille < 10.0:
        return "正常", 4
    if alpha_permille < 15.0:
        return "注意", 3
    if alpha_permille < 20.0:
        return "异常", 2
    return "严重", 1


GRADE_COLORS = {
    4: ("#E8F5E9", "#2E7D32"),   # 正常 绿
    3: ("#FFF8E1", "#F9A825"),   # 注意 黄
    2: ("#FFF3E0", "#E65100"),   # 异常 橙
    1: ("#FFEBEE", "#C62828"),   # 严重 红
}

ALERT_NAMES = {4: "蓝色", 3: "黄色", 2: "橙色", 1: "红色"}


# ---------------------------------------------------------------- 综合评估

@dataclass
class StandardEval:
    alpha_permille: float
    foundation_permille: float
    tower_height: float
    voltage_kv: float

    grade: str = "正常"
    grade_index: int = 4
    alert_color: str = "蓝色"

    operation_limit: float = 10.0
    operation_ok: bool = True

    foundation_limit: float = 6.0        # ‰（0.006 换算成 ‰）
    foundation_ok: bool = True

    major_hazard_limit: Optional[float] = None
    major_hazard: bool = False

    def summary(self) -> str:
        if self.major_hazard:
            return f"重大隐患（超 {self.major_hazard_limit:.0f}‰ 红线）"
        if not self.operation_ok:
            return f"超运行限值 {self.operation_limit:.0f}‰"
        if not self.foundation_ok:
            return "基础倾斜超设计允许值"
        return "符合标准"


def evaluate(alpha_permille: float,
             foundation_permille: float,
             tower_height: float,
             voltage_label: str = "110kV") -> StandardEval:
    """按现行标准综合判定"""
    v = parse_voltage(voltage_label)
    grade, idx = state_grade(alpha_permille, tower_height)
    mh_limit = major_hazard_limit_permille(tower_height, v)
    fnd_limit = foundation_tilt_limit(tower_height)

    ev = StandardEval(
        alpha_permille=float(alpha_permille),
        foundation_permille=float(foundation_permille),
        tower_height=float(tower_height),
        voltage_kv=v,
        grade=grade,
        grade_index=idx,
        alert_color=ALERT_NAMES[idx],
        operation_limit=operation_limit_permille(tower_height),
        operation_ok=alpha_permille <= operation_limit_permille(tower_height),
        foundation_limit=fnd_limit * 1000.0,
        foundation_ok=foundation_permille <= fnd_limit * 1000.0,
        major_hazard_limit=mh_limit,
        major_hazard=bool(mh_limit is not None and alpha_permille > mh_limit),
    )
    return ev
