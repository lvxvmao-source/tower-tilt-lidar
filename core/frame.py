# -*- coding: utf-8 -*-
"""
全系统坐标约定（唯一权威定义，其他模块一律以此为准）

    +x  = 南
    +y  = 东
    +z  = 上（铅垂向上）

这是右手系：从 +z 俯视，x 转向 y 为逆时针。
验算：南 × 东 = 上 ✓

———————————— 罗盘对照 ————————————

           北 (−x)
             │
   西 (−y) ──┼── 东 (+y)
             │
           南 (+x)

———————————— 站位方位角 b ————————————

`LidarConfig.station_bearings_deg` 里的 b 是**数学角**，从 +x 起逆时针：

    b = 0°    → +x → 南          b = 180° → −x → 北
    b = 90°   → +y → 东          b = 270° → −y → 西

测量学方位角 A（从北起、顺时针）与它的换算：

    A = 90° − b        b = 90° − A

接真实航线数据时务必换算，否则布站会整体转 90° 且方向相反，而且不会报错。

———————————— 塔基四角 ————————————

四个塔脚在 (±c, ±c)，c = 底宽/2 + 塔腿外伸：

    NW 西北 = (−c, −c)      NE 东北 = (−c, +c)
    SW 西南 = (+c, −c)      SE 东南 = (+c, +c)
"""
from typing import Tuple

AXIS_NAMES = {"x": "南", "y": "东", "z": "上"}

# 罗盘名 → 站位方位角（数学角，从 +x 南 逆时针）
COMPASS_TO_BEARING = {
    "南": 0.0,
    "东南": 45.0,
    "东": 90.0,
    "东北": 135.0,
    "北": 180.0,
    "西北": 225.0,
    "西": 270.0,
    "西南": 315.0,
}

# 塔基四角：名称 → (x 符号, y 符号)，x=南、y=东
CORNER_SIGNS = {
    "NW": (-1, -1),   # 西北
    "NE": (-1, +1),   # 东北
    "SE": (+1, +1),   # 东南
    "SW": (+1, -1),   # 西南
}

CORNER_CN = {"NW": "西北", "NE": "东北", "SE": "东南", "SW": "西南"}


def bearing_to_compass(b_deg: float, tolerance: float = 22.5) -> str:
    """数学角 b（从 +x 南 逆时针）→ 最近的四正/四隅方位名

    偏离最近方位超过 tolerance 度时返回带角度的描述，避免误报。
    """
    b = float(b_deg) % 360.0
    best_name, best_diff = "南", 999.0
    for name, ang in COMPASS_TO_BEARING.items():
        d = abs((b - ang + 180.0) % 360.0 - 180.0)
        if d < best_diff:
            best_name, best_diff = name, d
    return best_name if best_diff <= tolerance else f"{b:.0f}°"


def bearing_of_vector(dx: float, dy: float) -> float:
    """由水平位移分量求数学角 b（度，0~360）"""
    import math
    return math.degrees(math.atan2(dy, dx)) % 360.0


def compass_of_vector(dx: float, dy: float) -> str:
    """由水平位移分量直接给出方位名（例：塔顶往东南偏 → '东南'）"""
    return bearing_to_compass(bearing_of_vector(dx, dy))


def corner_xy(name: str, c: float) -> Tuple[float, float]:
    """塔脚名 → 平面坐标。c 为塔脚到塔位中心的水平距离。"""
    sx, sy = CORNER_SIGNS[name]
    return sx * c, sy * c


def corner_settlement_from_tilt(c: float, ax: float, ay: float) -> dict:
    """由基础倾角的反切值 (tan_x, tan_y) 求四角沉降量（下沉为正）

    基础平面：δ(x, y) = ax·x + ay·y
    """
    return {name: c * (ax * sx + ay * sy)
            for name, (sx, sy) in CORNER_SIGNS.items()}


def tilt_from_corner_settlement(c: float, d: dict) -> Tuple[float, float]:
    """由四角沉降量反求 (tan_x, tan_y)：最小二乘平面拟合"""
    ax = ((d["SE"] + d["SW"]) - (d["NE"] + d["NW"])) / (4.0 * c)
    ay = ((d["SE"] + d["NE"]) - (d["SW"] + d["NW"])) / (4.0 * c)
    return ax, ay
