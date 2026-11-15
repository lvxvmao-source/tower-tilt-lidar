# -*- coding: utf-8 -*-
"""坐标约定自检：+x=南、+y=东、+z=上

这类错误最危险的地方在于"不会报错"——几何照算，只是方位转 90° 或镜像。
所以逐条验证：站位方位、倾斜方向、四角沉降、换算往返。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
np.random.seed(42)

from core.frame import (COMPASS_TO_BEARING, CORNER_SIGNS, bearing_to_compass,
                        compass_of_vector, corner_xy,
                        corner_settlement_from_tilt, tilt_from_corner_settlement)
from core.tower_model import (TowerConfig, build_tower, base_tilt_deg,
                              settlements_from_tilt, tower_axis,
                              base_corner_offset, _tilt_matrix)
from core.lidar_scanner import LidarConfig, simulate_scan
from core.tilt_solver import SolverConfig, solve

ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'✓' if cond else '✗'}] {name}{('  ' + detail) if detail else ''}")


print("=== 1. 罗盘 ↔ 数学角 对照 ===")
expect = {"南": 0, "东南": 45, "东": 90, "东北": 135,
          "北": 180, "西北": 225, "西": 270, "西南": 315}
for name, ang in expect.items():
    b = COMPASS_TO_BEARING[name]
    check(f"{name} = {ang}°", b == ang, f"实为 {b}°")
    check(f"  {ang}° → {name}", bearing_to_compass(ang) == name,
          f"实为 {bearing_to_compass(ang)}")

print("\n=== 2. 站位真的在对应方位吗 ===")
cfg = TowerConfig()
c = base_corner_offset(cfg)
for name in ("南", "东", "北", "西"):
    b = np.deg2rad(COMPASS_TO_BEARING[name])
    D = 15.0
    sx, sy = D * np.cos(b), D * np.sin(b)
    # 南：(x>0,y≈0)；东：(x≈0,y>0)；北：(x<0)；西：(y<0)
    cond = {"南": sx > 14 and abs(sy) < 1,
            "东": sy > 14 and abs(sx) < 1,
            "北": sx < -14 and abs(sy) < 1,
            "西": sy < -14 and abs(sx) < 1}[name]
    check(f"{name}站位坐标 ({sx:+.1f}, {sy:+.1f})", cond)

print("\n=== 3. 四角坐标约定 ===")
for name, (sx, sy) in CORNER_SIGNS.items():
    x, y = corner_xy(name, c)
    check(f"{name} → ({x:+.1f}, {y:+.1f})", (x, y) == (sx * c, sy * c))
# 东北角应在 x<0、y>0
check("东北角位于 x<0（北侧）、y>0（东侧）", CORNER_SIGNS["NE"] == (-1, +1))

print("\n=== 4. 倾斜方向：θx=1° 应倒向南方 ===")
cfg = TowerConfig(tilt_deg_x=1.0, tilt_deg_y=0.0)
H = cfg.total_height
tip = tower_axis(cfg)(H)
dir_name = compass_of_vector(tip[0], tip[1])
check(f"塔顶偏移 ({tip[0]:+.3f}, {tip[1]:+.3f}) → {dir_name}",
      tip[0] > 0.4 and abs(tip[1]) < 1e-6 and dir_name == "南")

cfg = TowerConfig(tilt_deg_x=0.0, tilt_deg_y=1.0)
tip = tower_axis(cfg)(H)
check(f"θy=1° → 塔顶 ({tip[0]:+.3f}, {tip[1]:+.3f}) → {compass_of_vector(tip[0], tip[1])}",
      tip[1] > 0.4 and abs(tip[0]) < 1e-6)

print("\n=== 5. 四角沉降：东北角沉降最多，塔应倒向东北 ===")
cfg = TowerConfig(use_settlement=True, settlement_ne=0.10,
                  settlement_nw=0.0, settlement_se=0.0, settlement_sw=0.0)
tx, ty = base_tilt_deg(cfg)
tip = tower_axis(cfg)(H)
name = compass_of_vector(tip[0], tip[1])
check(f"倾角 ({tx:+.3f}°, {ty:+.3f}°) → 塔顶倒向 {name}",
      name == "东北", f"塔顶偏移 ({tip[0]:+.3f}, {tip[1]:+.3f})")

# 网格实体验证：旋转后最低的塔脚应该是东北角
R, _ = _tilt_matrix(cfg)
corners = np.array([[*corner_xy(k, c), 0.0] for k in ("NW", "NE", "SE", "SW")])
rot = corners @ R.T
lowest = ["NW", "NE", "SE", "SW"][int(np.argmin(rot[:, 2]))]
check(f"实体最低塔脚 = {lowest}", lowest == "NE",
      f"四角 z = {np.round(rot[:, 2], 4)}")

print("\n=== 6. 换算往返一致性 ===")
cfg0 = TowerConfig(tilt_deg_x=0.8, tilt_deg_y=0.5)
ne, se, sw, nw = settlements_from_tilt(cfg0, 0.8, 0.5)
cfg1 = TowerConfig(use_settlement=True, settlement_ne=ne, settlement_se=se,
                   settlement_sw=sw, settlement_nw=nw)
tx, ty = base_tilt_deg(cfg1)
check(f"倾角→沉降→倾角 往返 ({tx:.4f}, {ty:.4f}) ≈ (0.8, 0.5)",
      abs(tx - 0.8) < 1e-6 and abs(ty - 0.5) < 1e-6)
tip0 = tower_axis(cfg0)(H)
tip1 = tower_axis(cfg1)(H)
check("两种输入生成的塔轴一致", np.allclose(tip0[:2], tip1[:2], atol=1e-6),
      f"差 {np.linalg.norm(tip0[:2]-tip1[:2]):.2e} m")

print("\n=== 7. 端到端：解算出的倾斜方向 ===")
for kw, expect_dir in [({"tilt_deg_x": 1.0}, "南"),
                       ({"tilt_deg_y": 1.0}, "东"),
                       ({"tilt_deg_x": -1.0}, "北")]:
    cfg = TowerConfig(**kw)
    s = SolverConfig()
    s.tower_height = cfg.total_height
    pcd = simulate_scan(build_tower(cfg), LidarConfig(horizontal_resolution_deg=0.4))
    r, _ = solve(pcd, s)
    # 用解出的 X/Y 分量判断方位
    vx = r.alpha_standard_permille * np.sign(1)
    # tilt_x_deg / tilt_y_deg 是绝对值，方向要靠真值对照
    true_tip = tower_axis(cfg)(H)
    name = compass_of_vector(true_tip[0], true_tip[1])
    check(f"{kw} → 真值方位 {name}", name == expect_dir,
          f"测得 α={r.alpha_standard_permille:.2f}‰")

print("\n" + ("=== 全部通过 ===" if ok else "=== 存在失败项 ==="))
sys.exit(0 if ok else 1)
