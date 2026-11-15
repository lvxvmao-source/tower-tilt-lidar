# -*- coding: utf-8 -*-
"""
参数化角钢塔生成器（实验脚本入口）

原先是与 core/tower_model.py 并列的第二份实现，两边参数名不同、倾斜方向的
定义也不一致。现在统一为对 core.tower_model 的薄封装：
TowerParams 就是 TowerConfig，build_tower 就是同一份实现，
因此论文实验里的"真值"与界面里的"真值"含义完全一致。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tower_model import (      # noqa: F401
    TowerConfig,
    build_tower,
    load_tower_from_ply,
    base_tilt_deg,
    settlements_from_tilt,
    tower_axis,
    true_tilt,
)

# 兼容旧名字
TowerParams = TowerConfig


if __name__ == "__main__":
    import numpy as np
    import open3d as o3d

    params = TowerParams(tilt_deg_x=0.5, tilt_deg_y=0.2)
    tower = build_tower(params)
    print(f"塔网格顶点数：{len(tower.vertices)}")
    print(f"塔网格三角面数：{len(tower.triangles)}")
    print(f"塔倾斜真值：X方向 {params.tilt_deg_x}°, Y方向 {params.tilt_deg_y}°")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "demo_tower.ply")
    o3d.io.write_triangle_mesh(out_path, tower)
    print(f"已保存到：{out_path}")
