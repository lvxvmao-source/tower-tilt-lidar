# 杆塔倾斜激光测量系统

基于无人机激光点云的输电杆塔倾斜测量与变形分析系统。

作者：**白浪淘沙**

---

## 这是什么

一套完整的杆塔倾斜测量工具链：从参数化塔模型生成、多站激光扫描仿真，到点云解算、规范判定、弯曲定位、两期变形对比，全部封装为可交互的桌面程序。

核心思路是**把杆塔的几何先验直接写进优化目标**，而不是只用它来提取中间量：

> 塔身是一个线性收窄的正方形棱柱，即 $\max(|dx|,|dy|) = h_0 - k z$

多数同类方法把这条性质用于"求每层截面中心"，再把中心点拟合直线；本系统把它作为约束，让全部点云同时参与解算，观测数从二十余个提升到数千个量级。

---

## 主要方法

| 模块 | 方法 |
|---|---|
| **点云仿真** | 多站机械式激光雷达模型，射线矢量化生成 + Open3D 射线求交 |
| **横担剔除** | PCA 定方位 + 逐高度自适应包络 + Theil–Sen 稳健回归 |
| **中轴提取** | 分层窗口 + 四象限加权 + 对称分位中点 |
| **全局精化** | 方形棱柱约束 + Huber 抗差 + 分面分层均衡 + 二次弯曲项 + 协方差传播 |
| **弯曲定位** | 夹紧三次 B 样条重建塔轴 + 曲率正则 + 斜率剖面变点检测 |
| **两期对比** | 两期各自重建塔轴，轴对轴比较（免配准） |
| **横担健康** | 沿轴高度坐标 + 臂长反演 + 缺失判定 |
| **数据质量** | 残差 / 棱柱半径 比值判别，拦截非杆塔点云 |

精度（仿真实测，16 线雷达、0.2° 角分辨率、4 方位 × 5 层）：

- 倾斜率 α 平均绝对误差 **0.19‰**，最大 0.43‰
- 基础倾斜率 δ 平均 0.46‰，最大 1.33‰

---

## 目录结构

```
tower_tilt_project/
├── main.py                  程序入口
├── build_exe.spec           PyInstaller 打包配置
├── config/                  雷达预设
├── core/                    算法核心
│   ├── frame.py             坐标约定（+x=南、+y=东、+z=上）
│   ├── tower_model.py       参数化塔模型与真值
│   ├── lidar_scanner.py     多站扫描仿真
│   ├── tilt_solver.py       预处理 / 精化 / 指标
│   ├── deform_analysis.py   样条轴线 / 弯曲定位 / 两期对比
│   ├── crossarm_analysis.py 横担几何健康
│   ├── standards.py         规范限值与判定
│   └── point_io.py          实测点云导入与规范化
├── ui/                      PySide6 界面
├── experiments/             自检与消融实验脚本
└── 使用说明.txt             操作手册
```

---

## 环境要求

- Python 3.10+
- 依赖：`PySide6`、`open3d`、`pyvista`、`pyvistaqt`、`numpy`

```bash
pip install PySide6 open3d pyvista pyvistaqt numpy
```

---

## 运行

```bash
python main.py
```

界面左侧四个标签页：**塔模型**（参数/数据源）、**雷达**（扫描站位）、**算法**（解算参数）、**变形对比**（两期分析）。

最快体验：

1. 打开后 3D 视图已有一座 30 m 默认塔；
2. 「塔模型」页把 θx 调 1.0、θy 调 0.5；
3. 点工具栏「运行仿真」；
4. 右侧看倾斜率、规范判定、弯曲定位。

---

## 自检

```bash
python experiments/selftest_frame.py       # 坐标约定
python experiments/selftest_standard.py    # 标准指标与判定
python experiments/selftest_core.py        # 进阶模型
python experiments/selftest_crossarm.py    # 横担健康
python experiments/selftest_cloud_import.py # 实测点云往返
python experiments/ablation_refine.py      # 消融实验
```

---

## 打包为可执行文件

```bash
pip install pyinstaller
python -m PyInstaller build_exe.spec --noconfirm
# 产物：dist/杆塔倾斜测量系统.exe（约 640 MB，单文件）
```

---

## 使用说明

见根目录 **`使用说明.txt`**：含界面图示、最快上手步骤、两期对比用法、系统要求。

---

## 坐标约定

全系统统一（`core/frame.py`）：

```
+x = 南      +y = 东      +z = 上
```

右手系，从 +z 俯视 x→y 为逆时针。

站位方位角是**数学角**（自 +x 逆时针），与测量学方位角 $A$ 的换算为 $A = 90° - b$。

---

## 已知局限

- 折角幅度因样条平滑被低估约 50%，位置比幅度可靠
- 弱折角（<5‰）与轴心噪声同量级，检不出
- 横担端部下垂受扫描几何限制，不可测
- 两期对比要求两期数据处于同一坐标参考

---

## 许可

个人学习与研究用途。如需商用请联系作者。
