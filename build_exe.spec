# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：单文件可执行程序

用法：
    python -m PyInstaller build_exe.spec --noconfirm --clean

产物：
    dist/杆塔倾斜测量系统.exe
"""
from PyInstaller.utils.hooks import (collect_data_files, collect_dynamic_libs,
                                     collect_submodules)

# ---------------- 随包携带的资源 ----------------
datas = [("config/radar_presets.json", "config")]
datas += collect_data_files("open3d")

# open3d 的原生库（.dll / .pyd 依赖）
binaries = collect_dynamic_libs("open3d")

# ---------------- 隐式导入 ----------------
hiddenimports = [
    "vtkmodules.all",
    "vtkmodules.qt.QVTKRenderWindowInteractor",
    "pyvistaqt",
    "open3d",
    "open3d.cpu",
]
hiddenimports += collect_submodules("pyvistaqt")

# ---------------- 排除用不到的大件 ----------------
excludes = [
    # ── 深度学习 / 科学计算：仅被 open3d 的 ML 模块和 examples 引用 ──
    "open3d.ml", "open3d._ml3d",
    "torch", "torchvision", "torchaudio", "tensorflow",
    "sklearn", "jax", "onnx", "onnxruntime",
    "scipy", "pandas",
    # PIL 不能排除：matplotlib/colors.py 在导入时就用它
    # ── mypy：mypyc 编译，其 <hash>__mypyc 运行时打不进包；
    #    pyvista.typing.mypy_plugin 一旦发现 mypy 存在就会 import，导致启动崩溃 ──
    "mypy", "mypy_extensions", "mypyc",
    # ── 文档与测试工具，运行时不加载 ──
    "pytest", "docutils", "sphinx",
    # ── 未使用的 GUI / 可视化后端 ──
    "IPython", "tkinter", "PyQt5", "PyQt6", "PySide2",
]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="杆塔倾斜测量系统",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,         # 无黑窗，双击即用（调试时可临时改回 True）
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
