# -*- coding: utf-8 -*-
"""
倾斜解算器：预处理→切片→拟合，返回结果和中间几何体用于3D显示

链路：
  预处理(体素+SOR) → 高度截断 → PCA定横担方位 + 逐高度自适应包络剔除
      → 逐窗口对称中心提取中轴点(带置信度) → RANSAC + 置信度加权Huber选权迭代直线拟合
      → 倾斜解算
"""
import numpy as np
import open3d as o3d
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class SolverConfig:
    """算法参数"""
    voxel_size: float = 0.05
    z_min: float = 4.0
    z_max: float = 27.0
    window_h: float = 2.0
    step: float = 1.0
    ransac_thr: Optional[float] = None  # None=自适应
    max_iter: int = 3000
    tower_height: float = 30.0          # 用于把倾斜角折算成标准口径的塔顶偏移
    # 全局精化（核心创新点）：是否启用方形棱柱几何约束的全点云抗差拟合
    use_global_refine: bool = True
    # 分面分层均衡加权：消除"迎机面密、背机面稀"造成的系统性偏斜
    use_balance_weight: bool = True
    balance_power: float = 0.5          # 0=不均衡，1=完全均衡，0.5=折中
    # 样条轴线重建（基元拟合，用于弯曲定位）
    use_spline: bool = True
    knot_spacing: float = 2.0           # 样条节点间距 m，越小越能分辨局部弯曲


@dataclass
class TiltResult:
    tilt_x_deg: float
    tilt_y_deg: float
    total_tilt_deg: float
    tilt_rate_permille: float
    confidence: float
    num_axis_points: int
    num_inliers: int
    # ---- 标准口径指标（DL/T 741 / DL/T 5219）----
    tip_offset_mm: float = 0.0          # 塔顶水平偏移 S（相对塔基中心）
    alpha_standard_permille: float = 0.0  # 倾斜率 α = S/H×1000
    foundation_tilt_permille: float = 0.0  # 基础倾斜率 δ（塔基处轴斜率）
    axis_curve_permille: float = 0.0    # 弯曲程度：α_standard − δ
    # ---- 全局精化输出 ----
    alpha_sigma_permille: float = 0.0   # α 的 1σ 不确定度
    refine_used: bool = False           # 是否启用了全局精化
    refine_points: int = 0              # 参与精化的点数
    # ---- 弯曲定位（样条轴线 + 变点检测）----
    bend_found: bool = False
    bend_height: float = 0.0            # 折角高度 m
    bend_delta_permille: float = 0.0    # 折点上下斜率增量 ‰
    bend_grade: str = "无"
    spline_rms_mm: float = 0.0
    # ---- 数据质量校验 ----
    # 残差与塔身半宽之比。真实塔的塔身点都贴在棱柱面上，比值约 0.03；
    # 不是杆塔的点云（噪声、植被、错位）比值 0.3 以上。
    quality: str = "未知"
    quality_ratio: float = 0.0
    # 弯曲矢高（轴线相对两端连线的最大偏离）与矢跨比
    deflection_mm: float = 0.0
    deflection_ratio_permille: float = 0.0


def preprocess(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.05) -> o3d.geometry.PointCloud:
    pcd_ds = pcd.voxel_down_sample(voxel_size)
    pcd_ds, _ = pcd_ds.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    return pcd_ds


def crop_height(pcd: o3d.geometry.PointCloud,
                z_min: float = 4.0, z_max: float = 27.0) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points)
    mask = (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(pts[mask])
    return p


# ---------------------------------------------------------------- 基础工具

_SYM_PAIRS = ((1, 99), (3, 97), (5, 95), (10, 90), (20, 80))


def _symmetric_center(v: np.ndarray) -> float:
    """对称分布的中心估计：多组对称分位的中点，再取中位数。

    塔身是正方形四壁结构，沿任一水平轴方向的截面点云天然关于塔中轴对称，
    因此对称分位的中点必然落在中轴上，且与"哪一面点更密"完全无关。
    这是四象限中值做不到的：某一面点特别密时，象限中值会被那一面拉走。
    """
    v = np.asarray(v, dtype=float)
    if len(v) < 16:
        return float(np.median(v)) if len(v) else 0.0
    mids = [0.5 * (np.percentile(v, lo) + np.percentile(v, hi)) for lo, hi in _SYM_PAIRS]
    return float(np.median(mids))


def _robust_halfwidth(v: np.ndarray) -> float:
    """对称分布的半宽：对称分位间距的一半，取中位数抗单组异常"""
    v = np.asarray(v, dtype=float)
    if len(v) < 16:
        return float(np.ptp(v) / 2) if len(v) else 0.0
    halves = [0.5 * (np.percentile(v, hi) - np.percentile(v, lo)) for lo, hi in _SYM_PAIRS]
    return float(np.median(halves))


def _fit_line_1d(z: np.ndarray, y: np.ndarray):
    """对 (z, y) 做一次**稳健**一阶拟合（Theil-Sen + 残差剔除）。

    普通 np.polyfit 对离群窗口毫无抵抗力：塔顶或横担层的窗口一旦退化
    （只剩半层点、只有单侧面），就会把整条包络线拉偏，进而误删塔身。
    Theil-Sen 取两两斜率的中位数，可以容忍接近一半的窗口异常。
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(z) < 3:
        const = float(np.median(y)) if len(y) else 0.0
        return lambda zz: np.full(np.shape(np.asarray(zz, dtype=float)), const, dtype=float)

    i, j = np.triu_indices(len(z), k=1)
    dz = z[j] - z[i]
    ok = np.abs(dz) > 1e-9
    if not ok.any():
        const = float(np.median(y))
        return lambda zz: np.full(np.shape(np.asarray(zz, dtype=float)), const, dtype=float)

    slope = float(np.median((y[j][ok] - y[i][ok]) / dz[ok]))
    intercept = float(np.median(y - slope * z))

    # 用残差再剔一轮明显跑偏的窗口
    resid = y - (slope * z + intercept)
    mad = float(np.median(np.abs(resid - np.median(resid))))
    limit = max(0.25, 3.0 * 1.4826 * mad)
    inl = np.abs(resid - np.median(resid)) <= limit
    if inl.sum() >= 3 and inl.sum() < len(z):
        zi, yi = z[inl], y[inl]
        i, j = np.triu_indices(len(zi), k=1)
        dz = zi[j] - zi[i]
        ok = np.abs(dz) > 1e-9
        if ok.any():
            slope = float(np.median((yi[j][ok] - yi[i][ok]) / dz[ok]))
            intercept = float(np.median(yi - slope * zi))

    return lambda zz: slope * np.asarray(zz, dtype=float) + intercept


# ---------------------------------------------------------------- 横担剔除

def remove_crossarms_pca(pcd: o3d.geometry.PointCloud,
                         tol: float = 0.6,
                         window_h: float = 2.0,
                         step: float = 1.0) -> o3d.geometry.PointCloud:
    """PCA 定横担方位 + 逐高度自适应包络剔除横担。

    1) 对 XY 做 PCA，最大主成分即横担伸出方向 m，其正交方向 n 不含横担；
    2) 逐高度窗口：用 n 方向的分位跨度估计该高度处的塔身半宽（横担不污染 n 方向），
       用 m 方向的对称分位中点估计该高度的中轴位置（对单侧面点云密度不均免疫）；
    3) 对 中轴位置(z)、塔身半宽(z) 各做一次线性拟合（塔线性收窄），
       逐点判断 |m 投影 - 中轴(z)| 是否超出自适应包络。
    """
    pts = np.asarray(pcd.points)
    if len(pts) < 100:
        return pcd

    xy = pts[:, :2]
    centered = xy - xy.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(np.cov(centered.T))
    # 截断区间里没有横担时，正方形截面的 XY 协方差两个特征值基本相等，
    # 此时"主方向"只是数值噪声，硬套包络反而会切掉塔身四角。
    if eigvals[-1] < 1.15 * max(eigvals[0], 1e-12):
        return pcd
    m_dir = eigvecs[:, -1]                      # 横担伸出方向
    if m_dir[0] < 0:
        m_dir = -m_dir
    n_dir = np.array([-m_dir[1], m_dir[0]])     # 面内正交方向（不含横担）

    zs = pts[:, 2]
    proj_m = xy @ m_dir
    proj_n = xy @ n_dir

    z0, z1 = float(zs.min()), float(zs.max())
    z_centers, c_list, h_list = [], [], []
    z = z0
    while z + window_h <= z1:          # 窗口必须完整落在数据高度范围内
        mask = (zs >= z) & (zs < z + window_h)
        if int(mask.sum()) >= 30:
            z_centers.append(z + window_h / 2.0)
            c_list.append(_symmetric_center(proj_m[mask]))
            h_list.append(_robust_halfwidth(proj_n[mask]))
        z += step

    if len(z_centers) < 2:
        return pcd

    c_fit = _fit_line_1d(z_centers, c_list)
    h_fit = _fit_line_1d(z_centers, h_list)

    half_at = np.maximum(h_fit(zs), 0.2)
    center_at = c_fit(zs)
    # 1.08 补偿窗口内锥度，tol 为允许保留的横担根部余量
    keep = np.abs(proj_m - center_at) <= (half_at * 1.08 + tol)

    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(pts[keep])
    return p


# ---------------------------------------------------------------- 中轴提取

def extract_axis_weighted(pcd: o3d.geometry.PointCloud,
                          window_h: float, step: float) -> Tuple[np.ndarray, np.ndarray]:
    """沿高度滑窗，每个窗口提取一个中轴点及其置信度。

    中心用对称分位中点估计（对迎机面密、背机面稀不敏感）；
    置信度由四象限覆盖度、窗口点数、四向密度均衡度三部分合成。
    """
    pts = np.asarray(pcd.points)
    if len(pts) < 20:
        return np.empty((0, 3)), np.empty((0,))

    zs = pts[:, 2]
    z_min, z_max = float(zs.min()), float(zs.max())

    # 只取完整落在数据高度范围内的窗口：塔顶那半层数据只被单个站位扫到，
    # 单侧面会让对称中心彻底失效，宁可不要这 1~2 个点。
    windows = []
    z = z_min
    while z + window_h <= z_max + 1e-9:
        windows.append((z, z + window_h))
        z += step
    if len(windows) < 3:                       # 数据跨度不足，退化为单窗口
        windows = [(z_min, z_max)]

    axis_pts, confs = [], []
    for z0w, z1w in windows:
        mask = (zs >= z0w) & (zs < z1w)
        win = pts[mask]
        if len(win) < 10:
            continue

        cx = _symmetric_center(win[:, 0])
        cy = _symmetric_center(win[:, 1])

        # 四象限覆盖度：四面都被扫到时每个象限都有点
        dx = win[:, 0] - cx
        dy = win[:, 1] - cy
        quads = [(dx >= 0) & (dy >= 0), (dx < 0) & (dy >= 0),
                 (dx < 0) & (dy < 0), (dx >= 0) & (dy < 0)]
        counts = np.array([int(q.sum()) for q in quads], dtype=float)
        conf_quad = float((counts >= 3).sum()) / 4.0
        conf_num = min(len(win) / 60.0, 1.0)
        # 密度均衡度：某一面明显稀说明存在遮挡，该层应降权
        nz = counts[counts > 0]
        if len(nz) >= 2:
            balance = 1.0 - min(float(nz.max() - nz.min()) / float(nz.sum() + 1e-6), 1.0)
        else:
            balance = 0.0

        axis_pts.append([cx, cy, float(np.median(win[:, 2]))])
        confs.append(0.45 * conf_quad + 0.30 * conf_num + 0.25 * balance)

    if not axis_pts:
        return np.empty((0, 3)), np.empty((0,))
    return np.array(axis_pts), np.array(confs)


def _weighted_pca(pts, center, w):
    """加权PCA求第一主成分方向，保证 z 分量为正"""
    c = pts - center
    cov = (c.T * w) @ c / w.sum()
    _, evec = np.linalg.eigh(cov)
    v = evec[:, -1]
    if v[2] < 0:
        v = -v
    return v / np.linalg.norm(v)


def _auto_threshold(axis_pts: np.ndarray) -> float:
    """由轴点到最小二乘直线的残差尺度自动定 RANSAC 阈值（MAD 估计）"""
    if len(axis_pts) < 5:
        return 0.25
    c = axis_pts.mean(axis=0)
    q = axis_pts - c
    try:
        _, _, vt = np.linalg.svd(q, full_matrices=False)
        v = vt[0]
    except np.linalg.LinAlgError:
        return 0.25
    perp = q - np.outer(q @ v, v)
    d = np.linalg.norm(perp, axis=1)
    mad = float(np.median(np.abs(d - np.median(d))))
    return float(np.clip(2.5 * 1.4826 * mad, 0.12, 0.50))


def fit_line_robust(axis_pts, weights, thr=None, max_iter=3000, distance_threshold=None):
    """RANSAC 粗选内点 + 置信度加权 Huber 选权迭代精拟合

    thr 与 distance_threshold 是同一个参数的两种叫法（兼容早期实验脚本）。
    """
    if thr is None:
        thr = 0.25 if distance_threshold is None else float(distance_threshold)
    n = len(axis_pts)
    if n < 5:
        center = axis_pts.mean(axis=0) if n else np.zeros(3)
        return np.array([0.0, 0.0, 1.0]), center, n, 0.0

    rng = np.random.default_rng(42)
    best_mask = np.zeros(n, dtype=bool)
    best_cnt = 0
    for _ in range(max_iter):
        idx = rng.choice(n, 2, replace=False)
        p1, p2 = axis_pts[idx[0]], axis_pts[idx[1]]
        v = p2 - p1
        nv = np.linalg.norm(v)
        if nv < 1e-6:
            continue
        v = v / nv
        d = np.linalg.norm(np.cross(axis_pts - p1, v), axis=1)
        inl = d < thr
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_mask = inl

    if best_cnt < 3:                      # RANSAC 没找到足够内点，全部当内点处理
        best_mask = np.ones(n, dtype=bool)
        best_cnt = n

    inl_pts = axis_pts[best_mask]
    inl_w = np.maximum(np.asarray(weights, dtype=float)[best_mask], 0.01)
    center = np.average(inl_pts, axis=0, weights=inl_w)
    v = _weighted_pca(inl_pts, center, inl_w)

    w_iter = inl_w.copy()
    for _ in range(5):
        diff = inl_pts - center
        vh = np.array([v[0], v[1], 0.0])
        nh = np.linalg.norm(vh)
        if nh < 1e-6:
            break
        vh = vh / nh
        perp = np.array([-vh[1], vh[0], 0.0])
        dists = np.abs(diff @ perp)
        huber = np.where(dists < thr, 1.0, thr / np.maximum(dists, 1e-6))
        w_iter = inl_w * huber
        center = np.average(inl_pts, axis=0, weights=w_iter)
        v = _weighted_pca(inl_pts, center, w_iter)

    # 整体置信度取内点原始置信度的均值，避免 Huber 权重复计入
    mean_conf = float(np.mean(inl_w))
    return v, center, int(best_cnt), mean_conf


# ---------------------------------------------------------------- 全局精化（核心创新）

def _prism_design(pts: np.ndarray, theta: np.ndarray):
    """方形棱柱模型的残差与雅可比。

    几何先验：塔身是**线性收窄的正方形棱柱**，任一塔身点满足
        max(|dx|, |dy|) = h(z) = h0 − k·z
    其中 (dx, dy) 是该点相对塔轴的偏移，塔轴允许带二次弯曲项：
        axis_x(z) = px + vx·z + ax·z²
    于是"点偏离塔身表面多少"就是一条关于参数的线性方程，
    可以用加权最小二乘一次解出来，不需要逐层切片的中间步骤。
    """
    px, py, vx, vy, h0, k, ax, ay = theta
    z = pts[:, 2]
    dx = pts[:, 0] - (px + vx * z + ax * z * z)
    dy = pts[:, 1] - (py + vy * z + ay * z * z)
    abx = np.abs(dx)
    aby = np.abs(dy)
    use_x = abx >= aby                      # 该点归属 ±X 面还是 ±Y 面
    s = np.where(use_x, np.sign(dx), np.sign(dy))
    s[s == 0] = 1.0
    r = np.where(use_x, abx, aby) - (h0 - k * z)

    J = np.zeros((len(pts), 8))
    J[use_x, 0] = -s[use_x]
    J[use_x, 2] = -s[use_x] * z[use_x]
    J[use_x, 6] = -s[use_x] * z[use_x] ** 2
    J[~use_x, 1] = -s[~use_x]
    J[~use_x, 3] = -s[~use_x] * z[~use_x]
    J[~use_x, 7] = -s[~use_x] * z[~use_x] ** 2
    J[:, 4] = -1.0
    J[:, 5] = z
    return r, J


def _huber_weights(r: np.ndarray, huber_k: float = 1.5, cut: float = 6.0):
    mad = float(np.median(np.abs(r - np.median(r))))
    scale = max(1.4826 * mad, 1e-3)
    d = np.abs(r)
    w = np.where(d <= huber_k * scale, 1.0, huber_k * scale / np.maximum(d, 1e-9))
    w = np.where(d > cut * scale, 0.0, w)          # 硬剔除横担残渣等粗差
    return w, scale


def _balance_weights(pts: np.ndarray, theta: np.ndarray, power: float = 0.5) -> np.ndarray:
    """分面 × 分层的均衡权。

    绕飞扫描时迎机面点密、背机面点稀，同一面在不同高度上的密度也不一样。
    把点按「所在的四个面」×「2m 高度带」分格，每格权重 ∝ n^(-power)：
      power=0   不均衡（直接用原始点，信息利用最充分）
      power=1   完全均衡（每格总权重相同，最抗密度偏斜，但丢掉过采样带来的精度）
      power=0.5 折中（推荐）
    """
    px, py, vx, vy, h0, k, ax, ay = theta
    z = pts[:, 2]
    dx = pts[:, 0] - (px + vx * z + ax * z * z)
    dy = pts[:, 1] - (py + vy * z + ay * z * z)
    face = (dx >= 0).astype(np.int64) * 2 + (dy >= 0).astype(np.int64)
    band = np.floor((z - z.min()) / 2.0).astype(np.int64)
    key = face * 10000 + band
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    if power <= 0:
        return np.ones(len(pts))
    w = np.power(cnt[inv].astype(float), -float(power))
    return w / w.mean()


def refine_axis_prism(pts: np.ndarray, p0: np.ndarray, v: np.ndarray, H: float,
                      balance: bool = True, balance_power: float = 0.5,
                      iters: int = 8, huber_k: float = 1.5,
                      lam: float = 1e-5, strip: bool = True,
                      strip_margin: float = 0.15) -> Optional[dict]:
    """方形棱柱几何约束的全局抗差精化。

    动机：两阶段流程（逐层提中轴点 → 拟合直线）把上万个点压缩成 20 余个观测，
    斜率估计的随机误差被放大到 ~1‰。本方法把"塔身是线性收窄的正方形棱柱"
    这一几何先验直接写进残差函数，对全部塔身点做一次鲁棒 Gauss-Newton 求解，
    观测数从 20 提升到数千（按面×层计等效自由度约 4×20），随机误差随之下降。

    返回 dict(alpha, delta, sigma_alpha, v, params, n_used)；数据不足时返回 None。
    """
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    if n < 200:
        return None
    v = np.asarray(v, dtype=float)
    if abs(v[2]) < 1e-9:
        return None

    px, py = float(p0[0]), float(p0[1])
    vx, vy = float(v[0] / v[2]), float(v[1] / v[2])
    z = pts[:, 2]
    dx = pts[:, 0] - (px + vx * z)
    dy = pts[:, 1] - (py + vy * z)
    h0 = float(np.median(np.maximum(np.abs(dx), np.abs(dy))))
    theta = np.array([px, py, vx, vy, h0, 0.0, 0.0, 0.0], dtype=float)

    def iterate(th, extra_w, n_iter):
        for _ in range(n_iter):
            r, J = _prism_design(pts, th)
            w, _ = _huber_weights(r, huber_k)
            sw = np.sqrt(w * extra_w)
            A = np.vstack([J * sw[:, None], lam * np.eye(8)])
            b = np.concatenate([-r * sw, np.zeros(8)])
            delta = np.linalg.lstsq(A, b, rcond=None)[0]
            th = th + delta
            if np.linalg.norm(delta) < 1e-10:
                break
        return th

    theta = iterate(theta, np.ones(n), iters)

    wb = np.ones(n)
    if balance and balance_power > 0:
        wb = _balance_weights(pts, theta, balance_power)
        theta = iterate(theta, wb, iters)

    # 第二轮：用第一轮拟合出的包络把横担等"塔身之外"的点剔掉再拟合。
    # 比 PCA 包络法干净——包络是以真实塔轴为中心拟合出来的，
    # 不会像 PCA 那样把塔身四角一起切掉。
    if strip:
        px_, py_, vx_, vy_, h0_, k_, ax_, ay_ = theta
        zz = pts[:, 2]
        dx_ = pts[:, 0] - (px_ + vx_ * zz + ax_ * zz * zz)
        dy_ = pts[:, 1] - (py_ + vy_ * zz + ay_ * zz * zz)
        half_ = h0_ - k_ * zz
        inside = np.maximum(np.abs(dx_), np.abs(dy_)) <= (half_ + strip_margin)
        if 200 <= int(inside.sum()) < len(pts):
            pts = pts[inside]          # 必须先替换，iterate 闭包读的是这个变量
            n = len(pts)
            th2 = iterate(theta, np.ones(n), iters)
            if balance and balance_power > 0:
                th2 = iterate(th2, _balance_weights(pts, th2, balance_power), iters)
            theta = th2
            wb = np.ones(n)
            if balance and balance_power > 0:
                wb = _balance_weights(pts, theta, balance_power)

    px, py, vx, vy, h0, k, ax, ay = theta
    if not np.all(np.isfinite(theta)):
        return None

    tx_ = vx * H + ax * H * H
    ty_ = vy * H + ay * H * H
    rim = float(np.hypot(tx_, ty_))
    alpha = rim / max(H, 1e-6) * 1000.0
    delta = float(np.hypot(vx, vy)) * 1000.0
    if not np.isfinite(alpha) or alpha > 500.0:
        return None

    # 参数协方差 → α 的 1σ
    sigma = 0.0
    r, J = _prism_design(pts, theta)
    w, _ = _huber_weights(r, huber_k)
    ww = w * wb
    dof = max(int(np.sum(ww > 0)) - 8, 1)
    s0 = float(np.sqrt(max(np.sum(ww * r * r) / dof, 1e-12)))
    JtWJ = (J * ww[:, None]).T @ J + lam * np.eye(8)
    try:
        cov = np.linalg.pinv(JtWJ) * s0 * s0
        if rim > 1e-9:
            g = np.zeros(8)
            g[2] = 1000.0 * tx_ / rim
            g[3] = 1000.0 * ty_ / rim
            g[6] = 1000.0 * tx_ * H / rim
            g[7] = 1000.0 * ty_ * H / rim
            sigma = float(np.sqrt(max(g @ cov @ g, 0.0)))
    except np.linalg.LinAlgError:
        sigma = 0.0

    v_new = np.array([vx, vy, 1.0])
    v_new /= np.linalg.norm(v_new)

    return {
        "alpha": alpha,
        "delta": delta,
        "sigma_alpha": sigma,
        "v": v_new,
        "params": theta,
        "n_used": int(np.sum(ww > 0)),
    }


def axis_point_at(params: np.ndarray, z: float) -> np.ndarray:
    """由精化参数求某一高度处的塔轴坐标"""
    px, py, vx, vy, h0, k, ax, ay = params
    return np.array([px + vx * z + ax * z * z, py + vy * z + ay * z * z, z])


# ---------------------------------------------------------------- 主流程

def solve(pcd_raw: o3d.geometry.PointCloud,
          cfg: SolverConfig) -> Tuple[TiltResult, dict]:
    """完整解算，返回结果和中间几何体（用于3D显示）"""
    pcd1 = preprocess(pcd_raw, cfg.voxel_size)
    pcd2 = crop_height(pcd1, cfg.z_min, cfg.z_max)
    if len(pcd2.points) < 50:
        raise RuntimeError(
            f"截断区间 {cfg.z_min:.1f}~{cfg.z_max:.1f} m 内只有 {len(pcd2.points)} 个点，"
            "请检查塔高、扫描距离与截断参数是否匹配"
        )
    pcd3 = remove_crossarms_pca(pcd2)
    axis, conf = extract_axis_weighted(pcd3, cfg.window_h, cfg.step)

    if len(axis) < 5:
        raise RuntimeError(
            f"中轴点仅 {len(axis)} 个（需≥5），请增大窗口高度或放宽截断区间"
        )

    thr = cfg.ransac_thr if cfg.ransac_thr is not None else _auto_threshold(axis)
    v, p0, n_inl, mc = fit_line_robust(axis, conf, thr, cfg.max_iter)

    H = max(float(cfg.tower_height), 1e-3)
    zs = axis[:, 2]

    # ================= 全局精化（核心创新点） =================
    refined = None
    if cfg.use_global_refine:
        # 直接用截断后的原始点云：横担点交给精化内部的"拟合包络二次剔除"处理，
        # 比先用 PCA 包络硬切更干净（PCA 包络会连带切掉塔身四角）
        refined = refine_axis_prism(np.asarray(pcd2.points), p0, v, H,
                                    balance=cfg.use_balance_weight,
                                    balance_power=cfg.balance_power)

    if refined is not None:
        v = refined["v"]
        alpha_std = refined["alpha"]
        delta_permille = refined["delta"]
        sigma_alpha = refined["sigma_alpha"]
        tip_offset = alpha_std / 1000.0 * H
        params = refined["params"]
    else:
        # 退化路径：对轴点做二次拟合，一次项即塔基斜率
        try:
            qx = np.polyfit(zs, axis[:, 0], 2)
            qy = np.polyfit(zs, axis[:, 1], 2)
        except (np.linalg.LinAlgError, ValueError):
            qx = np.array([0.0, v[0] / max(v[2], 1e-9), p0[0]])
            qy = np.array([0.0, v[1] / max(v[2], 1e-9), p0[1]])
        x0, x_top = float(np.polyval(qx, 0.0)), float(np.polyval(qx, H))
        y0, y_top = float(np.polyval(qy, 0.0)), float(np.polyval(qy, H))
        tip_offset = float(np.hypot(x_top - x0, y_top - y0))
        alpha_std = tip_offset / H * 1000.0
        delta_permille = float(np.hypot(qx[1], qy[1])) * 1000.0
        sigma_alpha = 0.0
        params = None

    # 物理约束：塔身弯曲只会让塔顶偏移大于塔基倾角，δ 不可能超过 α
    delta_permille = float(np.clip(delta_permille, 0.0, max(alpha_std, 0.0)))

    # 统一用"塔顶偏移矢量"反算各方向角，保证 总角 与 倾斜率α 口径一致
    if params is not None:
        px, py, vx, vy, _h0, _k, ax, ay = params
        tx_vec = vx * H + ax * H * H
        ty_vec = vy * H + ay * H * H
    else:
        tx_vec = v[0] / max(v[2], 1e-9) * H
        ty_vec = v[1] / max(v[2], 1e-9) * H
    tx = np.rad2deg(np.arctan2(abs(tx_vec), H))
    ty = np.rad2deg(np.arctan2(abs(ty_vec), H))
    total_deg = np.rad2deg(np.arctan2(float(np.hypot(tx_vec, ty_vec)), H))
    rate = alpha_std

    # ---- 样条轴线重建 + 折角定位（基元拟合）----
    spline = None
    bend = None
    if cfg.use_spline and params is not None:
        try:
            from core.deform_analysis import fit_axis_spline, locate_bend
            p_init = params if params is not None else np.array(
                [p0[0], p0[1], v[0] / max(v[2], 1e-9), v[1] / max(v[2], 1e-9),
                 0.0, 0.0, 0.0, 0.0])
            spline = fit_axis_spline(np.asarray(pcd2.points), p_init, H,
                                     knot_spacing=cfg.knot_spacing)
            if spline is not None:
                bend = locate_bend(spline, cfg.z_min, cfg.z_max)
        except Exception:
            spline = None
            bend = None

    # ---- 数据质量：拟合残差 / 塔身半宽 ----
    # 真实杆塔的点都贴在棱柱面上，这个比值约 0.03；如果导入的是噪声、植被
    # 或坐标错位的点云，比值会到 0.3 以上。必须拦住，否则会给出一个看起来
    # 很正常的倾斜率（实测纯噪声能算出 α≈19‰）。
    #
    # 分母用**数据自身的棱柱半径**（相对拟合直线，恒为正），
    # 不能用拟合出的 h0 —— 数据无意义时 h0 会发散成负值，判据就失效了。
    quality, q_ratio = "未知", 0.0
    deflect_mm, deflect_ratio = 0.0, 0.0
    try:
        pa = np.asarray(pcd3.points)
        d = pa - p0
        perp = d - np.outer(d @ v, v)
        rr = np.maximum(np.abs(perp[:, 0]), np.abs(perp[:, 1]))
        half_data = float(np.median(rr))

        if spline is not None:
            rms_mm = float(spline.rms_mm)
            from core.deform_analysis import bending_metrics
            bm = bending_metrics(spline, cfg.z_min, cfg.z_max)
            deflect_mm = float(bm.deflection_mm)
            deflect_ratio = float(bm.ratio_permille)
        else:
            # 没做样条重建时，用"棱柱 + 线性锥度"估残差；
            # 直接对 rr 取残差会把锥度本身当成误差，真实塔也会被判不合格
            zb = np.floor((pa[:, 2] - pa[:, 2].min()) / 1.0).astype(int)
            zc, rc = [], []
            for b in np.unique(zb):
                m = zb == b
                if int(m.sum()) >= 30:
                    zc.append(float(np.median(pa[m, 2])))
                    rc.append(float(np.median(rr[m])))
            if len(zc) >= 3:
                h_fit = np.polyval(np.polyfit(np.array(zc), np.array(rc), 1), pa[:, 2])
            else:
                h_fit = np.full_like(rr, half_data)
            rms_mm = float(np.sqrt(np.mean((rr - h_fit) ** 2)) * 1000.0)

        if half_data > 0.05 and rms_mm > 0:
            q_ratio = rms_mm / (half_data * 1000.0)
            quality = ("合格" if q_ratio < 0.12
                       else ("可疑" if q_ratio < 0.25 else "不合格"))
    except Exception:
        quality, q_ratio = "未知", 0.0

    result = TiltResult(float(tx), float(ty), float(total_deg), float(rate),
                        mc, len(axis), n_inl,
                        tip_offset_mm=tip_offset * 1000.0,
                        alpha_standard_permille=alpha_std,
                        foundation_tilt_permille=delta_permille,
                        axis_curve_permille=alpha_std - delta_permille,
                        alpha_sigma_permille=float(sigma_alpha),
                        refine_used=refined is not None,
                        refine_points=(refined["n_used"] if refined else 0),
                        bend_found=bool(bend.found) if bend else False,
                        bend_height=float(bend.z_break) if bend else 0.0,
                        bend_delta_permille=float(bend.delta_slope) if bend else 0.0,
                        bend_grade=(bend.grade if bend else "无"),
                        spline_rms_mm=float(spline.rms_mm) if spline else 0.0,
                        quality=quality, quality_ratio=float(q_ratio),
                        deflection_mm=float(deflect_mm),
                        deflection_ratio_permille=float(deflect_ratio))

    # 构造中间几何体
    axis_pcd = o3d.geometry.PointCloud()
    axis_pcd.points = o3d.utility.Vector3dVector(axis)

    # 拟合轴线：优先用精化后的曲线（能显示塔身弯曲），否则用直线段
    z_lo = float(zs.min()) - 2.0
    z_hi = float(zs.max()) + 2.0
    if spline is not None:
        line_pts = spline.world(np.linspace(spline.z0, spline.z1, 30))
    elif params is not None:
        line_pts = np.array([axis_point_at(params, zz)
                             for zz in np.linspace(z_lo, z_hi, 24)])
    elif abs(v[2]) > 1e-6:
        t_lo = (z_lo - p0[2]) / v[2]
        t_hi = (z_hi - p0[2]) / v[2]
        line_pts = np.array([p0 + v * t_lo, p0 + v * t_hi])
    else:
        line_pts = np.array([p0 + v * -15.0, p0 + v * 15.0])

    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(line_pts)
    line.lines = o3d.utility.Vector2iVector(
        [[i, i + 1] for i in range(len(line_pts) - 1)])

    intermediates = {
        "raw": pcd_raw,
        "preprocessed": pcd1,
        "cropped": pcd2,
        "crossarm_removed": pcd3,
        "axis_points": axis_pcd,
        "fitted_line": line,
        # 供横担健康分析使用：含横担的截断点云 + 塔轴参数
        "crossarm_cloud": pcd2,
        "axis_params": (params if params is not None else
                        np.array([p0[0], p0[1],
                                  v[0] / max(v[2], 1e-9), v[1] / max(v[2], 1e-9),
                                  0.0, 0.0, 0.0, 0.0])),
        "axis_spline": spline,
    }
    return result, intermediates
