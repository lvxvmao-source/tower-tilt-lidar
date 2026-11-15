# -*- coding: utf-8 -*-
"""
杆塔变形分析：基元拟合定位弯曲 + 两期点云对比

两个相对行业常规做法的改造点：

【1】基元拟合：从"分段拟合"改成"单一基元 + 样条轴线 + 曲率正则"
    常规做法是把点云沿高度切成若干段，每段独立拟合平面/圆柱再拼起来。
    问题是段间不连续、短段杠杆太短噪声大，拼出来的"轴线"会抖，没法求曲率。
    这里改成：仍用"方形棱柱"这一个基元贯穿全塔，但塔轴不再假设为直线或
    二次曲线，而用**夹紧三次 B 样条**表示（曲线始终落在控制多边形内，
    不会在数据边界外推爆炸），并用**曲率惩罚**约束光滑性：
        min  Σ wᵢ·rᵢ(θ)²  +  λ·∫|x''(z)|²dz + λ·∫|y''(z)|²dz
    一次解出连续可导的塔轴，曲率剖面直接指出"哪一节弯得最厉害"。

【2】两期对比：从"点对点距离"改成"轴对轴比较"
    M3C2 / C2C 那类方法要把两期点云精确配准后逐点算距离，代价是配准误差
    直接进结果、对点密度差异敏感、输出是一堆距离而不是结构量。
    这里改成两期各自独立重建塔轴（以各自塔基中心为原点），直接比较两条
    轴线——免配准、与点云密度无关，输出的就是"哪个高度位移了多少"。
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------- B 样条基

def _clamped_knots(z0: float, z1: float, n_inner: int, degree: int = 3) -> np.ndarray:
    inner = np.linspace(z0, z1, n_inner + 2)[1:-1]
    return np.concatenate([[z0] * (degree + 1), inner, [z1] * (degree + 1)])


def _bsp_one(i: int, p: int, knots: np.ndarray, z: np.ndarray) -> np.ndarray:
    if p == 0:
        lo, hi = knots[i], knots[i + 1]
        if i == len(knots) - 2:
            return ((z >= lo) & (z <= hi)).astype(float)
        return ((z >= lo) & (z < hi)).astype(float)
    d1 = knots[i + p] - knots[i]
    d2 = knots[i + p + 1] - knots[i + 1]
    out = np.zeros_like(z)
    if d1 > 0:
        out = out + (z - knots[i]) / d1 * _bsp_one(i, p - 1, knots, z)
    if d2 > 0:
        out = out + (knots[i + p + 1] - z) / d2 * _bsp_one(i + 1, p - 1, knots, z)
    return out


def _clip_span(z: np.ndarray, knots: np.ndarray, p: int = 3) -> np.ndarray:
    """把 z 夹到 [z0, z1) 内。

    Cox-de Boor 的半开区间在右端点会全部失效（基函数全 0），
    必须用 nextafter 退一个 ulp，否则塔顶会塌到 0。
    """
    z = np.asarray(z, dtype=float)
    hi = np.nextafter(knots[-p - 1], -np.inf)
    return np.clip(z, knots[p], hi)


def _bsp(z: np.ndarray, knots: np.ndarray, p: int = 3) -> np.ndarray:
    z = _clip_span(z, knots, p)
    n = len(knots) - p - 1
    B = np.zeros((len(z), n))
    for i in range(n):
        B[:, i] = _bsp_one(i, p, knots, z)
    return B


def _bsp_d1(z: np.ndarray, knots: np.ndarray, p: int = 3) -> np.ndarray:
    """dN/dz，与 _bsp 列数一致"""
    z = _clip_span(z, knots, p)
    n = len(knots) - p - 1
    Bm = _bsp(z, knots, p - 1)                 # 列数 n+1
    D = np.zeros((len(z), n))
    for i in range(n):
        d1 = knots[i + p] - knots[i]
        d2 = knots[i + p + 1] - knots[i + 1]
        if d1 > 0:
            D[:, i] += p * Bm[:, i] / d1
        if d2 > 0 and i + 1 < Bm.shape[1]:
            D[:, i] -= p * Bm[:, i + 1] / d2
    return D


def _bsp_d2(z: np.ndarray, knots: np.ndarray, p: int = 3) -> np.ndarray:
    """d²N/dz²，与 _bsp 列数一致"""
    z = _clip_span(z, knots, p)
    n = len(knots) - p - 1
    Dm = _bsp_d1(z, knots, p - 1)              # 列数 n+1
    D2 = np.zeros((len(z), n))
    for i in range(n):
        d1 = knots[i + p] - knots[i]
        d2 = knots[i + p + 1] - knots[i + 1]
        if d1 > 0:
            D2[:, i] += p * Dm[:, i] / d1
        if d2 > 0 and i + 1 < Dm.shape[1]:
            D2[:, i] -= p * Dm[:, i + 1] / d2
    return D2


# ---------------------------------------------------------------- 加权工具

def _huber_w(r: np.ndarray, k: float = 1.5, cut: float = 6.0) -> np.ndarray:
    mad = float(np.median(np.abs(r - np.median(r))))
    scale = max(1.4826 * mad, 1e-3)
    d = np.abs(r)
    w = np.where(d <= k * scale, 1.0, k * scale / np.maximum(d, 1e-9))
    return np.where(d > cut * scale, 0.0, w)


def _balance_w(z: np.ndarray, dx: np.ndarray, dy: np.ndarray, power: float = 0.3):
    if power <= 0:
        return np.ones(len(z))
    face = (dx >= 0).astype(np.int64) * 2 + (dy >= 0).astype(np.int64)
    band = np.floor((z - z.min()) / 2.0).astype(np.int64)
    _, inv, cnt = np.unique(face * 10000 + band, return_inverse=True, return_counts=True)
    w = np.power(cnt[inv].astype(float), -power)
    return w / w.mean()


def _beam_frame(params0: np.ndarray):
    """由全局塔轴方向构造梁坐标系 (v, e1, e2)。

    关键：棱柱约束必须在**轴垂直截面**里检验。塔一倾斜，水平截面会被拉成
    平行四边形，宽度又被锥度放大，底部就凭空生出很大的假曲率。
    """
    px, py, vx, vy, h0, k, ax, ay = np.asarray(params0, dtype=float)
    v = np.array([vx, vy, 1.0], dtype=float)
    v /= np.linalg.norm(v)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(ref @ v)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = ref - float(ref @ v) * v
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(v, e1)
    return v, e1, e2


# ---------------------------------------------------------------- 轴线对象

@dataclass
class AxisSpline:
    """夹紧三次 B 样条表示的塔轴 x(z), y(z)"""
    knots: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    h0: float
    k: float
    z0: float
    z1: float
    v: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    e1: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    e2: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    n_points: int = 0
    rms_mm: float = 0.0

    def value(self, z) -> np.ndarray:
        """塔轴横向坐标 (u, w)。超出数据范围时按端点斜率线性外推。

        不能简单夹住：塔高 H 通常大于点云覆盖的最高处，夹住等于把
        塔顶那一段的偏移丢掉，α 会被系统性低估。
        """
        tt = np.atleast_1d(np.asarray(z, dtype=float))
        tc = np.clip(tt, self.z0, self.z1)
        B = _bsp(tc, self.knots)
        uv = np.column_stack([B @ self.cx, B @ self.cy])
        for mask, ref, delta in ((tt > self.z1, self.z1, tt - self.z1),
                                 (tt < self.z0, self.z0, tt - self.z0)):
            if mask.any():
                sl = self.slope(np.array([ref]))[0]
                uv[mask] += np.outer(delta[mask], sl)
        return uv

    def slope(self, z) -> np.ndarray:
        zz = np.clip(np.atleast_1d(np.asarray(z, dtype=float)), self.z0, self.z1)
        B1 = _bsp_d1(zz, self.knots)
        return np.column_stack([B1 @ self.cx, B1 @ self.cy])

    def direction_at_end(self, top: bool = True) -> np.ndarray:
        zz = self.z1 if top else self.z0
        return self.slope(np.array([zz]))[0]

    def curvature(self, z) -> Tuple[np.ndarray, np.ndarray]:
        """曲率 κ(z)（1/m）与弯曲方位角（度，指向曲率中心）"""
        zz = np.clip(np.atleast_1d(np.asarray(z, dtype=float)), self.z0, self.z1)
        B1 = _bsp_d1(zz, self.knots)
        B2 = _bsp_d2(zz, self.knots)
        x1, y1 = B1 @ self.cx, B1 @ self.cy
        x2, y2 = B2 @ self.cx, B2 @ self.cy
        num = np.sqrt(y2 ** 2 + x2 ** 2 + (x1 * y2 - y1 * x2) ** 2)
        den = np.power(x1 ** 2 + y1 ** 2 + 1.0, 1.5)
        kappa = num / np.maximum(den, 1e-9)
        az = np.degrees(np.arctan2(y2, x2))
        return kappa, az

    def world(self, t) -> np.ndarray:
        """由梁坐标还原世界坐标（用于 3D 显示）"""
        tt = np.atleast_1d(np.asarray(t, dtype=float))
        uv = self.value(tt)
        return (uv[:, 0:1] * self.e1[None, :]
                + uv[:, 1:2] * self.e2[None, :]
                + tt[:, None] * self.v[None, :])

    def alpha_permille(self, H: float) -> float:
        """倾斜率 α：必须在**世界坐标**下算。

        梁坐标系里的横向偏移只反映"相对整体倾斜方向的残余弯曲"，
        整体倾斜已经被吸收进基向量 v 了，直接用梁坐标会得到接近 0 的假值。
        """
        p0 = self.world(0.0)[0]
        pH = self.world(H)[0]
        dh = float(np.hypot(pH[0] - p0[0], pH[1] - p0[1]))
        dz = float(pH[2] - p0[2])
        return dh / max(abs(dz), 1e-6) * 1000.0

    def delta_permille(self) -> float:
        """基础倾斜率 δ：塔基处轴线相对铅垂线的斜率（世界坐标）"""
        p0 = self.world(0.0)[0]
        p1 = self.world(1.0)[0]
        dx = float(p1[0] - p0[0])
        dy = float(p1[1] - p0[1])
        dz = float(p1[2] - p0[2])
        return float(np.hypot(dx, dy) / max(abs(dz), 1e-6) * 1000.0)


# ---------------------------------------------------------------- 拟合

def fit_axis_spline(pts: np.ndarray, params0: np.ndarray, H: float,
                    knot_spacing: float = 2.0, lam: float = 1.0e4,
                    prior_mu: float = 2.0e3, iters: int = 8,
                    balance_power: float = 0.3,
                    strip_margin: float = 0.2) -> Optional[AxisSpline]:
    """在方形棱柱基元约束下拟合连续塔轴。

    求解变量是**相对全局棱柱轴的样条修正量** Δ(t)，不是轴线本身：
        x(t) = x_global(t) + Δx(t)
    好处是初值 Δ=0 就落在合理解附近（全局轴已经很准），不会因为初值离谱
    让 Huber 权重把观测全剔掉。目标函数：
        Σ wᵢ·rᵢ² + λ∫|x''|² + μΣΔ²
    曲率项保证光滑，先验项防止局部跑飞。
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 400 or params0 is None:
        return None

    px0, py0, vx0, vy0, h00, k00, aq0x, aq0y = np.asarray(params0, dtype=float)

    # 按全局包络剔除横担等塔身之外的点
    zz = pts[:, 2]
    dx0 = pts[:, 0] - (px0 + vx0 * zz + aq0x * zz * zz)
    dy0 = pts[:, 1] - (py0 + vy0 * zz + aq0y * zz * zz)
    inside = np.maximum(np.abs(dx0), np.abs(dy0)) <= (h00 - k00 * zz + strip_margin)
    if 400 <= int(inside.sum()) < len(pts):
        pts = pts[inside]
    if len(pts) < 400:
        return None

    v, e1, e2 = _beam_frame(params0)
    t_all = pts @ v
    u_all = pts @ e1
    w_all = pts @ e2
    z, x, y = t_all, u_all, w_all

    z0, z1 = float(np.percentile(z, 1)), float(np.percentile(z, 99))
    span = z1 - z0
    if span < 4.0:
        return None

    n_inner = int(max(2, round(span / max(knot_spacing, 0.5)) - 1))
    knots = _clamped_knots(z0, z1, n_inner)
    A = _bsp(z, knots)
    nc = A.shape[1]

    # 全局棱柱轴投影到样条基，作为先验中心
    tg = np.linspace(z0, z1, 200)
    zg = tg * v[2]
    P = np.column_stack([px0 + vx0 * zg + aq0x * zg * zg,
                         py0 + vy0 * zg + aq0y * zg * zg,
                         zg])
    Ag = _bsp(tg, knots)
    cx_g = np.linalg.lstsq(Ag, P @ e1, rcond=None)[0]
    cy_g = np.linalg.lstsq(Ag, P @ e2, rcond=None)[0]

    h0_i = float(h00)
    k_i = float(k00 * v[2])
    theta = np.concatenate([np.zeros(nc), np.zeros(nc), [h0_i, k_i]])
    npar = 2 * nc + 2

    zg2 = np.linspace(z0, z1, 120)
    Ad2 = _bsp_d2(zg2, knots)
    sl = np.sqrt(lam)
    sp_ = np.sqrt(prior_mu)
    Lpen = np.zeros((Ad2.shape[0] * 2 + 2 * nc, npar))
    bpen = np.zeros(Lpen.shape[0])
    Lpen[:Ad2.shape[0], :nc] = sl * Ad2
    Lpen[Ad2.shape[0]:2 * Ad2.shape[0], nc:2 * nc] = sl * Ad2
    bpen[:Ad2.shape[0]] = -sl * (Ad2 @ cx_g)
    bpen[Ad2.shape[0]:2 * Ad2.shape[0]] = -sl * (Ad2 @ cy_g)
    r0 = 2 * Ad2.shape[0]
    Lpen[r0:r0 + nc, :nc] = sp_ * np.eye(nc)
    Lpen[r0 + nc:r0 + 2 * nc, nc:2 * nc] = sp_ * np.eye(nc)

    w_bal = np.ones(len(z))
    for it in range(iters):
        dxx, dyy = theta[:nc], theta[nc:2 * nc]
        cx = cx_g + dxx
        cy = cy_g + dyy
        h0_, k_ = theta[-2], theta[-1]
        dx = x - A @ cx
        dy = y - A @ cy
        abx, aby = np.abs(dx), np.abs(dy)
        use_x = abx >= aby
        s = np.where(use_x, np.sign(dx), np.sign(dy))
        s[s == 0] = 1.0
        r = np.where(use_x, abx, aby) - (h0_ - k_ * z)

        J = np.zeros((len(z), npar))
        J[use_x, :nc] = -s[use_x, None] * A[use_x]
        J[~use_x, nc:2 * nc] = -s[~use_x, None] * A[~use_x]
        J[:, -2] = -1.0
        J[:, -1] = z

        w = _huber_w(r) * w_bal
        sw = np.sqrt(w)
        Aa = np.vstack([J * sw[:, None], Lpen])
        ba = np.concatenate([-r * sw, bpen])
        delta = np.linalg.lstsq(Aa, ba, rcond=None)[0]
        theta = theta + delta

        if it == 1 and balance_power > 0:
            w_bal = _balance_w(z, x - A @ (cx_g + theta[:nc]),
                               y - A @ (cy_g + theta[nc:2 * nc]), balance_power)
        if np.linalg.norm(delta) < 1e-12:
            break

    cx = cx_g + theta[:nc]
    cy = cy_g + theta[nc:2 * nc]
    sp = AxisSpline(knots=knots, cx=cx, cy=cy,
                    h0=float(theta[-2]), k=float(theta[-1]),
                    z0=z0, z1=z1, v=v, e1=e1, e2=e2, n_points=len(z))
    r = np.maximum(np.abs(x - A @ cx), np.abs(y - A @ cy)) - (sp.h0 - sp.k * z)
    sp.rms_mm = float(np.sqrt(np.mean(r ** 2)) * 1000.0)
    return sp


# ---------------------------------------------------------------- 弯曲定位

@dataclass
class BendingMetrics:
    z_peak: float = 0.0
    kappa_max: float = 0.0
    radius_min: float = float("inf")
    deflection_mm: float = 0.0
    z_deflect: float = 0.0
    ratio_permille: float = 0.0
    azimuth_deg: float = 0.0
    grade: str = "—"
    zs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    kappa: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dev_mm: np.ndarray = field(default_factory=lambda: np.zeros(0))


GRADE_LEVELS = ((0.5, "轻微"), (2.0, "明显"), (float("inf"), "严重"))


def bending_metrics(sp: AxisSpline, z0: float = None, z1: float = None,
                    n: int = 240, margin: float = 0.5) -> BendingMetrics:
    """从样条轴线提取弯曲定位指标。

    矢高 deflection = 轴线上各点相对"两端点连线"的横向偏离（结构上的标准量）
    矢跨比 = 矢高 / 跨越长度（‰）
    """
    za = (sp.z0 + margin) if z0 is None else float(z0) + margin
    zb = (sp.z1 - margin) if z1 is None else float(z1) - margin
    if zb <= za:
        return BendingMetrics()

    zs = np.linspace(za, zb, n)
    kappa, az = sp.curvature(zs)
    i = int(np.argmax(kappa))

    pts = sp.value(zs)
    p0 = sp.value(za)[0]
    p1 = sp.value(zb)[0]
    t = (zs - za) / (zb - za)
    chord = p0[None, :] + (p1 - p0)[None, :] * t[:, None]
    dev = np.linalg.norm(pts - chord, axis=1) * 1000.0
    j = int(np.argmax(dev))

    ratio = dev[j] / max(zb - za, 1e-6)
    grade = next(name for th, name in GRADE_LEVELS if ratio < th)

    return BendingMetrics(
        z_peak=float(zs[i]), kappa_max=float(kappa[i]),
        radius_min=float(1.0 / max(kappa[i], 1e-12)),
        deflection_mm=float(dev[j]), z_deflect=float(zs[j]),
        ratio_permille=float(ratio), azimuth_deg=float(az[i]), grade=grade,
        zs=zs, kappa=kappa, dev_mm=dev,
    )


@dataclass
class BendLocation:
    """局部弯曲的定位结果（基于斜率剖面的变点检测）"""
    found: bool = False
    z_break: float = 0.0          # 折角高度 m
    slope_below: float = 0.0      # 折点以下平均斜率 ‰
    slope_above: float = 0.0      # 折点以上平均斜率 ‰
    delta_slope: float = 0.0      # 斜率增量 ‰
    r2_gain: float = 0.0          # 两段模型相对单段的误差下降比例
    grade: str = "无"


def locate_bend(sp: AxisSpline, z0: float, z1: float,
                min_delta_permille: float = 1.5,
                min_gain: float = 0.30) -> BendLocation:
    """沿高度用**两段常斜率模型**拟合斜率剖面，最优断点即折角位置。

    为什么不直接取曲率极值：曲率是斜率的一阶导，对轴心噪声极其敏感
    （二阶差分把 10mm 的位置噪声放大成 0.005 的曲率噪声）。
    变点模型只用一阶量、且利用了整段数据，鲁棒得多。
    """
    za = z0 + 0.08 * (z1 - z0)
    zb = z1 - 0.08 * (z1 - z0)
    if zb <= za:
        return BendLocation()

    zs = np.linspace(za, zb, 300)
    th = np.linalg.norm(sp.slope(zs), axis=1) * 1000.0
    base = float(np.mean((th - th.mean()) ** 2))
    if base <= 0:
        return BendLocation()

    best_sse, best_hb, best_a, best_b = np.inf, 0.0, 0.0, 0.0
    for hb in np.linspace(za, zb, 60):
        m1 = zs < hb
        m2 = ~m1
        if m1.sum() < 12 or m2.sum() < 12:
            continue
        a = float(th[m1].mean())
        b = float(th[m2].mean())
        sse = float(((th[m1] - a) ** 2).sum() + ((th[m2] - b) ** 2).sum())
        if sse < best_sse:
            best_sse, best_hb, best_a, best_b = sse, float(hb), a, b

    if not np.isfinite(best_sse):
        return BendLocation()

    gain = 1.0 - best_sse / (base * len(th))
    delta = best_b - best_a
    found = (abs(delta) >= min_delta_permille) and (gain >= min_gain)

    if not found:
        grade = "无"
    elif abs(delta) < 5.0:
        grade = "轻微"
    elif abs(delta) < 12.0:
        grade = "明显"
    else:
        grade = "严重"

    return BendLocation(found=found, z_break=best_hb,
                        slope_below=best_a, slope_above=best_b,
                        delta_slope=delta, r2_gain=float(gain), grade=grade)


# ---------------------------------------------------------------- 两期对比

@dataclass
class EpochComparison:
    days: float = 0.0
    alpha1: float = 0.0
    alpha2: float = 0.0
    delta_alpha: float = 0.0
    delta1: float = 0.0
    delta2: float = 0.0
    delta_delta: float = 0.0
    base_shift_mm: float = 0.0
    tip_shift_mm: float = 0.0
    max_shift_mm: float = 0.0
    z_max_shift: float = 0.0
    rate_mm_per_year: float = 0.0
    zs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    shift_mm: np.ndarray = field(default_factory=lambda: np.zeros(0))


def compare_epochs(sp1: AxisSpline, sp2: AxisSpline, H: float,
                   days: float = 0.0, n: int = 160,
                   margin: float = 0.5) -> EpochComparison:
    """比较两期塔轴（各自以塔基中心为原点）。"""
    za = max(sp1.z0, sp2.z0) + margin
    zb = min(sp1.z1, sp2.z1) - margin
    if zb <= za:
        return EpochComparison()

    zs = np.linspace(za, zb, n)
    p1 = sp1.world(zs)
    p2 = sp2.world(zs)
    b1 = sp1.world(0.0)[0]
    b2 = sp2.world(0.0)[0]

    # 只比较水平位移（竖向沉降是另一套指标），且都以各自塔基为原点
    d1 = p1[:, :2] - b1[None, :2]
    d2 = p2[:, :2] - b2[None, :2]
    shift = np.linalg.norm(d2 - d1, axis=1) * 1000.0

    k = int(np.argmax(shift))
    a1, a2 = sp1.alpha_permille(H), sp2.alpha_permille(H)
    dlt1, dlt2 = sp1.delta_permille(), sp2.delta_permille()

    tip1 = float(np.linalg.norm(sp1.world(H)[0][:2] - b1[:2])) * 1000.0
    tip2 = float(np.linalg.norm(sp2.world(H)[0][:2] - b2[:2])) * 1000.0

    rate = float(shift[k]) / float(days) * 365.0 if days and days > 0 else 0.0

    return EpochComparison(
        days=float(days),
        alpha1=a1, alpha2=a2, delta_alpha=a2 - a1,
        delta1=dlt1, delta2=dlt2, delta_delta=dlt2 - dlt1,
        base_shift_mm=float(np.linalg.norm(b2[:2] - b1[:2]) * 1000.0),
        tip_shift_mm=float(tip2 - tip1),
        max_shift_mm=float(shift[k]),
        z_max_shift=float(zs[k]),
        rate_mm_per_year=rate,
        zs=zs, shift_mm=shift,
    )
