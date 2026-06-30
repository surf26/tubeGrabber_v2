"""
纯数学：像素反投影、齐次变换、深度采样。
全部是无副作用的函数，不碰硬件，可单独 import 做单元测试。

坐标系记号：
    p_cam   相机坐标系下的 3D 点 (x右, y下, z前)，单位米
    p_base  机械臂基座坐标系下的 3D 点
    T_a2b   把 a 坐标系的点变到 b 坐标系：p_b = T_a2b @ p_a
"""
from __future__ import annotations

import numpy as np

from core.types import Intrinsics


def invert(T: np.ndarray) -> np.ndarray:
    """齐次变换求逆（比 np.linalg.inv 更稳更快）。"""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """由旋转 3x3 和平移 3 拼出 4x4 齐次变换。"""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """用 4x4 齐次变换把 3D 点 p 变换到另一个坐标系，返回 (3,)。"""
    p = np.asarray(p, dtype=np.float64).reshape(3)
    ph = np.array([p[0], p[1], p[2], 1.0])
    return (T @ ph)[:3]


def deproject(u: float, v: float, z: float, intr: Intrinsics) -> np.ndarray:
    """
    像素 (u,v) + 深度 z(米) -> 相机坐标系 3D 点 (米)。这是结算的第一步。
        X = (u - cx) * z / fx
        Y = (v - cy) * z / fy
        Z = z
    """
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    return np.array([x, y, z], dtype=np.float64)


def sample_depth(depth_img: np.ndarray, u: float, v: float, *,
                 patch: int, scale: float,
                 z_min: float, z_max: float) -> float | None:
    """
    在像素 (u,v) 周围取 patch×patch 窗口，把原始深度乘 scale 换算成米，
    剔除 0 和超出 [z_min, z_max] 的无效值后取中位数。

    返回米；窗口内全无效则返回 None（上层应跳过这个目标）。
    单像素深度经常是空洞(0)，所以一定要取邻域中位数，不能直接读中心点。
    """
    h, w = depth_img.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    r = max(patch // 2, 0)
    u0, u1 = max(ui - r, 0), min(ui + r + 1, w)
    v0, v1 = max(vi - r, 0), min(vi + r + 1, h)
    if u0 >= u1 or v0 >= v1:
        return None

    region = depth_img[v0:v1, u0:u1].astype(np.float64) * scale
    valid = region[(region > z_min) & (region < z_max)]
    if valid.size == 0:
        return None
    return float(np.median(valid))
