"""
工程里流转的数据结构。全部普通 dataclass，方便打印、序列化、测试。

数据流：
    Detection(YOLO,含类别) ─┐
                           ├─> PoseSolver ─> GraspTarget ─> RackModel ─> RackMap
    depth_img(相机)       ─┘                                  │
                                                  Task(行列号) ┘─> GraspPlanner ─> GraspPlan
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Intrinsics:
    """相机内参（针孔模型）。必须是'深度已对齐到的那一路彩色流'的内参。"""
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_matrix(cls, K: np.ndarray) -> "Intrinsics":
        return cls(fx=float(K[0, 0]), fy=float(K[1, 1]),
                   cx=float(K[0, 2]), cy=float(K[1, 2]))


@dataclass
class Detection:
    """一次目标检测结果（像素坐标系）。cls 区分 试管/空槽。"""
    pixel: tuple[float, float]                 # 目标中心 (u, v)
    bbox: tuple[float, float, float, float]    # [x1, y1, x2, y2]
    conf: float
    cls: int = 0


@dataclass
class GraspTarget:
    """结算后的一个目标（试管或空槽），含相机系和基座系下的 3D 位置（米）。"""
    base_xyz: np.ndarray        # (3,) 基座坐标系下位置（最终要用的）
    cam_xyz: np.ndarray         # (3,) 相机坐标系下位置（中间量，便于调试）
    pixel: tuple[float, float]  # 对应像素
    depth_m: float              # 采样到的深度（米）
    conf: float
    cls: int = 0                # 类别：试管 or 空槽（对照 config.TUBE_CLS/EMPTY_CLS）
    source: str = "hand_cam"    # 来自哪台相机（本工程只有眼在手上一台）
    bbox: tuple[float, float, float, float] | None = None  # [x1,y1,x2,y2] 画图用

    def __repr__(self) -> str:
        x, y, z = self.base_xyz
        return (f"GraspTarget(cls={self.cls} base=({x*1000:.1f},{y*1000:.1f},{z*1000:.1f})mm "
                f"depth={self.depth_m*1000:.1f}mm conf={self.conf:.2f} pixel={self.pixel})")


@dataclass
class GraspPlan:
    """一次抓取/放置的三个关键位姿，每个都是 [x,y,z,rx,ry,rz]（米/弧度，基座系）。"""
    approach: list[float]                                # 预抓取/预放置点（目标正上方）
    grasp: list[float] = field(default_factory=list)     # 抓取/放置点（下扎到位）
    retreat: list[float] = field(default_factory=list)   # 抬起点
