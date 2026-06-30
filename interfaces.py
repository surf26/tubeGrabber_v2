"""
接口契约层 —— 整个工程的"插座"。

pipeline / align 只依赖这里定义的抽象协议，不关心背后是真硬件(hardware/)还是 Mock(mocks/)。
用 typing.Protocol（结构化类型）：实现类不需显式继承，方法签名对得上即视为实现了该协议。
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from core.types import Detection, Intrinsics


@runtime_checkable
class RGBDCamera(Protocol):
    """彩色 + 深度（深度已对齐到彩色）相机。"""

    @property
    def intrinsics(self) -> Intrinsics: ...
    @property
    def depth_scale(self) -> float: ...

    def get_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 (color_bgr[H,W,3] uint8, depth[H,W] 原始单位)，两者像素一一对齐。"""
        ...

    def stop(self) -> None: ...


@runtime_checkable
class Detector(Protocol):
    """目标检测器（双类 YOLO：试管 / 空槽）。"""

    def detect(self, color_bgr: np.ndarray) -> list[Detection]: ...


@runtime_checkable
class Robot(Protocol):
    """机械臂。"""

    def get_flange2base(self) -> np.ndarray:
        """末端(flange) -> 基座 的 4x4 齐次变换。"""
        ...

    def get_pose(self) -> list[float]:
        """当前末端位姿 [x,y,z,rx,ry,rz]（米/弧度）。"""
        ...

    def move_to_pose(self, pose: list[float], speed: float, block: bool = True) -> None:
        """笛卡尔位姿运动到 [x,y,z,rx,ry,rz]（米/弧度，基座系）。"""
        ...

    def enable_gripper_power(self) -> None:
        """给末端工具供电 24V（夹爪动作前必须先调，否则夹爪不动）。"""
        ...

    def open_gripper(self) -> None: ...
    def close_gripper(self) -> None: ...
    def close(self) -> None: ...
