"""
占位实现：假相机 / 假双类检测器 / 假机械臂。只依赖 numpy，可在 Mac 上跑。
和 hardware/ 里的真实现满足同一套 interfaces 协议，可无缝替换。

MockDetector 造一个【网格试管架】：3 行 × 4 列，大部分是试管(tube)，少数空槽(empty)，
配合 MockCamera 的水平面深度，让你不接硬件就能验证：
反投影 + eye_in_hand 坐标变换 + 网格建图 + 行列查找 + look-and-move + 抓放流程。
"""
from __future__ import annotations

import numpy as np

from core.types import Detection, Intrinsics

# 双板试管架在像素上的布局：左板 3 列 + 右板 3 列（split_x=640 分界），各 4 行。
# 配合 MockCamera 的内参/平面深度，会还原成 base 系的规则网格。
_LEFT_US = [200, 320, 440]      # 左板 3 列（u<640）
_RIGHT_US = [840, 960, 1080]    # 右板 3 列（u>=640）
_VS = [180, 300, 420, 540]      # 4 行
# 空槽位置（其余都是试管）：左板一个、右板一个
_EMPTY_PX = {(440, 180), (840, 540)}


def mock_rack_detections(tube_cls: int = 1, empty_cls: int = 0):
    """返回 [(u, v, cls), ...]，给 MockDetector / MockCamera 共用（左右各 3 列 × 4 行 = 24）。"""
    out = []
    for v in _VS:
        for u in _LEFT_US + _RIGHT_US:
            cls = empty_cls if (u, v) in _EMPTY_PX else tube_cls
            out.append((float(u), float(v), cls))
    return out


class MockCamera:
    """假 RGBD 相机：固定内参 + 一片水平面深度图。"""

    def __init__(self, width=1280, height=720, plane_z_m=0.5, depth_scale=0.001):
        self.width, self.height = width, height
        self._intr = Intrinsics(fx=900.0, fy=900.0, cx=width / 2, cy=height / 2)
        self._depth_scale = depth_scale
        raw = int(round(plane_z_m / depth_scale))
        self._depth = np.full((height, width), raw, dtype=np.uint16)
        self._color = np.full((height, width, 3), 80, dtype=np.uint8)

    @property
    def intrinsics(self) -> Intrinsics:
        return self._intr

    @property
    def depth_scale(self) -> float:
        return self._depth_scale

    def get_frames(self):
        return self._color.copy(), self._depth.copy()

    def stop(self):
        pass


class MockDetector:
    """假双类检测器：返回设定好的网格(试管+空槽)。"""

    def __init__(self, detections, conf=0.9):
        self._dets = detections      # [(u,v,cls), ...]
        self.conf = conf

    def detect(self, color_bgr):
        return [Detection(pixel=(u, v), bbox=(u - 18, v - 18, u + 18, v + 18),
                          conf=self.conf, cls=cls)
                for (u, v, cls) in self._dets]


class MockRobot:
    """假机械臂：记录被要求移动到的位姿并打印，不连真硬件。
       默认末端位姿是一个"竖直向下、离桌面约 0.5m"的全局视野姿态，
       让结算出的网格落在 config 工作区内、安全检查能过。"""

    def __init__(self, flange2base: np.ndarray | None = None):
        if flange2base is None:
            # 末端竖直向下：cam x->base x, cam y->base -y, cam z(前)->base -z；平移把桌面落在 base 系工作区内
            # （t_y=0.2,t_z=0.5 让结算出的网格落在 config 的临时工作区 y∈[0,0.45]/z 内，demo 安全检查能过）
            flange2base = np.array([
                [1, 0, 0, 0.00],
                [0, -1, 0, 0.20],
                [0, 0, -1, 0.50],
                [0, 0, 0, 1.0],
            ], dtype=np.float64)
        self._f2b = np.asarray(flange2base, dtype=np.float64)
        self.last_pose = None

    def get_flange2base(self):
        return self._f2b.copy()

    def get_pose(self):
        T = self._f2b
        return [T[0, 3], T[1, 3], T[2, 3], 3.1416, 0.0, 0.0]

    def move_to_pose(self, pose, speed, block=True):
        self.last_pose = list(pose)
        print(f"    [MockRobot] move_to_pose(speed={speed}) -> {[round(p, 4) for p in pose]}")

    def enable_gripper_power(self):
        print("    [MockRobot] enable_gripper_power(24V)")

    def open_gripper(self):
        print("    [MockRobot] open_gripper()")

    def close_gripper(self):
        print("    [MockRobot] close_gripper()")

    def close(self):
        print("    [MockRobot] close()")
