"""
位姿结算核心：检测结果 + 深度图 + 手眼标定 → 基座坐标系下的 3D 目标。

纯逻辑：输入 ndarray / dataclass，输出 dataclass，不碰任何相机/机械臂 SDK，
所以能用 Mock 数据在 Mac 上完整测试。

本工程是【眼在手上】(eye_in_hand)：
    p_base = T_flange2base @ T_cam2gripper @ p_cam
其中 T_flange2base 随机械臂运动而变，必须在每次拍照时实时读入。
（保留 eye_to_hand 分支只为单元测试方便手算验证，本工程实际不用。）
"""
from __future__ import annotations

import numpy as np

from core import geometry
from core.types import Detection, GraspTarget, Intrinsics


class PoseSolver:
    def __init__(self, intrinsics: Intrinsics, handeye: np.ndarray, mode: str, *,
                 depth_scale: float, depth_patch: int,
                 depth_min_m: float, depth_max_m: float,
                 source: str = "hand_cam"):
        """
        intrinsics : 深度对齐到的彩色流内参
        handeye    : eye_in_hand 传 T_cam2gripper；eye_to_hand 传 T_cam2base
        mode       : "eye_in_hand"（本工程）或 "eye_to_hand"（仅测试）
        """
        if mode not in ("eye_to_hand", "eye_in_hand"):
            raise ValueError(f"mode 只能是 eye_to_hand / eye_in_hand，收到 {mode}")
        self.intr = intrinsics
        self.handeye = np.asarray(handeye, dtype=np.float64)
        self.mode = mode
        self.depth_scale = depth_scale
        self.depth_patch = depth_patch
        self.depth_min = depth_min_m
        self.depth_max = depth_max_m
        self.source = source

    def solve_one(self, det: Detection, depth_img: np.ndarray,
                  flange2base: np.ndarray | None = None) -> GraspTarget | None:
        """单个检测框 → base 系 GraspTarget。深度无效(空洞)返回 None，由上层跳过。
        eye_in_hand 模式必须传 flange2base（拍这帧时的末端位姿）。"""
        u, v = det.pixel
        z = geometry.sample_depth(depth_img, u, v,
                                  patch=self.depth_patch, scale=self.depth_scale,
                                  z_min=self.depth_min, z_max=self.depth_max)
        if z is None:
            return None
        p_cam = geometry.deproject(u, v, z, self.intr)
        base_xyz = self._cam_to_base(p_cam, flange2base)
        return GraspTarget(base_xyz=base_xyz, cam_xyz=p_cam, pixel=(u, v),
                           depth_m=z, conf=det.conf, cls=det.cls,
                           source=self.source, bbox=det.bbox)

    def solve(self, detections: list[Detection], depth_img: np.ndarray,
              flange2base: np.ndarray | None = None) -> list[GraspTarget]:
        out = []
        for d in detections:
            t = self.solve_one(d, depth_img, flange2base)
            if t is not None:
                out.append(t)
        return out

    def _cam_to_base(self, p_cam: np.ndarray, flange2base: np.ndarray | None) -> np.ndarray:
        if self.mode == "eye_to_hand":
            return geometry.transform_point(self.handeye, p_cam)          # T_cam2base @ p_cam
        if flange2base is None:
            raise ValueError("eye_in_hand 模式必须提供 flange2base（机械臂当前末端位姿）")
        p_gripper = geometry.transform_point(self.handeye, p_cam)         # T_cam2gripper @ p_cam
        return geometry.transform_point(flange2base, p_gripper)           # T_flange2base @ p_gripper
