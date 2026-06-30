"""
look-and-move 精对准 —— 阶段B"微调"，替代复杂的 IBVS 视觉伺服。

为什么这样做（写给第一次做的人）：
    相机垂直向下、抓取姿态固定不变，所以"对准"本质就是把 xyz 算准。
    高处全局图算出的坐标有几毫米~一厘米误差（远、畸变大、深度糙）。
    办法很直接：移到目标正上方一个近距离高度 → 重新拍一张 → 重新结算 → 得到更准的坐标，
    必要时再挪一点点重拍，直到两次结果几乎不变（收敛）。
    这就是 "看一眼→挪过去→再看一眼"，数学上等价于伺服，但简单、可解释、好调。

关联策略：近距离重检测会得到该类的多个目标，取"结算出的 base 坐标离当前估计最近"的那个，
          稳健且不依赖目标恰好落在画面正中。
"""
from __future__ import annotations

import numpy as np

import interfaces
from core.grasp_planner import GraspPlanner
from core.solver import PoseSolver
from core.types import GraspTarget


class LookAndMoveAligner:
    def __init__(self, camera: interfaces.RGBDCamera, detector: interfaces.Detector,
                 solver: PoseSolver, planner: GraspPlanner, robot: interfaces.Robot, *,
                 observe_height_m: float, converge_m: float, max_iters: int,
                 speed: float, target_cls: int):
        self.camera = camera
        self.detector = detector
        self.solver = solver
        self.planner = planner
        self.robot = robot
        self.observe_h = observe_height_m
        self.converge = converge_m
        self.max_iters = max_iters
        self.speed = speed
        self.target_cls = target_cls

    def _observe_pose(self, xyz: np.ndarray) -> list[float]:
        """目标正上方 observe_height 处的法兰位姿（竖直向下姿态）。"""
        x, y, z = (float(v) for v in xyz)
        flange_z = z + self.planner.tool_z_off + self.observe_h
        return [x, y, flange_z, self.planner.rx, self.planner.ry, self.planner.rz]

    def refine(self, target: GraspTarget) -> GraspTarget:
        """对 target 做 look-and-move 精修，返回坐标更准的 GraspTarget。"""
        cur = target
        for it in range(self.max_iters):
            # 1) 移到当前估计的正上方近距离观察点
            obs = self._observe_pose(cur.base_xyz)
            self.planner.check_pose(obs)               # 越界直接拒绝
            self.robot.move_to_pose(obs, speed=self.speed, block=True)

            # 2) 重新拍 + 实时读位姿 + 重检测(只留目标类)
            color, depth = self.camera.get_frames()
            flange2base = self.robot.get_flange2base()
            dets = [d for d in self.detector.detect(color) if d.cls == self.target_cls]
            cands = self.solver.solve(dets, depth, flange2base)
            if not cands:
                print(f"    [align] 第{it+1}次重检测没找到目标类，保持上一次估计")
                break

            # 3) 关联：取 base 坐标离当前估计最近的那个
            new = min(cands, key=lambda t: float(np.linalg.norm(t.base_xyz - cur.base_xyz)))
            delta = float(np.linalg.norm(new.base_xyz - cur.base_xyz))
            print(f"    [align] 第{it+1}次：Δ={delta*1000:.2f}mm  "
                  f"base={(new.base_xyz*1000).round(1).tolist()}mm")
            cur = new
            if delta < self.converge:
                print(f"    [align] 已收敛(<{self.converge*1000:.1f}mm)")
                break
        return cur
