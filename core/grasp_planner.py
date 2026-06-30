"""
抓取规划：把结算出的 base 坐标点 → 机械臂能执行的位姿，并做安全检查。

v1 策略（盖子基本水平朝上）：姿态固定竖直向下，只把算出的 xyz 填进去，
彻底绕开"旋转矩阵转欧拉角"的坑。三个关键位姿：
    approach 预抓取点：目标正上方 APPROACH_HEIGHT_M —— 粗定位就停这
    grasp    抓取点：目标点 + z 偏置（伺服/抓取阶段才用）
    retreat  抬起点：抓取点正上方（抓取阶段才用）

将来要做"沿表面法向接近"，在这里把 fixed orientation 换成由法向算的姿态即可，
上层 pipeline 不用改。
"""
from __future__ import annotations

import numpy as np

from core.types import GraspPlan, GraspTarget


class WorkspaceError(Exception):
    """位姿越界，拒绝执行。"""


class GraspPlanner:
    def __init__(self, orientation_rxryrz: tuple[float, float, float], *,
                 approach_height_m: float, grasp_z_offset_m: float,
                 retreat_height_m: float, tool_z_offset_m: float,
                 ws_bounds: tuple[float, float, float, float, float, float]):
        """
        orientation_rxryrz : 固定竖直向下姿态 [rx,ry,rz]（弧度），从示教器读
        ws_bounds          : (x_min,x_max,y_min,y_max,z_min,z_max) 安全范围（米）
        """
        self.rx, self.ry, self.rz = orientation_rxryrz
        self.approach_h = approach_height_m
        self.grasp_z_off = grasp_z_offset_m
        self.retreat_h = retreat_height_m
        self.tool_z_off = tool_z_offset_m
        (self.x_min, self.x_max, self.y_min, self.y_max,
         self.z_min, self.z_max) = ws_bounds

    def _pose(self, xyz: np.ndarray) -> list[float]:
        x, y, z = (float(v) for v in xyz)
        return [x, y, z, self.rx, self.ry, self.rz]

    def plan(self, target: GraspTarget, z_offset: float | None = None) -> GraspPlan:
        """生成 approach/grasp/retreat 三个位姿（不做安全检查，检查交给 check_pose）。
        z_offset 不传时用抓取偏置 self.grasp_z_off；放置时可传 place 偏置覆盖。"""
        x, y, z = target.base_xyz
        z_off = self.grasp_z_off if z_offset is None else z_offset

        # 抓取/放置点法兰 z = 目标z + 工具长度 + z偏置（负=往下扎）
        grasp_z = z + self.tool_z_off + z_off
        grasp = self._pose(np.array([x, y, grasp_z]))
        approach = self._pose(np.array([x, y, grasp_z + self.approach_h]))
        retreat = self._pose(np.array([x, y, grasp_z + self.retreat_h]))
        return GraspPlan(approach=approach, grasp=grasp, retreat=retreat)

    def check_pose(self, pose: list[float]) -> None:
        """安全检查：越界抛 WorkspaceError。move 前必调。"""
        x, y, z = pose[0], pose[1], pose[2]
        if not (self.x_min <= x <= self.x_max):
            raise WorkspaceError(f"x={x:.3f} 超出 [{self.x_min},{self.x_max}]")
        if not (self.y_min <= y <= self.y_max):
            raise WorkspaceError(f"y={y:.3f} 超出 [{self.y_min},{self.y_max}]")
        if not (self.z_min <= z <= self.z_max):
            raise WorkspaceError(f"z={z:.3f} 超出 [{self.z_min},{self.z_max}]（Z下限是地板保护）")
