"""
真实 RM65-B 机械臂：连接 + 读位姿 + 笛卡尔运动 + 夹爪。
实现 interfaces.Robot 协议。依赖官方 Robotic_Arm SDK (RM_API2)。

下面的 API 签名已对照睿尔曼官方文档核实（2026-06，RM_API2 Python）：
  - rm_movej_p / rm_movel / rm_movej:
    https://develop.realman-robotics.com/en/robot/apipython/classes/movePlan/
  - 夹爪 rm_set_gripper_*:
    https://develop.realman-robotics.com/en/robot/apipython/classes/gripperControl/
  - 位姿格式确认为欧拉角 [x,y,z,rx,ry,rz]（米/弧度），不是四元数。
  - 视觉垂直抓取参考（和本工程场景几乎一致，强烈建议看）：
    https://develop.realman-robotics.com/en/AI/developerGuide/verticalGrab/

读位姿部分沿用 handeye_calib/rm_robot.py 的稳妥写法（rm_algo_pos2matrix + 校验布局）。
"""
from __future__ import annotations

import numpy as np
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e


class RMRobot:
    def __init__(self, ip: str, port: int = 8080):
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.arm.rm_create_robot_arm(ip, port)
        if self.handle.id == -1:
            raise RuntimeError(f"连接机械臂失败: {ip}:{port}")
        print(f"机械臂连接成功，handle id = {self.handle.id}")

    # ---- 读状态 ----
    def get_pose(self) -> list[float]:
        status, state = self.arm.rm_get_current_arm_state()
        if status != 0:
            raise RuntimeError(f"读取机械臂状态失败，错误码: {status}")
        return state["pose"]

    def get_flange2base(self) -> np.ndarray:
        pose = self.get_pose()
        matrix = self.arm.rm_algo_pos2matrix(pose)
        T = np.array(matrix.data, dtype=np.float64).reshape(4, 4)
        if not np.allclose(T[:3, 3], pose[:3], atol=1e-4) or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-4):
            raise RuntimeError(f"pos2matrix 矩阵布局异常：pose={pose[:3]} T=\n{T}")
        return T

    # ---- 运动 ----
    def move_to_pose(self, pose: list[float], speed: float, block: bool = True) -> None:
        """
        笛卡尔位姿运动到 [x,y,z,rx,ry,rz]（米/弧度，基座系）。
        官方签名: rm_movej_p(pose, v, r, connect, block) -> int
          v=速度比例(1~100), r=交融半径(0~100), connect=轨迹连接(0立即执行), block=阻塞(1阻塞)
        movej_p 是给末端位姿、由 SDK 做关节空间规划，比 movel 更不易碰奇异点，适合点到点。
        """
        ret = self.arm.rm_movej_p(pose, int(speed), 0, 0, 1 if block else 0)
        if ret != 0:
            raise RuntimeError(f"rm_movej_p 失败，错误码: {ret}，目标 {pose}")

    # ---- 夹爪（抓取阶段才用，签名已核实）----
    def enable_gripper_power(self) -> None:
        """
        ★ 必坑提醒：夹爪要先把末端工具供电设成 24V 才会动（官方试管抓取 demo 里就是这步）。
        rm_set_tool_voltage(voltage_type)：常见取值 0=0V,1=5V,2=12V,3=24V（按你的 SDK/夹爪核对）。
        做抓取前调用一次即可（粗定位阶段用不到）。
        """
        ret = self.arm.rm_set_tool_voltage(3)
        if ret != 0:
            raise RuntimeError(f"rm_set_tool_voltage 失败，错误码: {ret}（夹爪可能无法供电）")

    def open_gripper(self) -> None:
        # rm_set_gripper_release(speed, block, timeout) ; speed 1~1000
        self.arm.rm_set_gripper_release(500, True, 5)

    def close_gripper(self) -> None:
        # rm_set_gripper_pick_on(speed, force, block, timeout) ; speed 1~1000, force 50~1000
        self.arm.rm_set_gripper_pick_on(500, 200, True, 5)

    def close(self) -> None:
        self.arm.rm_delete_robot_arm()
