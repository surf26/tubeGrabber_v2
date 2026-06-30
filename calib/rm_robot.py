"""
睿尔曼 RM65-B 机械臂连接与位姿获取封装。

依赖官方 Robotic_Arm SDK (RM_API2): pip install Robotic_Arm
文档: https://develop.realman-robotics.com/en/robot/apipython/getStarted/

rm_get_current_arm_state() 返回的 state["pose"] 是 [x, y, z, rx, ry, rz]
（位置单位：米，姿态单位：弧度，欧拉角），基坐标系下的末端(flange)位姿。
旋转矩阵的换算直接调用 SDK 自带的 rm_algo_pos2matrix，避免自己猜欧拉角顺序出错。
"""
from __future__ import annotations

import numpy as np
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e


class RMRobot:
    def __init__(self, ip: str, port: int = 8080):
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.arm.rm_create_robot_arm(ip, port)
        if self.handle.id == -1:
            raise RuntimeError(f"连接机械臂失败: {ip}:{port}，请检查网线/IP/端口")
        print(f"机械臂连接成功，handle id = {self.handle.id}")

    def get_pose(self) -> list[float]:
        """末端(flange)在基座坐标系下的位姿 [x, y, z, rx, ry, rz]（米，弧度）"""
        status, state = self.arm.rm_get_current_arm_state()
        if status != 0:
            raise RuntimeError(f"读取机械臂状态失败，错误码: {status}")
        return state["pose"]

    def get_flange2base(self) -> np.ndarray:
        """末端(flange) -> 基座(base) 的 4x4 齐次变换矩阵"""
        pose = self.get_pose()
        matrix = self.arm.rm_algo_pos2matrix(pose)
        T = np.array(matrix.data, dtype=np.float64).reshape(4, 4)

        # 安全校验：rm_matrix_t.data 是否按行优先排布、且平移已写入第 4 列。
        # 如果 SDK 实际是列优先，或 pos2matrix 没写平移，这里会立刻报错，
        # 而不是悄悄用一个错误的矩阵把整套标定带偏。平移应等于 pose 的 xyz（单位都是米）。
        if not np.allclose(T[:3, 3], pose[:3], atol=1e-4) or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-4):
            raise RuntimeError(
                "rm_algo_pos2matrix 返回的矩阵布局和预期不符。\n"
                f"  pose(xyz) = {pose[:3]}\n"
                f"  T =\n{T}\n"
                "可能是行优先/列优先问题，请检查 SDK 版本或改用 T.reshape(4,4).T。"
            )
        return T

    def close(self):
        self.arm.rm_delete_robot_arm()
