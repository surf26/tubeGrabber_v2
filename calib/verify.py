#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
眼在手上(eye_in_hand)标定结果 —— 实时验证（不依赖采集数据集，直连相机+机械臂）。

棋盘格固定不动，机械臂随便挪到不同姿态，实时打印棋盘格原点在【机械臂基座坐标系】下
的位置 (x,y,z mm)。标定准确的话，挪动手臂时这个值应该几乎不变。

用法： python verify.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import config
from chessboard import find_target2cam
from orbbec_camera import OrbbecCamera
from rm_robot import RMRobot

MODE = "eye_in_hand"


def main():
    result_path = Path(config.RESULT_DIR) / f"{MODE}_handeye.json"
    if not result_path.exists():
        raise FileNotFoundError(f"找不到 {result_path}，请先跑 calibrate.py")
    T_cam2gripper = np.array(json.loads(result_path.read_text())["matrix"])

    robot = RMRobot(config.ROBOT_IP, config.ROBOT_PORT)
    cam = OrbbecCamera()

    print("按 q 退出。挪动机械臂，观察打印的 xyz 是否保持稳定。\n")
    try:
        while True:
            frame = cam.get_color_frame()
            T_t2c, _, vis = find_target2cam(frame, cam.camera_matrix, cam.dist_coeffs)
            if T_t2c is not None:
                T_f2b = robot.get_flange2base()
                xyz_mm = (T_f2b @ T_cam2gripper @ T_t2c)[:3, 3] * 1000  # target in base
                text = f"target in base: x={xyz_mm[0]:.1f} y={xyz_mm[1]:.1f} z={xyz_mm[2]:.1f} mm"
                print(text)
                cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(vis, "chessboard not found", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("verify - q:quit", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.stop()
        robot.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
