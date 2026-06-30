#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线自测：不接硬件，用 Mock 把
  全局看 -> 双类检测 -> 结算 -> 建网格 -> 取任务 -> 粗定位 -> look-and-move 微调 -> 抓 -> 搬 -> 放
整条逻辑在 Mac 上跑一遍，验证数学/网格/流程。只依赖 numpy。

运行（在 tube_grabber_V2/ 目录下）：
    python demo_offline.py
"""
from __future__ import annotations

import numpy as np

import config
from core.grasp_planner import GraspPlanner
from core.solver import PoseSolver
from core.types import Detection, Intrinsics
from mocks.mocks import MockCamera, MockDetector, MockRobot, mock_rack_detections
from pipeline import TubeGrabberPipeline


def sanity_check_math():
    """eye_in_hand 可手算例子：相机=末端(T_cam2gripper=I)，末端位姿把相机系点平移/翻转到 base。
       中心像素、深度 0.5m -> 相机系(0,0,0.5)；经 flange2base 应得 base=(0,0.55,0.10)。"""
    intr = Intrinsics(fx=900.0, fy=900.0, cx=640.0, cy=360.0)
    T_cam2gripper = np.eye(4)
    solver = PoseSolver(intr, T_cam2gripper, "eye_in_hand", depth_scale=0.001,
                        depth_patch=3, depth_min_m=0.1, depth_max_m=2.0)
    f2b = np.array([[1, 0, 0, 0.0], [0, -1, 0, 0.55], [0, 0, -1, 0.60], [0, 0, 0, 1.0]])
    depth = np.full((720, 1280), 500, dtype=np.uint16)  # 0.5m
    t = solver.solve_one(Detection(pixel=(640, 360), bbox=(0, 0, 0, 0), conf=1.0), depth, f2b)
    expect = np.array([0.0, 0.55, 0.10])
    assert t is not None and np.allclose(t.base_xyz, expect, atol=1e-6), (t.base_xyz, expect)
    print(f"[数学自测] 通过 ✅  base={t.base_xyz} == 期望 {expect}")


def main():
    print("=" * 60)
    sanity_check_math()
    print("=" * 60)

    dets = mock_rack_detections(tube_cls=config.TUBE_CLS, empty_cls=config.EMPTY_CLS)
    camera = MockCamera()
    detector = MockDetector(dets)
    robot = MockRobot()                 # 默认竖直向下全局视野位姿
    T_cam2gripper = np.eye(4)           # demo 里相机=末端，简化数学

    solver = PoseSolver(
        camera.intrinsics, T_cam2gripper, "eye_in_hand",
        depth_scale=camera.depth_scale, depth_patch=config.DEPTH_PATCH,
        depth_min_m=config.DEPTH_VALID_MIN_M, depth_max_m=config.DEPTH_VALID_MAX_M)

    planner = GraspPlanner(
        config.GRASP_ORIENTATION_RXRYRZ,
        approach_height_m=config.APPROACH_HEIGHT_M,
        grasp_z_offset_m=config.GRASP_Z_OFFSET_M,
        retreat_height_m=config.RETREAT_HEIGHT_M,
        tool_z_offset_m=config.TOOL_Z_OFFSET,
        ws_bounds=(config.WORKSPACE_X_MIN, config.WORKSPACE_X_MAX,
                   config.WORKSPACE_Y_MIN, config.WORKSPACE_Y_MAX,
                   config.WORKSPACE_Z_MIN, config.WORKSPACE_Z_MAX))

    pipe = TubeGrabberPipeline(
        camera, detector, solver, planner, robot,
        global_view_pose=config.GLOBAL_VIEW_POSE, speed=config.ROBOT_SPEED,
        tube_cls=config.TUBE_CLS, empty_cls=config.EMPTY_CLS,
        rack_params=dict(split_x=config.BOARD_SPLIT_X,
                         n_rows=config.RACK_ROWS, n_cols=config.RACK_COLS,
                         left_rows=config.LEFT_ROWS, right_rows=config.RIGHT_ROWS),
        align_params=dict(observe_height_m=config.ALIGN_OBSERVE_HEIGHT_M,
                          converge_m=config.ALIGN_CONVERGE_M, max_iters=config.ALIGN_MAX_ITERS),
        place_z_offset_m=config.PLACE_Z_OFFSET_M,
        grasp_geom=dict(cap_d=config.CAP_DIAMETER_M, gap=config.CAP_GAP_M,
                        finger_t=config.FINGER_THICKNESS_M,
                        gripper_max_open=config.GRIPPER_MAX_OPEN_M,
                        margin=config.GRASP_CLEARANCE_MARGIN_M))

    # 任务：把 left.C2 的试管搬到 right.A1 的空槽（见 mocks 双板布局：空槽在 left.A1 / right.A1）
    src_sid, dst_sid = "left.C2", "right.A1"
    for mode in ("detect", "plan", "move", "grasp"):
        print("\n" + "#" * 60)
        print(f"# 模式：{mode}   任务 src={src_sid} -> dst={dst_sid}")
        print("#" * 60)
        pipe.run(mode, src_sid=src_sid, dst_sid=dst_sid)


if __name__ == "__main__":
    main()
