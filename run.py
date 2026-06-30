#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真机入口：单相机(眼在手上) 试管抓取/放置。
在【接了机械臂和相机的那台机器】上运行（参数都在 config.py）。

务必按这个顺序逐步放开，别一上来就 grasp：
    python run.py --mode detect                              # 看网格地图对不对，不动手
    python run.py --mode plan  --src left.A1 --dst right.C3  # 看预抓取/放置位姿 + 安全检查，不动手
    python run.py --mode move  --src left.A1 --dst right.C3  # 移到源试管上方 + 精对准，停住不抓
    python run.py --mode grasp --src left.A1 --dst right.C3  # 才真的抓-搬-放

--src / --dst 是孔位编号 board.row.col（left.A1 / right.C3），以 detect 打印的网格 ASCII 图为准。
"""
from __future__ import annotations

import argparse

import config
from core.calib_io import load_handeye
from core.grasp_planner import GraspPlanner
from core.solver import PoseSolver
from pipeline import TubeGrabberPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["detect", "plan", "move", "grasp"], default="detect")
    ap.add_argument("--src", help="源试管孔位（如 left.A1）")
    ap.add_argument("--dst", help="目标空槽孔位（如 right.C3）")
    ap.add_argument("--show", action="store_true", help="弹窗显示检测+网格标注图")
    ap.add_argument("--save", action="store_true", help="把标注图存到 outputs/")
    args = ap.parse_args()

    # 硬件 import 放函数内，保证没硬件时 import 本文件不报错（demo_offline 不受影响）
    from hardware.orbbec_rgbd import OrbbecRGBD
    from hardware.rm_robot import RMRobot
    from hardware.yolo_detector import YoloDetector

    # 眼在手上：读 T_cam2gripper
    T_cam2gripper, name = load_handeye(config.EYE_IN_HAND_RESULT)
    print(f"已加载手眼标定：{name}\n{T_cam2gripper}")

    camera = OrbbecRGBD(serial=config.HAND_CAMERA_SERIAL,
                        width=config.CAMERA_WIDTH, height=config.CAMERA_HEIGHT,
                        fps=config.CAMERA_FPS)
    detector = YoloDetector(config.YOLO_MODEL_PATH, conf=config.YOLO_CONF,
                            iou=config.YOLO_IOU, imgsz=config.YOLO_IMG_SIZE)
    robot = RMRobot(config.ROBOT_IP, config.ROBOT_PORT)   # eye_in_hand 结算/移动都要它

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
    try:
        rack = pipe.run(args.mode, src_sid=args.src, dst_sid=args.dst)

        if (args.show or args.save) and pipe.last_color is not None:
            import viz
            vis = viz.draw_rack(pipe.last_color, rack)
            if args.save:
                print(f"\n标注图已保存：{viz.save_image(vis, name=f'rack_{args.mode}.jpg')}")
            if args.show:
                viz.show_image(vis)
    finally:
        camera.stop()
        robot.close()


if __name__ == "__main__":
    main()
