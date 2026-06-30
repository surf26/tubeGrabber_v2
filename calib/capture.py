#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
眼在手上(eye_in_hand)手眼标定 —— 数据采集。

物理摆法：相机装在末端随手臂动，棋盘格【固定不动】（平放桌上）。
操作：
  - 用示教器把臂移到能看清棋盘格的姿态；
  - 窗口里按 c 采集一组（图像 + 当前机械臂法兰位姿 + 棋盘格在相机系的位姿）；
  - 不断换【不同朝向】(不要只平移！纯平移约束不了旋转) 重复，采够 MIN_SAMPLES 组以上；
  - 按 q 结束，自动存 data/eye_in_hand/dataset.json。

用法： python capture.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2

import config
from chessboard import find_target2cam
from orbbec_camera import OrbbecCamera
from rm_robot import RMRobot

MODE = "eye_in_hand"


def main():
    out_dir = Path(config.DATA_DIR) / MODE
    out_dir.mkdir(parents=True, exist_ok=True)

    robot = RMRobot(config.ROBOT_IP, config.ROBOT_PORT)
    cam = OrbbecCamera()

    samples = []
    idx = 0
    print(f"\n[{MODE}] 窗口里按 c 采集，按 q 结束。棋盘格要固定不动！\n")

    try:
        while True:
            frame = cam.get_color_frame()
            T_t2c, _, vis = find_target2cam(frame, cam.camera_matrix, cam.dist_coeffs)

            ok = T_t2c is not None
            cv2.putText(vis, f"chessboard: {'FOUND' if ok else 'NOT FOUND'}  samples: {len(samples)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if ok else (0, 0, 255), 2)
            cv2.imshow("capture - c:capture  q:quit", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                if not ok:
                    print("  没找到棋盘格，跳过")
                    continue
                T_f2b = robot.get_flange2base()
                img_name = f"sample_{idx:03d}.png"
                cv2.imwrite(str(out_dir / img_name), frame)
                samples.append({
                    "image": img_name,
                    "flange2base": T_f2b.tolist(),
                    "target2cam": T_t2c.tolist(),
                })
                idx += 1
                print(f"  已采集第 {idx} 组")
    finally:
        cam.stop()
        robot.close()
        cv2.destroyAllWindows()

    if not samples:
        print("没采到数据，退出。")
        return

    meta = {
        "mode": MODE,
        "camera_matrix": cam.camera_matrix.tolist(),
        "dist_coeffs": cam.dist_coeffs.tolist(),
        "chessboard_size": list(config.CHESSBOARD_SIZE),
        "square_size_m": config.SQUARE_SIZE_M,
        "samples": samples,
    }
    out_path = out_dir / "dataset.json"
    out_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n共 {len(samples)} 组，已存 {out_path}")
    if len(samples) < config.MIN_SAMPLES:
        print(f"建议至少 {config.MIN_SAMPLES} 组再标定，当前偏少。")


if __name__ == "__main__":
    main()
