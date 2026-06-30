#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
眼在手上(eye_in_hand)手眼标定 —— 求解。

读 capture.py 采集的 dataset.json，跑 cv2.calibrateHandEye 求 T_cam2gripper
（相机坐标系 -> 末端/flange 坐标系 的变换）。标定板固定不动，直接用 (flange2base, target2cam)。

输出 results/eye_in_hand_handeye.json，主工程 config.EYE_IN_HAND_RESULT 指向它。

用法： python calibrate.py            # 默认 tsai
       python calibrate.py --method park
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import config

MODE = "eye_in_hand"
_METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def rotation_angle_deg(R: np.ndarray) -> float:
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))


def consistency_check(samples, T_cam2gripper: np.ndarray):
    """标定板固定在 base 下，target2base = flange2base @ T_cam2gripper @ target2cam
       在所有样本里应几乎恒定。分散度越小标定越好。"""
    consts = [np.array(s["flange2base"]) @ T_cam2gripper @ np.array(s["target2cam"])
              for s in samples]
    trans = np.array([T[:3, 3] for T in consts])
    std_mm = trans.std(axis=0) * 1000
    mean_R = consts[0][:3, :3]
    angs = [rotation_angle_deg(mean_R.T @ T[:3, :3]) for T in consts]
    print("\n一致性检查（越小越好）：")
    print(f"  平移标准差(mm): x={std_mm[0]:.2f} y={std_mm[1]:.2f} z={std_mm[2]:.2f}")
    print(f"  旋转偏差(deg): mean={np.mean(angs):.3f} max={np.max(angs):.3f}")
    print("  经验：平移标准差<5mm、旋转<1° 算不错。偏大多半是棋盘格尺寸不准/姿态变化太小/没贴平。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=list(_METHODS.keys()), default="tsai")
    args = ap.parse_args()

    path = Path(config.DATA_DIR) / MODE / "dataset.json"
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}，请先跑 capture.py 采集")
    data = json.loads(path.read_text())
    samples = data["samples"]
    if len(samples) < 3:
        raise RuntimeError("样本太少，至少 3 组，建议 15+")

    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    for s in samples:
        T_f2b = np.array(s["flange2base"])
        T_t2c = np.array(s["target2cam"])
        R_g2b.append(T_f2b[:3, :3]); t_g2b.append(T_f2b[:3, 3])
        R_t2c.append(T_t2c[:3, :3]); t_t2c.append(T_t2c[:3, 3])

    R_x, t_x = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=_METHODS[args.method])
    T_x = np.eye(4); T_x[:3, :3] = R_x; T_x[:3, 3] = t_x.reshape(3)

    print(f"\n[{MODE}] T_cam2gripper (相机 -> 末端flange)：\n{T_x}")
    consistency_check(samples, T_x)

    out_dir = Path(config.RESULT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{MODE}_handeye.json"
    out_path.write_text(json.dumps({
        "mode": MODE,
        "method": args.method,
        "num_samples": len(samples),
        "matrix_name": "T_cam2gripper",
        "matrix": T_x.tolist(),
        "camera_matrix": data.get("camera_matrix"),
        "dist_coeffs": data.get("dist_coeffs"),
    }, indent=2, ensure_ascii=False))
    print(f"\n结果已存 {out_path}")


if __name__ == "__main__":
    main()
