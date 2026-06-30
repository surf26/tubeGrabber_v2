"""
读取手眼标定结果 / 内参。把"文件格式"这件事集中在这里，别散落到各处。

手眼结果文件就是 handeye_calib/calibrate.py 产出的 json：
    {"mode": "...", "matrix_name": "T_cam2base"/"T_cam2gripper", "matrix": [[...]] }
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.types import Intrinsics


def load_handeye(path: str | Path) -> tuple[np.ndarray, str]:
    """
    读手眼标定 4x4 矩阵。返回 (matrix, matrix_name)。
    matrix_name 为 'T_cam2base'(眼在手外) 或 'T_cam2gripper'(眼在手上)。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"找不到手眼标定结果 {path}\n"
            f"请先在 handeye_calib 里跑 calibrate.py 生成，或在 config.py 里改成正确路径。")
    data = json.loads(path.read_text())
    T = np.array(data["matrix"], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{path} 里的 matrix 不是 4x4：{T.shape}")
    name = data.get("matrix_name", "unknown")
    return T, name


def load_intrinsics(path: str | Path) -> Intrinsics:
    """
    从 json 读相机内参。支持两种格式：
      1) {"fx":..,"fy":..,"cx":..,"cy":..}
      2) {"camera_matrix": [[fx,0,cx],[0,fy,cy],[0,0,1]]}
    （capture.py 存的 dataset.json 里就有 camera_matrix，可复用。）
    """
    data = json.loads(Path(path).read_text())
    if "camera_matrix" in data:
        return Intrinsics.from_matrix(np.array(data["camera_matrix"]))
    return Intrinsics(fx=data["fx"], fy=data["fy"], cx=data["cx"], cy=data["cy"])
