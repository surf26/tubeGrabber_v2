"""
眼在手上(eye_in_hand)手眼标定的配置。

机械臂 IP / 相机序列号 / 分辨率 这些和主工程共用一份，直接从上级 config 取，
避免两处填写不一致。标定特有的参数（棋盘格、采样数、目录）写在这里。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 把工程根(tube_grabber_V2/)加入路径，复用主 config 的机械臂/相机参数
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import config as _main   # noqa: E402

# ---- 和主工程共用 ----
ROBOT_IP = _main.ROBOT_IP
ROBOT_PORT = _main.ROBOT_PORT
CAMERA_WIDTH = _main.CAMERA_WIDTH
CAMERA_HEIGHT = _main.CAMERA_HEIGHT
CAMERA_FPS = _main.CAMERA_FPS
CAMERA_SERIAL = _main.HAND_CAMERA_SERIAL   # 手上那台

# ---- 标定板：普通棋盘格 ----
# 内部角点数 (列, 行)：数黑白格交界的角点，不是格子数。10x7 个格子 → 内角点 (9,6)。
CHESSBOARD_SIZE = (9, 6)
# ★ 每个格子真实边长(米)。务必用卡尺实测打印件，别用设计尺寸。
SQUARE_SIZE_M = 0.0245

# ---- 采集 / 结果目录（都在 calib/ 下）----
DATA_DIR = str(Path(__file__).resolve().parent / "data")
RESULT_DIR = str(Path(__file__).resolve().parent / "results")
MIN_SAMPLES = 12   # 至少采这么多组再标定，建议 15~20+
