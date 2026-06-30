"""普通棋盘格检测 + solvePnP，求标定板在相机坐标系下的位姿 (target2cam)。"""
from __future__ import annotations

import cv2
import numpy as np

import config

_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def _object_points(pattern_size, square_size) -> np.ndarray:
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), dtype=np.float64)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    return objp


_OBJECT_POINTS = _object_points(config.CHESSBOARD_SIZE, config.SQUARE_SIZE_M)


def find_target2cam(image: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
    """
    在图像中找棋盘格，返回 (T_target2cam, corners, vis_image)。
    找不到时返回 (None, None, vis_image)。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    found, corners = cv2.findChessboardCorners(
        gray, config.CHESSBOARD_SIZE,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK,
    )

    vis = image.copy()
    if not found:
        return None, None, vis

    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _CRITERIA)
    cv2.drawChessboardCorners(vis, config.CHESSBOARD_SIZE, corners, found)

    ok, rvec, tvec = cv2.solvePnP(
        _OBJECT_POINTS, corners, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None, corners, vis

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T, corners, vis
