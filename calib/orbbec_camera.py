"""
奥比中光 Orbbec Gemini 336 彩色相机封装：取图 + 读取出厂内参。

用的是 Orbbec 官方 SDK：pip install pyorbbecsdk
文档：https://orbbec.github.io/pyorbbecsdk/

对外接口和原来的 RealSense 版保持一致：
    cam.get_color_frame() -> BGR ndarray
    cam.camera_matrix     -> 3x3
    cam.dist_coeffs       -> (5,)  OpenCV 顺序 [k1,k2,p1,p2,k3]
    cam.stop()
所以 capture.py / verify.py 只需把 import 换成本模块即可。
"""
from __future__ import annotations

import cv2
import numpy as np
from pyorbbecsdk import Pipeline, Config, Context, OBSensorType, OBFormat

import config


def _pick_device(serial: str):
    """根据序列号挑相机；serial 为空时用唯一一台（接了多台会报错提示）。"""
    dev_list = Context().query_devices()
    count = dev_list.get_count()
    if count == 0:
        raise RuntimeError("没有检测到 Orbbec 相机，检查 USB 连接")

    found = {}
    for i in range(count):
        d = dev_list.get_device_by_index(i)
        found[d.get_device_info().get_serial_number()] = d
    print(f"检测到 {count} 台 Orbbec 相机，序列号: {list(found.keys())}")

    if serial:
        if serial not in found:
            raise RuntimeError(f"未找到序列号 {serial} 的相机，当前接的是 {list(found.keys())}")
        return found[serial]

    if count > 1:
        raise RuntimeError(
            f"接了 {count} 台相机但没指定用哪台。请把要用的序列号填进 config.py 的 "
            f"CAMERA_SERIAL（当前检测到: {list(found.keys())}）"
        )
    return next(iter(found.values()))


def _frame_to_bgr(color_frame) -> np.ndarray:
    """把 Orbbec 彩色帧转成 OpenCV 的 BGR ndarray，兼容常见的几种像素格式。"""
    fmt = color_frame.get_format()
    w, h = color_frame.get_width(), color_frame.get_height()
    data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)

    if fmt == OBFormat.RGB:
        return cv2.cvtColor(data.reshape((h, w, 3)), cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return data.reshape((h, w, 3))
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == OBFormat.YUYV:
        return cv2.cvtColor(data.reshape((h, w, 2)), cv2.COLOR_YUV2BGR_YUYV)
    if fmt == OBFormat.UYVY:
        return cv2.cvtColor(data.reshape((h, w, 2)), cv2.COLOR_YUV2BGR_UYVY)
    raise RuntimeError(f"暂不支持的彩色格式 {fmt}，请在 _frame_to_bgr 里补一个分支")


class OrbbecCamera:
    def __init__(self, width=config.CAMERA_WIDTH, height=config.CAMERA_HEIGHT,
                 fps=config.CAMERA_FPS, serial=config.CAMERA_SERIAL):
        device = _pick_device(serial)
        self.pipeline = Pipeline(device)
        ob_config = Config()

        profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        # 优先按 config 里的分辨率/帧率取 RGB；取不到就退回设备默认的彩色流。
        try:
            color_profile = profile_list.get_video_stream_profile(width, height, OBFormat.RGB, fps)
        except Exception:
            color_profile = profile_list.get_default_video_stream_profile()
            print("警告：取不到指定的 RGB 流参数，已退回设备默认彩色流。")
        ob_config.enable_stream(color_profile)

        self.pipeline.start(ob_config)

        # 出厂内参直接从彩色流 profile 拿（比 device.get_calibration_camera_param 更稳，
        # 后者在部分 V2 SDK 版本上有返回全 0 的已知问题）。
        intr = color_profile.get_intrinsic()
        self.camera_matrix = np.array([
            [intr.fx, 0, intr.cx],
            [0, intr.fy, intr.cy],
            [0, 0, 1],
        ], dtype=np.float64)

        dist = color_profile.get_distortion()
        # OpenCV 顺序：[k1, k2, p1, p2, k3]
        self.dist_coeffs = np.array([dist.k1, dist.k2, dist.p1, dist.p2, dist.k3], dtype=np.float64)

        print("Orbbec Gemini 336 彩色相机出厂内参：")
        print(f"  camera_matrix=\n{self.camera_matrix}")
        print(f"  dist_coeffs={self.dist_coeffs}")

        # 丢弃开机前几帧，让自动曝光稳定
        for _ in range(10):
            self.pipeline.wait_for_frames(1000)

    def get_color_frame(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames(1000)
        if frames is None:
            raise RuntimeError("取帧超时，未拿到画面")
        color_frame = frames.get_color_frame()
        if color_frame is None:
            raise RuntimeError("这一帧没有彩色数据")
        return _frame_to_bgr(color_frame)

    def stop(self):
        self.pipeline.stop()
