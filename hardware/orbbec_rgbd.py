"""
真实 Orbbec Gemini 336 RGBD 相机：彩色 + 深度 + 深度对齐到彩色(D2C)。
实现 interfaces.RGBDCamera 协议。依赖 pyorbbecsdk2（import 仍写 pyorbbecsdk）。

★ 关键点：必须开启 D2C 对齐，否则彩色像素 (u,v) 对不上深度图同一点，结算全错。
内参取的是【对齐后彩色流】的内参，深度用对齐后的深度帧。

注意：不同 SDK 版本对齐 API 略有差异，下面用 align_mode=HW/SW + AlignFilter 两种兜底，
若你的版本报错，按官方 examples 里的 depth-to-color 对齐示例改这一处即可（其余不用动）。
"""
from __future__ import annotations

import numpy as np
from pyorbbecsdk import (AlignFilter, Config, Context, OBFormat,
                         OBSensorType, OBStreamType, Pipeline)

from core.types import Intrinsics


def _pick_device(serial: str):
    dev_list = Context().query_devices()
    count = dev_list.get_count()
    if count == 0:
        raise RuntimeError("没有检测到 Orbbec 相机，检查 USB3.0 连接")
    found = {}
    for i in range(count):
        d = dev_list.get_device_by_index(i)
        found[d.get_device_info().get_serial_number()] = d
    print(f"检测到 {count} 台 Orbbec：{list(found.keys())}")
    if serial:
        if serial not in found:
            raise RuntimeError(f"未找到序列号 {serial}，当前接的是 {list(found.keys())}")
        return found[serial]
    if count > 1:
        raise RuntimeError(f"接了多台相机，请在 config 里指定序列号：{list(found.keys())}")
    return next(iter(found.values()))


def _color_to_bgr(frame) -> np.ndarray:
    import cv2
    fmt = frame.get_format()
    w, h = frame.get_width(), frame.get_height()
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)
    if fmt == OBFormat.RGB:
        return cv2.cvtColor(data.reshape((h, w, 3)), cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return data.reshape((h, w, 3))
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == OBFormat.YUYV:
        return cv2.cvtColor(data.reshape((h, w, 2)), cv2.COLOR_YUV2BGR_YUYV)
    raise RuntimeError(f"暂不支持的彩色格式 {fmt}")


class OrbbecRGBD:
    def __init__(self, serial: str = "", width=1280, height=720, fps=30):
        device = _pick_device(serial)
        self.pipeline = Pipeline(device)
        cfg = Config()

        # 彩色流
        color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile = color_profiles.get_video_stream_profile(width, height, OBFormat.RGB, fps)
        except Exception:
            color_profile = color_profiles.get_default_video_stream_profile()
            print("警告：取不到指定彩色流参数，已退回默认。")
        cfg.enable_stream(color_profile)

        # 深度流
        depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_default_video_stream_profile()
        cfg.enable_stream(depth_profile)

        # D2C 对齐：用官方示例验证过的 AlignFilter 软件对齐（对齐到彩色流），跨版本最稳。
        # 社区反馈硬件对齐 cfg.set_align_mode(OBAlignMode.HW_MODE) 更省算力，但依赖固件/分辨率
        # 组合，容易"配置成功但实际没对齐"。第一版优先用最不易翻车的软件对齐。
        # 参考：https://github.com/orbbec/pyorbbecsdk 的 align_filter 示例
        self._align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

        self.pipeline.start(cfg)

        # 内参（对齐到彩色，所以用彩色流内参）
        intr = color_profile.get_intrinsic()
        self._intr = Intrinsics(fx=intr.fx, fy=intr.fy, cx=intr.cx, cy=intr.cy)

        # 深度 scale：原始值 -> 米。SDK 的 get_depth_scale() 通常给"原始->毫米"，再 /1000。
        self._depth_scale = 0.001  # 兜底；start 后用首帧实测覆盖

        # 丢前几帧等自动曝光稳定，并实测 depth_scale
        for _ in range(10):
            frames = self.pipeline.wait_for_frames(1000)
            if frames is None:
                continue
            df = frames.get_depth_frame()
            if df is not None:
                try:
                    self._depth_scale = float(df.get_depth_scale()) / 1000.0
                except Exception:
                    pass
        print(f"Orbbec RGBD 就绪：内参={self._intr}, depth_scale={self._depth_scale}")

    @property
    def intrinsics(self) -> Intrinsics:
        return self._intr

    @property
    def depth_scale(self) -> float:
        return self._depth_scale

    def get_frames(self):
        frames = self.pipeline.wait_for_frames(1000)
        if frames is None:
            raise RuntimeError("取帧超时")
        if self._align_filter is not None:
            frames = self._align_filter.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            raise RuntimeError("彩色或深度帧缺失")

        color = _color_to_bgr(color_frame)
        dh, dw = depth_frame.get_height(), depth_frame.get_width()
        depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((dh, dw))
        return color, depth

    def stop(self):
        self.pipeline.stop()
