"""
真实 YOLO 双类检测器（试管 / 空槽）。实现 interfaces.Detector 协议。依赖 ultralytics。

每个检测框输出本工程的 Detection（含像素中心 + 类别 cls）。
加载时打印 model.names，方便你核对 config.TUBE_CLS / EMPTY_CLS 的整数 id 填得对不对。
"""
from __future__ import annotations

import numpy as np

from core.types import Detection


class YoloDetector:
    def __init__(self, model_path, conf=0.5, iou=0.45, imgsz=640):
        from ultralytics import YOLO
        print(f"加载 YOLO 模型：{model_path}")
        self.model = YOLO(str(model_path))
        self.names = self.model.names   # {id: name}
        print(f"模型类别 names = {self.names}  ← 核对 config.TUBE_CLS/EMPTY_CLS 是否对应 tube/empty")
        self.conf, self.iou, self.imgsz = conf, iou, imgsz

    def detect(self, color_bgr: np.ndarray) -> list[Detection]:
        results = self.model(color_bgr, conf=self.conf, iou=self.iou,
                             imgsz=self.imgsz, verbose=False)
        r = results[0]
        dets = []
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dets.append(Detection(
                pixel=((x1 + x2) / 2, (y1 + y2) / 2),
                bbox=(x1, y1, x2, y2),
                conf=float(box.conf[0]),
                cls=int(box.cls[0]),
            ))
        return dets
