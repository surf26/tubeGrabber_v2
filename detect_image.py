#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯检测可视化（不依赖机械臂）：跑 YOLO 双类模型，把检测框+类别+置信度画出来并存图。
用来快速验证模型/类别映射，不做坐标结算、不连机械臂。

用法：
    # 对一张静态图（只需 ultralytics，Mac 也能跑）
    python detect_image.py --image path/to.jpg

    # 对手上相机实时取一帧（需 pyorbbecsdk，在接了相机的机器上）
    python detect_image.py --live

    # 对一个文件夹里所有图批量出图
    python detect_image.py --dir path/to/images/

输出存到 outputs/detect_*.jpg；加 --show 弹窗显示。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

import config
import viz
from hardware.yolo_detector import YoloDetector


def _process(detector: YoloDetector, color, stem: str, show: bool):
    dets = detector.detect(color)
    vis = viz.draw_detections(color, dets, tube_cls=config.TUBE_CLS, names=detector.names)
    path = viz.save_image(vis, name=f"detect_{stem}.jpg")
    n_tube = sum(d.cls == config.TUBE_CLS for d in dets)
    print(f"  {stem}: 共 {len(dets)} 个框（tube={n_tube} empty={len(dets)-n_tube}）-> {path}")
    if show:
        viz.show_image(vis)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="单张图片路径")
    g.add_argument("--dir", help="图片文件夹（批量）")
    g.add_argument("--live", action="store_true", help="手上相机实时取一帧")
    ap.add_argument("--show", action="store_true", help="弹窗显示")
    args = ap.parse_args()

    detector = YoloDetector(config.YOLO_MODEL_PATH, conf=config.YOLO_CONF,
                            iou=config.YOLO_IOU, imgsz=config.YOLO_IMG_SIZE)

    if args.image:
        color = cv2.imread(args.image)
        if color is None:
            raise FileNotFoundError(f"读不到图片：{args.image}")
        _process(detector, color, Path(args.image).stem, args.show)

    elif args.dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = sorted(p for p in Path(args.dir).iterdir() if p.suffix.lower() in exts)
        if not files:
            raise FileNotFoundError(f"{args.dir} 里没有图片")
        for p in files:
            color = cv2.imread(str(p))
            if color is not None:
                _process(detector, color, p.stem, args.show)

    else:  # --live
        from hardware.orbbec_rgbd import OrbbecRGBD
        cam = OrbbecRGBD(serial=config.HAND_CAMERA_SERIAL,
                         width=config.CAMERA_WIDTH, height=config.CAMERA_HEIGHT,
                         fps=config.CAMERA_FPS)
        try:
            color, _ = cam.get_frames()
            _process(detector, color, "live", args.show)
        finally:
            cam.stop()


if __name__ == "__main__":
    main()
