"""
可视化：把网格地图(每格的框 + 行列号 + base坐标 + 占用/空)画到彩色图上。
依赖 opencv，只在 run.py 用 --show/--save 时才会用到，不影响 core 纯逻辑。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.rack_model import RackMap


def draw_rack(color_bgr: np.ndarray, rack: RackMap) -> np.ndarray:
    """每孔：检测框 + 中心点 + 编号(board.row.col) + base坐标(mm)。绿=有试管, 红=空槽。"""
    out = color_bgr.copy()
    for slot in rack.slots:
        t = slot.target
        u, v = int(round(t.pixel[0])), int(round(t.pixel[1]))
        color = (0, 255, 0) if slot.occupied else (0, 0, 255)   # 绿=tube 红=empty
        if t.bbox is not None:
            x1, y1, x2, y2 = (int(round(c)) for c in t.bbox)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.circle(out, (u, v), 4, (0, 215, 255), -1)
        x, y, z = (t.base_xyz * 1000).tolist()
        cv2.putText(out, slot.sid, (u - 24, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        cv2.putText(out, f"({x:.0f},{y:.0f},{z:.0f})", (u - 30, v + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    cv2.putText(out, f"24-slot  green=tube red=empty",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_detections(color_bgr: np.ndarray, detections, tube_cls: int,
                    names: dict | None = None) -> np.ndarray:
    """画原始 YOLO 检测框（不结算坐标，不需要机械臂）：框+类别名+置信度。
    tube=绿、empty(或其它)=红。names 是 {cls_id: name}，没有就直接显示 id。"""
    out = color_bgr.copy()
    n_tube = 0
    for d in detections:
        is_tube = (d.cls == tube_cls)
        n_tube += int(is_tube)
        color = (0, 255, 0) if is_tube else (0, 0, 255)
        x1, y1, x2, y2 = (int(round(c)) for c in d.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        name = names.get(d.cls, str(d.cls)) if names else str(d.cls)
        cv2.putText(out, f"{name} {d.conf:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(out, f"det={len(detections)}  tube={n_tube}  empty={len(detections)-n_tube}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def save_image(img: np.ndarray, out_dir: str | Path = "outputs", name: str = "rack.jpg") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    cv2.imwrite(str(path), img)
    return path


def show_image(img: np.ndarray, window: str = "rack (按任意键关闭)") -> None:
    cv2.imshow(window, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
