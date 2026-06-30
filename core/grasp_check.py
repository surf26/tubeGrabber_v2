"""
抓取可行性 / 干涉 静态检查（纯几何，不碰硬件）。

下扎抓取时两根手指从试管【两侧的缝隙】插下去：
    相邻盖中心间距 pitch，盖直径 cap_d → 两盖之间的净缝隙 = pitch - cap_d = 盖边到边间距 gap。
    一根手指厚 finger_t 要塞进这条缝里，单边余量 = (gap - finger_t) / 2。
另外夹爪张开宽度必须 > 盖直径才能套下去。

这些只取决于固定的几何参数，跟具体抓哪根无关，所以开跑时打印一次即可。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClearanceReport:
    ok: bool
    lines: list[str]

    def text(self) -> str:
        head = "抓取几何检查：" + ("可行 ✅" if self.ok else "有风险 ⚠️")
        return head + "\n" + "\n".join("  " + ln for ln in self.lines)


def assess(*, cap_d: float, gap: float, finger_t: float,
           gripper_max_open: float, margin: float) -> ClearanceReport:
    lines: list[str] = []
    ok = True

    # 1) 两指能否套住盖：张开宽度 > 盖直径
    open_room = gripper_max_open - cap_d
    if open_room <= 0:
        ok = False
        lines.append(f"夹爪张开 {gripper_max_open*1000:.0f}mm ≤ 盖直径 {cap_d*1000:.0f}mm，套不下去 ❌")
    else:
        lines.append(f"夹爪张开 {gripper_max_open*1000:.0f}mm > 盖直径 {cap_d*1000:.0f}mm，"
                     f"富余 {open_room*1000:.0f}mm（套得下）")

    # 2) 手指能否塞进相邻盖之间的缝隙
    side = (gap - finger_t) / 2.0    # 单边余量
    lines.append(f"相邻盖缝隙 {gap*1000:.0f}mm，指厚 {finger_t*1000:.0f}mm → 单边余量 {side*1000:.1f}mm")
    if side < 0:
        ok = False
        lines.append("缝隙比手指还窄，指头插不进相邻试管之间 ❌（需更薄的指/更大间距/换夹取策略）")
    elif side < margin:
        ok = False
        lines.append(f"单边余量 < 期望 {margin*1000:.1f}mm，非常紧 ⚠️："
                     "指尖必须精准对准缝隙中心；建议先在边角试管上验证、降速、考虑抓取方向选间距更大的轴")
    else:
        lines.append("缝隙余量够，但仍建议指尖对准缝隙中心")

    return ClearanceReport(ok=ok, lines=lines)
