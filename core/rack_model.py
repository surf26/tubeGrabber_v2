"""
网格地图（试管架建模）—— 左右双板 × 4行(A–D) × 3列(1–3) = 24 孔。

编号沿用 github.com/surf26/detector 的 slot_mapper 约定，保证你输入的 `board.row.col`
（如 left.A1 / right.C3）和你在 detector 里看到的叠加图完全一致：
  - 按 board_split_x（像素 u，默认图像宽/2）分左/右板；
  - 板内按像素 v 纵向分 4 行：图像【下方】是 ri=0；left 行序=[D,C,B,A]、right=[A,B,C,D]
    （即 left 底行=D，right 底行=A，和 detector 一致）；
  - 行内按像素 u 排 3 列：right 板从左到右=1,2,3；left 板从右到左=1,2,3。
几何映射在【像素空间】做（和 detector 一致），每个孔再挂上 solver 算出的 base 坐标供机械臂用。

占用与否来自类别：cls == tube_cls → 有试管(T)，否则空槽(.)；没检测到 → unknown(?)。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from core.types import GraspTarget

BOARDS = ("left", "right")
DEFAULT_LEFT_ROWS = ["D", "C", "B", "A"]
DEFAULT_RIGHT_ROWS = ["A", "B", "C", "D"]


@dataclass
class Slot:
    board: str
    row: str
    col: int
    occupied: bool | None        # True=有试管, False=空槽, None=未检测到
    target: GraspTarget | None    # 对应结算目标（None=未检测到）

    @property
    def sid(self) -> str:
        return f"{self.board}.{self.row}{self.col}"


def parse_sid(s: str) -> tuple[str, str, int]:
    """'left.A1' / 'left A 1' / 'leftA1' → ('left','A',1)。"""
    s = s.strip().replace(".", " ").replace("_", " ")
    if " " in s:
        parts = s.split()
        if len(parts) == 3:
            board, row, col = parts[0].lower(), parts[1].upper(), int(parts[2])
            return board, row, col
        s = "".join(parts)
    # 紧凑形式 leftA1 / rightC3
    low = s.lower()
    for b in BOARDS:
        if low.startswith(b):
            rest = s[len(b):]
            return b, rest[0].upper(), int(rest[1:])
    raise ValueError(f"无法解析孔位编号: {s!r}（示例 left.A1 / right.C3）")


class RackMap:
    """24 孔位地图。按 board.row.col 查孔；区分占用/空/未检测。"""

    def __init__(self, slots: list[Slot], rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self._by_sid = {s.sid: s for s in slots}

    @property
    def slots(self) -> list[Slot]:
        """所有被检测到的孔位（target 非空）。"""
        return [s for s in self._by_sid.values() if s.target is not None]

    @property
    def n_detected(self) -> int:
        return len(self.slots)

    def get(self, board: str, row: str, col: int) -> Slot | None:
        return self._by_sid.get(f"{board}.{row}{col}")

    def get_sid(self, sid: str) -> Slot | None:
        b, r, c = parse_sid(sid)
        return self.get(b, r, c)

    def source_tube(self, sid: str) -> Slot:
        s = self.get_sid(sid)
        if s is None or s.target is None:
            raise KeyError(f"{sid} 没检测到（unknown），无法作为源试管")
        if not s.occupied:
            raise ValueError(f"{sid} 是空槽，没有试管可抓")
        return s

    def dest_empty(self, sid: str) -> Slot:
        s = self.get_sid(sid)
        if s is None or s.target is None:
            raise KeyError(f"{sid} 没检测到（unknown），无法作为目标空槽")
        if s.occupied:
            raise ValueError(f"{sid} 已有试管，不能放入")
        return s

    def ascii_grid(self) -> str:
        """两块板并排打印。T=试管 .=空槽 ?=未检测到。"""
        def cell(board, row, col):
            s = self.get(board, row, col)
            return "?" if s is None or s.target is None else ("T" if s.occupied else ".")
        head = "      " + " ".join(f"c{c}" for c in range(1, self.cols + 1))
        lines = [f"{'left':<6}{'':<{self.cols*3}}   {'right'}", head + "     " + head.strip()]
        rows_lr = sorted(set(DEFAULT_LEFT_ROWS) | set(DEFAULT_RIGHT_ROWS))  # A..D
        for row in rows_lr:
            lcells = "  ".join(cell("left", row, c) for c in range(1, self.cols + 1))
            rcells = "  ".join(cell("right", row, c) for c in range(1, self.cols + 1))
            lines.append(f"{row:<5} {lcells}     {row:<3} {rcells}")
        lines.append("（T=试管  .=空槽  ?=未检测到）")
        return "\n".join(lines)


def _assign_board_rows_cols(targets: list[GraspTarget], board: str,
                            rows: list[str], n_cols: int) -> list[Slot]:
    """板内：按 v 分行、按 u 排列（复刻 slot_mapper 的纵向分桶 + 行内排序）。"""
    if not targets:
        return []
    vs = [t.pixel[1] for t in targets]
    v_max, v_min = max(vs), min(vs)
    v_span = max(v_max - v_min, 1.0)

    by_row: dict[int, list[GraspTarget]] = defaultdict(list)
    for t in targets:
        ri = int((v_max - t.pixel[1]) / v_span * (len(rows) - 0.001))
        ri = min(max(ri, 0), len(rows) - 1)
        by_row[ri].append(t)

    slots: list[Slot] = []
    for ri, items in by_row.items():
        # right 板列从左到右(+u)，left 板从右到左(-u)
        items = sorted(items, key=lambda t: t.pixel[0], reverse=(board == "left"))
        for ci, t in enumerate(items[:n_cols]):
            slots.append(Slot(board=board, row=rows[ri], col=ci + 1,
                              occupied=None, target=t))   # occupied 由调用方填
    return slots


def build_rack(targets: list[GraspTarget], *, tube_cls: int, image_width: int,
               split_x: float | None = None,
               n_rows: int = 4, n_cols: int = 3,
               left_rows: list[str] | None = None,
               right_rows: list[str] | None = None) -> RackMap:
    """把结算目标按左右双板 + 行列映射成 24 孔地图。"""
    left_rows = left_rows or DEFAULT_LEFT_ROWS
    right_rows = right_rows or DEFAULT_RIGHT_ROWS
    sx = split_x if split_x is not None else image_width / 2.0

    left_t = [t for t in targets if t.pixel[0] < sx]
    right_t = [t for t in targets if t.pixel[0] >= sx]

    slots = (_assign_board_rows_cols(left_t, "left", left_rows, n_cols)
             + _assign_board_rows_cols(right_t, "right", right_rows, n_cols))
    for s in slots:
        s.occupied = (s.target.cls == tube_cls)
    return RackMap(slots, n_rows, n_cols)
