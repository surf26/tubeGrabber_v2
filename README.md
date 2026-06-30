# tube_grabber_V2 — 单相机(眼在手上)试管抓取/放置

RM65-B + Orbbec Gemini 336(装在末端，垂直向下) + INSPIRE EG2-4C2 夹爪。
任务：识别试管架上每个槽的"有试管/空槽"，按你给的【行,列】把某根试管抓起、放进某个空槽。

## 整体思路（coarse-to-fine，单眼在手）

1. **全局看**：机械臂抬到 `GLOBAL_VIEW_POSE`(竖直向下高位)，相机一次看全盘。
2. **建网格**：YOLO 双类检测(`tube`/`empty`) → 每个框反投影+坐标变换到基座系 → 按 base 坐标聚成 **行×列** 网格，每格标"占用/空"。
3. **取任务**：你输入 `--src 行,列 --dst 行,列`，查网格拿到源试管 / 目标空槽的 base 坐标。
4. **粗定位**：移到目标正上方。
5. **微调(look-and-move)**：在正上方近距离重拍一张、重新结算，必要时挪一点再拍，直到坐标收敛(见 `align.py`)。比 IBVS 简单、可解释。
6. **抓→搬→放**：开爪→24V 供电→下扎→闭合→抬起→搬到空槽上方→微调→下放→开爪→抬起→回全局视野。

坐标链(眼在手上)：`p_base = T_flange2base(实时读) · T_cam2gripper(标定) · deproject(u,v,z)`

## 目录

```
config.py            所有参数，带 ★ 的上真机前必须实测填写
calib/               眼在手上手眼标定（要重标）：capture→calibrate→verify
core/                纯逻辑(Mac 可单测)：geometry/solver/grasp_planner/rack_model/...
hardware/            真实硬件：orbbec_rgbd / rm_robot / yolo_detector
mocks/               占位实现：造网格试管架，供离线验证
align.py             look-and-move 精对准
pipeline.py          流程编排(detect/plan/move/grasp 四档递进)
run.py               真机入口
demo_offline.py      Mac 上用 Mock 跑通全链路（不接硬件）
```

## 用法

### 0. Mac 上先验证逻辑（不接硬件）
```
python demo_offline.py
```

### 1. 重新做眼在手上标定（在接了硬件的机器上）
```
cd calib
python list_cameras.py            # 确认相机序列号
python capture.py                 # 棋盘格固定不动，多姿态采 15+ 组
python calibrate.py               # 求 T_cam2gripper → results/eye_in_hand_handeye.json
python verify.py                  # 挪动手臂看 target-in-base 是否稳定
```

### 2. 真机运行（务必逐级放开）
```
python run.py --mode detect                              # 看网格地图对不对
python run.py --mode plan  --src left.A1 --dst right.C3  # 看位姿+安全检查+抓取几何检查
python run.py --mode move  --src left.A1 --dst right.C3  # 移到源上方+精对准，不抓
python run.py --mode grasp --src left.A1 --dst right.C3  # 才真的抓-搬-放
```
`--src/--dst` 是孔位编号 `board.row.col`（left.A1 / right.C3）。加 `--show` / `--save` 看/存标注图。

## 已填 / 待填参数（见 config.py）
已填(实测)：`TOOL_Z_OFFSET=0.22`、抓取几何(盖直径22 / 缝隙10 / 指厚8 / 露出22 / 夹深10mm)、
模型 `models/tube_empty_yolo.pt`(来自 github.com/surf26/detector，类别 **0=empty,1=tube**)。

仍待你上真机填的 ★：
- `GRASP_ORIENTATION_RXRYRZ`：示教器读的竖直向下姿态
- `GLOBAL_VIEW_POSE`：能拍全盘的高位姿态
- `GRIPPER_MAX_OPEN_M`：夹爪最大张开宽度（查 EG2-4C2 规格书，要 > 盖直径 22mm）
- `PLACE_Z_OFFSET_M`：放置下放深度（move 档观察后实测）
- `BOARD_SPLIT_X`：左右分界像素（两板不对称时）
- `WORKSPACE_*`：jog 四角实测的安全边界
- 重新跑 `calib/` 得到 `eye_in_hand_handeye.json`

## 孔位编号约定（board.row.col，沿用 detector 仓库）
左右两块板 × 4 行(A–D) × 3 列(1–3)。`detect` 会打印 ASCII 图，对照实物；
方向不对就改 config 的 `BOARD_SPLIT_X / LEFT_ROWS / RIGHT_ROWS`。
```
left              right
      c1 c2 c3     c1 c2 c3
A     .  T  T     A   .  T  T     T=有试管  .=空槽  ?=未检测到
B     T  T  T     B   T  T  T
C     T  T  T     C   T  T  T
D     T  T  T     D   T  T  T
```

## ⚠ 抓取间距很紧
盖缝隙 10mm、指厚 8mm → 单边仅 **1mm** 余量。`plan` 档会打印抓取几何检查。
上真机务必：指尖对准缝隙中心、降速、先拿边角试管验证。
