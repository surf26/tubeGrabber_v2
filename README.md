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

四档 `detect → plan → move → grasp` 从“只看”到“真抓”逐级放开。**第一次务必一档一档来，别跳。**
`--src/--dst` 是孔位编号 `board.row.col`（如 left.A1 / right.C3）。加 `--show`/`--save` 看/存标注图。

---

#### 准备（每次开跑前）
1. 在**接了臂+相机**的那台机器（已装 `Robotic_Arm` / `pyorbbecsdk` / `ultralytics`，`conda activate surf`）。
2. 机械臂上电、连网，`ping 192.168.1.18` 通；急停按钮放手边。
3. 相机 USB3.0 插好：`python calib/list_cameras.py` 能看到 `CP84B4100090`。
4. `config.ROBOT_SPEED` 保持小（默认 10）。

#### 第 1 档 `detect`——只看，不动手（除移到全局视野）
```
python run.py --mode detect
```
预期输出（节选）：
```
已加载手眼标定：T_cam2gripper ...
模型类别 names = {0: 'empty', 1: 'tube'}   ← 核对是不是 0=empty,1=tube
  -> 移到全局视野位姿 [x=8.9 y=208.3 z=350.6 mm | ...]
检测+结算到 N 个孔位（共 N 个目标）：
left              right
      c1 c2 c3     c1 c2 c3
A     .  T  T     A   .  T  T
...
```
**必须核对 3 件事**，有一个不对就先停下调，别往下走：
- ① `model.names` 是 `{0: empty, 1: tube}`；不是就改 config 的 `EMPTY_CLS/TUBE_CLS`。
- ② ASCII 网格的 `left/right`、`A–D` 行、`1–3` 列方向和**实物一致**；不一致改 `BOARD_SPLIT_X / LEFT_ROWS / RIGHT_ROWS`。
- ③ 每孔的 base 坐标 `(x,y,z)` 落在机械臂正常工作区内、量级合理（mm）。
  - 加 `--show` 可弹窗看检测框+编号+坐标叠加图核对。

#### 第 2 档 `plan`——看位姿 + 安全 + 抓取几何，仍不动手
```
python run.py --mode plan --src left.A1 --dst right.C3
```
会打印：源/目标的预抓取位姿、抓取几何检查（夹爪张开 vs 盖直径、缝隙余量）、安全检查结果。
- 安全检查 ❌ 越界：说明 `WORKSPACE_*` 太紧或目标确实不可达 → 按 detect 看到的真实坐标把 `WORKSPACE_*` 收到合理范围再试。

#### 第 3 档 `move`——粗定位 + 精对准，停在源上方（**第一次真动臂！**）
```
python run.py --mode move --src left.A1 --dst right.C3
```
机械臂会移到源试管上方，然后 look-and-move 反复“拍一张→挪一点”，打印每次 `Δ` 直到收敛：
```
[align] 第1次：Δ=6.20mm base=[...]
[align] 第2次：Δ=0.80mm base=[...]
[align] 已收敛(<2.0mm)
```
**手放急停旁**。看它有没有稳稳停在试管正上方、Δ 是否在收敛。不收敛通常是深度噪声/检测抖动 → 调 `DEPTH_*` 或 `YOLO_CONF`。

#### 第 4 档 `grasp`——真抓真放
```
python run.py --mode grasp --src left.A1 --dst right.C3
```
完整序列：精对准 → 供电24V → 开爪 → 下扎 → 闭合 → 抬起 → 搬到空槽上方 → 精对准 → 下放 → 开爪 → 抬起 → 回全局视野。
- **抓取间距很紧（见文末）**：先拿边角试管验证，确认指尖对准缝隙中心。
- 放置太浅/太深：调 `PLACE_Z_OFFSET_M`（负=多往下放）。

#### 常见问题
| 现象 | 可能原因 / 处理 |
|------|------|
| `detect` 检测不到试管 | 相机没看到 / `YOLO_CONF` 太高 / 深度全空洞（看 `DEPTH_VALID_*` 范围） |
| 网格行列方向反 | 改 `BOARD_SPLIT_X / LEFT_ROWS / RIGHT_ROWS` |
| 坐标整体偏移 | 手眼标定不准 → 重跑 `calib/`；或确认 `GLOBAL_VIEW_POSE` 单位/数值 |
| `plan` 安全检查总不过 | `WORKSPACE_*` 没按真实坐标设 |
| `align` 的 Δ 不收敛 | 深度噪声/检测抖动 → 调 `DEPTH_PATCH / DEPTH_VALID_* / YOLO_CONF` |
| 夹爪不动 | 没供电：确认 `enable_gripper_power()`（末端 24V）生效 |

## 参数状态（见 config.py）
**已填**：
- 手眼标定 `calib/results/eye_in_hand_handeye.json`（T_cam2gripper，20 样本）
- `GLOBAL_VIEW_POSE` / `GRASP_ORIENTATION_RXRYRZ`（示教器实测，竖直向下）
- `TOOL_Z_OFFSET=0.22`、抓取几何(盖直径22 / 缝隙10 / 指厚8 / 露出22 / 夹深10mm)、`GRIPPER_MAX_OPEN_M=0.032`
- 模型 `models/tube_empty_yolo.pt`(来自 github.com/surf26/detector，类别 **0=empty,1=tube**)

**仍建议上真机后按实测收紧/确认**：
- `WORKSPACE_*`：现在是围绕全局位姿的临时框，跑 `detect` 看真实坐标后收紧 + jog 四角确认
- `PLACE_Z_OFFSET_M`：放置下放深度（`move` 档观察空槽实际深度后调）
- `BOARD_SPLIT_X`：左右分界像素（两板不对称时才需填）

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
