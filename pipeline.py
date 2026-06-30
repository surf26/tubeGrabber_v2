"""
流程编排：单相机(眼在手上) 试管抓取/放置。

只依赖 interfaces 抽象，不关心真硬件还是 Mock。四种运行模式，从安全到危险递进：
    "detect" 只感知：移到全局视野 -> 拍 -> 双类检测 -> 结算 -> 建网格 -> 打印网格地图。不抓不放。
    "plan"   再规划：解析"源试管/目标空槽"行列号 -> 算预抓取/预放置位姿 + 安全检查。不动臂(除全局视野)。
    "move"   再移动：移到源试管上方(粗定位)，再 look-and-move 精对准停住。不下扎、不抓。
    "grasp"  才执行：精对准 -> 下扎抓取 -> 搬运 -> 目标空槽上方精对准 -> 下扎放下 -> 抬起回全局视野。

调试顺序：detect 看网格对不对 → plan 看位姿/越界 → move 看对准精度 → 最后才 grasp。
"""
from __future__ import annotations

import interfaces
from align import LookAndMoveAligner
from core import grasp_check
from core.grasp_planner import GraspPlanner, WorkspaceError
from core.rack_model import RackMap, build_rack
from core.solver import PoseSolver
from core.types import GraspTarget


class TubeGrabberPipeline:
    def __init__(self, camera: interfaces.RGBDCamera, detector: interfaces.Detector,
                 solver: PoseSolver, planner: GraspPlanner, robot: interfaces.Robot, *,
                 global_view_pose: list[float], speed: float,
                 tube_cls: int, empty_cls: int,
                 rack_params: dict, align_params: dict,
                 place_z_offset_m: float = 0.0, grasp_geom: dict | None = None):
        self.camera = camera
        self.detector = detector
        self.solver = solver
        self.planner = planner
        self.robot = robot
        self.global_view_pose = global_view_pose
        self.speed = speed
        self.tube_cls = tube_cls
        self.empty_cls = empty_cls
        self.rack_params = rack_params      # split_x,n_rows,n_cols,left_rows,right_rows
        self.align_params = align_params    # observe_height_m,converge_m,max_iters
        self.place_z_off = place_z_offset_m
        self.grasp_geom = grasp_geom        # 抓取几何，用于干涉检查（None=跳过）
        self.last_color = None

    # ---------- 移到全局视野（安全高位，竖直向下看全盘） ----------
    def move_to_global_view(self):
        print(f"  -> 移到全局视野位姿 {self._fmt(self.global_view_pose)}")
        self.planner.check_pose(self.global_view_pose)
        self.robot.move_to_pose(self.global_view_pose, speed=self.speed, block=True)

    # ---------- 感知：拍 -> 双类检测 -> 结算 -> 建网格 ----------
    def perceive_global(self) -> tuple[RackMap, list[GraspTarget]]:
        color, depth = self.camera.get_frames()
        self.last_color = color
        flange2base = self.robot.get_flange2base()       # eye_in_hand 必须实时读
        detections = self.detector.detect(color)
        targets = self.solver.solve(detections, depth, flange2base)
        rack = build_rack(targets, tube_cls=self.tube_cls,
                          image_width=color.shape[1], **self.rack_params)
        return rack, targets

    # ---------- look-and-move 精对准器（按目标类别构造） ----------
    def _aligner(self, target_cls: int) -> LookAndMoveAligner:
        return LookAndMoveAligner(
            self.camera, self.detector, self.solver, self.planner, self.robot,
            observe_height_m=self.align_params["observe_height_m"],
            converge_m=self.align_params["converge_m"],
            max_iters=self.align_params["max_iters"],
            speed=self.speed, target_cls=target_cls)

    # ---------- 抓取一个试管 ----------
    def pick(self, target: GraspTarget) -> None:
        print("\n[抓取] 源试管精对准 ...")
        refined = self._aligner(self.tube_cls).refine(target)
        plan = self.planner.plan(refined)
        for p in (plan.approach, plan.grasp, plan.retreat):
            self.planner.check_pose(p)
        self.robot.enable_gripper_power()
        self.robot.open_gripper()
        print(f"  -> 预抓取(上方) {self._fmt(plan.approach)}")
        self.robot.move_to_pose(plan.approach, speed=self.speed, block=True)
        print(f"  -> 下扎抓取   {self._fmt(plan.grasp)}")
        self.robot.move_to_pose(plan.grasp, speed=self.speed, block=True)
        self.robot.close_gripper()
        print(f"  -> 抬起       {self._fmt(plan.retreat)}")
        self.robot.move_to_pose(plan.retreat, speed=self.speed, block=True)

    # ---------- 放置到一个空槽 ----------
    def place(self, target: GraspTarget) -> None:
        print("\n[放置] 目标空槽精对准 ...")
        refined = self._aligner(self.empty_cls).refine(target)
        plan = self.planner.plan(refined, z_offset=self.place_z_off)
        for p in (plan.approach, plan.grasp, plan.retreat):
            self.planner.check_pose(p)
        print(f"  -> 预放置(上方) {self._fmt(plan.approach)}")
        self.robot.move_to_pose(plan.approach, speed=self.speed, block=True)
        print(f"  -> 下放到槽内   {self._fmt(plan.grasp)}")
        self.robot.move_to_pose(plan.grasp, speed=self.speed, block=True)
        self.robot.open_gripper()
        print(f"  -> 抬起         {self._fmt(plan.retreat)}")
        self.robot.move_to_pose(plan.retreat, speed=self.speed, block=True)

    # ---------- 总入口 ----------
    def run(self, mode: str = "detect",
            src_sid: str | None = None,
            dst_sid: str | None = None) -> RackMap:
        assert mode in ("detect", "plan", "move", "grasp")

        self.move_to_global_view()
        rack, targets = self.perceive_global()

        print(f"\n检测+结算到 {rack.n_detected} 个孔位（共 {len(targets)} 个目标）：")
        print(rack.ascii_grid())
        if not targets:
            print("  （没有有效目标：可能没检测到、或深度全是空洞）")
            return rack
        if mode == "detect":
            return rack

        # 抓取几何/干涉静态检查（固定参数，打印一次）
        if self.grasp_geom is not None:
            print("\n" + grasp_check.assess(**self.grasp_geom).text())

        # plan/move/grasp 都需要任务孔位编号（board.row.col）
        if src_sid is None or dst_sid is None:
            raise ValueError("plan/move/grasp 模式必须给 --src 和 --dst 孔位编号（如 left.A1）")
        src = rack.source_tube(src_sid)
        dst = rack.dest_empty(dst_sid)
        print(f"\n任务：抓 源试管[{src.sid}] {src.target}")
        print(f"      放 目标空槽[{dst.sid}] {dst.target}")

        src_plan = self.planner.plan(src.target)
        dst_plan = self.planner.plan(dst.target)
        print(f"  源 预抓取 approach = {self._fmt(src_plan.approach)}")
        print(f"  目标 预放置 approach = {self._fmt(dst_plan.approach)}")
        if mode == "plan":
            try:
                self.planner.check_pose(src_plan.approach)
                self.planner.check_pose(dst_plan.approach)
                print("  安全检查：通过 ✅")
            except WorkspaceError as e:
                print(f"  安全检查：不通过 ❌ {e}")
            return rack

        if mode == "move":
            # 只演示"粗定位 + 精对准"，停在源试管上方，不抓
            print(f"\n[粗定位] 移到源试管上方 {self._fmt(src_plan.approach)}")
            self.planner.check_pose(src_plan.approach)
            self.robot.move_to_pose(src_plan.approach, speed=self.speed, block=True)
            refined = self._aligner(self.tube_cls).refine(src.target)
            print(f"  精对准后源试管 base = {(refined.base_xyz*1000).round(1).tolist()}mm（停在上方，不抓）")
            return rack

        # mode == "grasp"：完整抓-搬-放
        self.pick(src.target)
        self.place(dst.target)
        print("\n[完成] 回到全局视野")
        self.move_to_global_view()
        return rack

    @staticmethod
    def _fmt(pose: list[float]) -> str:
        x, y, z, rx, ry, rz = pose
        return (f"[x={x*1000:.1f} y={y*1000:.1f} z={z*1000:.1f} mm | "
                f"rx={rx:.3f} ry={ry:.3f} rz={rz:.3f} rad]")
