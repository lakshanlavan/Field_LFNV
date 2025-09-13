#!/usr/bin/env python3
# ROW (ODOM) → PRE_RESET (ODOM) → TURN (IGN GT global path) → POST_TURN_RESET (ODOM) → REF_WAIT → ROW (ODOM)

import math, rclpy, threading, subprocess, re, sys
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, Quaternion, PoseStamped

PI = math.pi

def normalize_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))

def unwrap_to_ref(angle: float, ref: float) -> float:
    while angle - ref >  PI: angle -= 2.0*PI
    while angle - ref < -PI: angle += 2.0*PI
    return angle

def quat_to_yaw(q: Quaternion) -> float:
    return math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))

def yaw_from_quat_xyzw(qx,qy,qz,qw) -> float:
    siny_cosp = 2.0*(qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0*(qy*qy + qz*qz)
    return math.atan2(siny_cosp, cosy_cosp)

def closest_angle_to_ref(target: float, ref: float) -> float:
    base = normalize_angle(target)
    cand = [base, base + 2.0*PI, base - 2.0*PI]
    best = min(cand, key=lambda a: abs(normalize_angle(unwrap_to_ref(a, ref) - ref)))
    return unwrap_to_ref(best, ref)

def se2_apply_inverse_of_pose(pose_xyz_th, wpts_xyz_th):
    rx, ry, rth = pose_xyz_th
    c, s = math.cos(-rth), math.sin(-rth)
    out = []
    for (x, y, yaw) in wpts_xyz_th:
        dx, dy = x - rx, y - ry
        X = c*dx - s*dy
        Y = s*dx + c*dy
        YAW = normalize_angle(yaw - rth)
        out.append((X, Y, YAW))
    return out

# -------- GLOBAL Ω waypoint sets (each ends at next row start) --------
OMEGA_WPTS_GLOBAL_1 = [
    (3.68076,  6.46997,  +1.9486),
    (3.50943,  6.90584,  +1.1687),
    (3.69496,  7.33528,  +0.3545),
    (4.12500,  7.50000,  -0.3535),
    (4.56985,  7.33528,  -1.1673),
    (4.75061,  6.90584,  -1.9521),
    (4.57326,  6.46997,  -1.7203),
    (4.50000,  6.00000,  -1.5708),
]


OMEGA_WPTS_GLOBAL_2 = [
    (4.42928, -4.97129,  -1.9486),
    (4.25559, -5.40895,  -1.1687),
    (4.43819, -5.83832,  -0.3545),
    (4.87500, -6.00000,   0.3535),
    (5.31313, -5.83832,   1.1673),
    (5.49646, -5.40895,   1.9521),
    (5.32100, -4.97129,   1.7203),
    (5.25000, -4.25000,   1.5708),
]


OMEGA_WPTS_GLOBAL_3 = [
    (5.18076,  6.46997,  +1.9486),
    (5.00943,  6.90584,  +1.1687),
    (5.19496,  7.33528,  +0.3545),
    (5.62500,  7.50000,  -0.3535),
    (6.07063,  7.33528,  -1.1673),
    (6.25148,  6.90584,  -1.9521),
    (6.07350,  6.46997,  -1.7203),
    (6.00000,  6.00000,  -1.5708),
]


OMEGA_WPTS_GLOBAL_4 = [
    (2.92928, -4.97129,  -1.9486),
    (2.75559, -5.40895,  -1.1687),
    (2.93819, -5.83832,  -0.3545),
    (3.37500, -6.00000,   0.3535),
    (3.81313, -5.83832,   1.1673),
    (3.99646, -5.40895,   1.9521),
    (3.82100, -4.97129,   1.7203),
    (3.75000, -4.50000,   1.5708),
]

# Registry for easy lookup
TURN_PATHS = {
    1: OMEGA_WPTS_GLOBAL_1,
    2: OMEGA_WPTS_GLOBAL_2,
    3: OMEGA_WPTS_GLOBAL_3,
    4: OMEGA_WPTS_GLOBAL_4,
}

class RowManagerFSM(Node):
    def __init__(self):
        super().__init__('row_manager_fsm')

        # --- fixed pose plan ---
        self.row_pose_source  = 'odom'
        self.turn_pose_source = 'gt_ign'

        # topics / params
        self.gt_pose_topic    = self.declare_parameter('gt_pose_topic', '/gt_pose').value
        self.odom_topic       = self.declare_parameter('odom_topic', '/odom_local').value
        self.world            = self.declare_parameter('world', 'field').value
        self.model_name_match = self.declare_parameter('model_name_match', 'a200_0000/robot').value
        self.ign_pose_topic   = f"/world/{self.world}/dynamic_pose/info"

        # reference gating (for REF_WAIT)
        self.ref_topic        = self.declare_parameter('ref_topic', '/row_centerline_path').value  # nav_msgs/Path
        self.ref_gate_dist    = float(self.declare_parameter('ref_gate_dist', 0.6).value)          # meters
        self.ref_timeout_s    = float(self.declare_parameter('ref_timeout_s', 3.0).value)          # optional watchdog

        # geometry
        self.sequence           = self.declare_parameter('sequence', list(range(1,16+1))).value
        self.line_separation    = float(self.declare_parameter('line_separation', 0.75).value)
        self.line_num           = int(self.declare_parameter('line_num', 16).value)
        self.field_width        = float(self.declare_parameter('field_width', 11.25).value)
        self.row_length         = float(self.declare_parameter('row_length', 10.5).value)
        self.headland           = float(self.declare_parameter('headland', 3.5).value)
        self.auto_total_depth   = bool(self.declare_parameter('auto_total_depth', True).value)
        self.field_depth        = float(self.declare_parameter('field_depth', 0.0).value)
        if self.auto_total_depth:
            self.field_depth = self.row_length + 2.0*self.headland
        self.auto_row_distance  = bool(self.declare_parameter('auto_row_distance', True).value)
        self.row_distance       = float(self.declare_parameter('row_distance', 0.0).value)
        if self.auto_row_distance or self.row_distance <= 0.0:
            self.row_distance = self.row_length

        # PID (ROW defaults)
        self.v_max         = float(self.declare_parameter('v_max', 0.22).value)
        self.v_min         = float(self.declare_parameter('v_min', 0.02).value)
        self.v_cap_near    = float(self.declare_parameter('v_cap_near', 0.05).value)
        self.kp_lin        = float(self.declare_parameter('kp_lin', 0.8).value)
        self.kp_yaw        = float(self.declare_parameter('kp_yaw', 1.5).value)
        self.kd_yaw        = float(self.declare_parameter('kd_yaw', 0.12).value)
        self.ki_yaw        = float(self.declare_parameter('ki_yaw', 0.0).value)
        self.w_max         = float(self.declare_parameter('w_max', 2.2).value)
        self.w_min_spin    = float(self.declare_parameter('w_min_spin', 0.20).value)
        self.final_stop_lin= bool(self.declare_parameter('final_stop_lin', True).value)

        # --- TURN-specific boosts (snappier) ---
        self.v_max_turn      = float(self.declare_parameter('v_max_turn', 0.28).value)
        self.kp_lin_turn     = float(self.declare_parameter('kp_lin_turn', 1.05).value)
        self.w_max_turn      = float(self.declare_parameter('w_max_turn', 2.8).value)
        self.w_min_spin_turn = float(self.declare_parameter('w_min_spin_turn', 0.30).value)

        # yaw final
        self.kp_yaw_final  = float(self.declare_parameter('kp_yaw_final', 1.7).value)
        self.kd_yaw_final  = float(self.declare_parameter('kd_yaw_final', 0.14).value)
        self.ki_yaw_final  = float(self.declare_parameter('ki_yaw_final', 0.02).value)

        # tolerances (slightly looser mid, faster settle)
        self.pos_tol_mid   = float(self.declare_parameter('pos_tol_mid', 0.08).value)   # was 0.05
        self.pos_tol_final = float(self.declare_parameter('pos_tol_final', 0.04).value)
        self.yaw_tol_final = float(self.declare_parameter('yaw_tol_final', 0.07).value)
        self.final_settle_time = float(self.declare_parameter('final_settle_time', 0.25).value)  # was 0.40

        # shaping of linear slow-down during turns
        self.turn_lin_floor = float(self.declare_parameter('turn_lin_floor', 0.35).value)  # min fraction of v to keep
        self.turn_lin_gain  = float(self.declare_parameter('turn_lin_gain', 0.60).value)   # 0..1 (higher = more slow-down)

        # reset gates
        self.reset_wait_sec = float(self.declare_parameter('reset_wait_sec', 0.35).value)
        self.odom_zero_xy   = float(self.declare_parameter('odom_zero_xy', 0.02).value)
        self.odom_zero_th   = float(self.declare_parameter('odom_zero_th', 0.03).value)

        # ---- NEW: Turn order (cycle through sets) ----
        # Provide as a parameter like: turn_order: [1,2,3,4]
        self.turn_order = self.declare_parameter('turn_order', [1,2,3,4]).value
        if not self.turn_order:
            self.turn_order = [1,2,3,4]
        # internal pointer to current turn index in `turn_order`
        self.turn_order_i = 0

        # derived sanity
        span_x = (self.line_num - 1.0)*self.line_separation
        if span_x > self.field_width + 1e-9:
            raise RuntimeError("Rows span exceeds field width")

        # state
        self.idx = 0
        self.state = "ROW"   # ROW → PRE_RESET → TURN → POST_TURN_RESET → REF_WAIT → ROW
        self.seg_start_xy = None

        # path
        self.path_wpts = []   # will be set right before each TURN
        self.wp_i = 0
        self.final_reached_t = None

        # pose buffers
        self._odom_ok=False;   self._odom_xyzth=(0.0,0.0,0.0)
        self._gtros_ok=False;  self._gtros_xyzth=(0.0,0.0,0.0)
        self._ign_ok=False;    self._ign_xyzth=(0.0,0.0,0.0)

        # active pose
        self.active_pose_source = self.row_pose_source
        self.x=0.0; self.y=0.0; self.th=0.0; self.pose_ready=False

        # yaw PID internals
        self._yaw_err_prev = 0.0
        self._yaw_I = 0.0

        # ref wait buffers
        self._last_reset_walltime = 0.0
        self._ref_latest = None   # (stamp_sec, min_dist)
        self._ref_first_seen_after_reset = False

        # pubs/subs
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.pub_ai  = self.create_publisher(Bool, '/ai_enabled',  latched)
        self.pub_mpc = self.create_publisher(Bool, '/mpc_enabled', latched)
        self.pub_hb  = self.create_publisher(Bool, '/segment_heartbeat', 10)
        self.pub_cmd = self.create_publisher(Twist, '/a200_0000/cmd_vel', 10)

        # pose feeds
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 20)
        self.create_subscription(PoseStamped, self.gt_pose_topic, self.gt_pose_cb, 50)
        self._stop_ign = False
        threading.Thread(target=self._ign_pose_reader, daemon=True).start()

        # reference feed for REF_WAIT
        self.create_subscription(Path, self.ref_topic, self.ref_cb, 5)

        self.dt = 0.05
        self.create_timer(self.dt, self.tick)  # 20 Hz

        self.heartbeat(); self.set_ai(True); self.set_mpc(True)
        self.get_logger().info("FSM ready. ROW=ODOM → TURN=GT_IGN. Continuous multi-turn mode ENABLED.")

    # ------------ callbacks ------------
    def _pull_active_pose(self):
        if self.active_pose_source == 'odom' and self._odom_ok:
            self.x, self.y, self.th = self._odom_xyzth; self.pose_ready=True; return
        if self.active_pose_source == 'gt_ros' and self._gtros_ok:
            self.x, self.y, self.th = self._gtros_xyzth; self.pose_ready=True; return
        if self.active_pose_source == 'gt_ign' and self._ign_ok:
            self.x, self.y, self.th = self._ign_xyzth; self.pose_ready=True; return
        self.pose_ready=False

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position; q = msg.pose.pose.orientation
        self._odom_xyzth = (float(p.x), float(p.y), float(quat_to_yaw(q)))
        self._odom_ok = True

    def gt_pose_cb(self, msg: PoseStamped):
        p = msg.pose.position; q = msg.pose.orientation
        self._gtros_xyzth = (float(p.x), float(p.y), float(yaw_from_quat_xyzw(q.x,q.y,q.z,q.w)))
        self._gtros_ok = True

    def _ign_pose_reader(self):
        try:
            proc = subprocess.Popen(["ign","topic","-e","-t", self.ign_pose_topic],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
        except FileNotFoundError:
            self.get_logger().error("`ign` not found on PATH; cannot read GT pose.")
            return
        pose_block, depth = [], 0
        try:
            for line in proc.stdout:
                if self._stop_ign: break
                s = line.rstrip("\n")
                if "pose {" in s:
                    pose_block = [s]; depth = s.count("{") - s.count("}")
                    continue
                if depth > 0:
                    pose_block.append(s)
                    depth += s.count("{") - s.count("}")
                    if depth <= 0:
                        block = "\n".join(pose_block)
                        mname = re.search(r'name:\s*"([^"]+)"', block)
                        name  = mname.group(1) if mname else ""
                        if not name or (self.model_name_match not in name):
                            pose_block = []; depth = 0; continue
                        def grab(section, key):
                            m = re.search(fr'{section}\s*\{{[^}}]*\b{key}:\s*([-0-9.eE]+)', block)
                            return float(m.group(1)) if m else None
                        x  = grab("position","x"); y  = grab("position","y")
                        qx = grab("orientation","x"); qy = grab("orientation","y")
                        qz = grab("orientation","z"); qw = grab("orientation","w")
                        if None in (x,y,qx,qy,qz,qw): pose_block = []; depth = 0; continue
                        th = yaw_from_quat_xyzw(qx,qy,qz,qw)
                        self._ign_xyzth = (x,y,th); self._ign_ok = True
                        pose_block = []; depth = 0
        except Exception as e:
            self.get_logger().warn(f"[ign_pose_reader] {e}")
        finally:
            try: proc.terminate()
            except: pass

    def ref_cb(self, msg: Path):
        if not self.pose_ready: 
            return
        min_d = float('inf')
        for p in msg.poses:
            dx = p.pose.position.x - self.x
            dy = p.pose.position.y - self.y
            d = math.hypot(dx, dy)
            if d < min_d: min_d = d
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
        self._ref_latest = (stamp_sec, min_d)

    # ------------ utils ------------
    def set_ai(self, on: bool):  self.pub_ai.publish(Bool(data=on));  self.get_logger().info(f"AI {'ENABLED' if on else 'DISABLED'}")
    def set_mpc(self, on: bool): self.pub_mpc.publish(Bool, '/mpc_enabled') if False else self.pub_mpc.publish(Bool(data=on)); self.get_logger().info(f"MPC {'ENABLED' if on else 'DISABLED'}")
    def heartbeat(self):         self.pub_hb.publish(Bool(data=True)); self.get_logger().info("HEARTBEAT published.")
    def stop(self):              self.pub_cmd.publish(Twist()); self.get_logger().info("cmd_vel: STOP published.")
    def dist_from(self, xy0):    return math.hypot(self.x - xy0[0], self.y - xy0[1])

    # ---- NEW: choose the next TURN path (cycles through) ----
    def _choose_next_turn_path(self):
        if not self.turn_order:  # fallback
            self.turn_order = [1,2,3,4]
        turn_id = self.turn_order[self.turn_order_i % len(self.turn_order)]
        if turn_id not in TURN_PATHS:
            self.get_logger().warn(f"turn_id={turn_id} not in TURN_PATHS; defaulting to 1")
            turn_id = 1
        self.turn_order_i += 1
        # Copy so we can safely modify per-turn (residual)
        return [tuple(w) for w in TURN_PATHS[turn_id]]

    # ------------ main loop ------------
    def tick(self):
        self._pull_active_pose()
        if not self.pose_ready: return

        if self.state == "ROW":
            if self.active_pose_source != 'odom':
                self.active_pose_source = 'odom'
                self.get_logger().info("[POSE] Switched active source → ODOM (ROW)")
            if self.seg_start_xy is None:
                self.seg_start_xy = (self.x, self.y)
                self.get_logger().info(f"ROW segment started at {self.seg_start_xy}  pose=({self.x:+.3f},{self.y:+.3f},{self.th:+.3f})")
            d = self.dist_from(self.seg_start_xy)
            if d >= self.row_distance:
                self.get_logger().info(f"Row distance reached ({d:.3f} m) → disable AI/MPC → HEARTBEAT → TURN prep")
                self.set_ai(False); self.set_mpc(False); self.stop(); self.heartbeat()

                # ---- NEW: pick the next TURN path here ----
                self.path_wpts = self._choose_next_turn_path()

                # wait for odom zero → residual compensate → switch to IGN turn
                self.reset_wait_started = self.get_clock().now().nanoseconds * 1e-9
                self._pre_reset_t0 = self.reset_wait_started
                self._pre_reset_checks = 0
                self.state = "PRE_RESET"

        elif self.state == "PRE_RESET":
            now = self.get_clock().now().nanoseconds * 1e-9
            soft_xy = max(self.odom_zero_xy, 0.12)
            soft_th = max(self.odom_zero_th, 0.05)
            max_wait = 1.5
            if now - self.reset_wait_started >= self.reset_wait_sec:
                self._pre_reset_checks += 1
                self.get_logger().info(f"Post-reset odom check: pose=({self.x:+.3f},{self.y:+.3f},{self.th:+.3f})")
                strict_ok = (abs(self.x) <= self.odom_zero_xy and abs(self.y) <= self.odom_zero_xy and abs(self.th) <= self.odom_zero_th)
                soft_ok   = (abs(self.x) <= soft_xy and abs(self.y) <= soft_xy and abs(self.th) <= soft_th)
                waited = now - self._pre_reset_t0
                if strict_ok or soft_ok or waited >= max_wait:
                    residual = (self.x, self.y, self.th)
                    if not (strict_ok or soft_ok):
                        self.get_logger().warn(
                            f"Odom still offset after {waited:.2f}s "
                            f"(x={self.x:+.3f}, y={self.y:+.3f}, th={self.th:+.3f}). "
                            "Proceeding with residual compensation of the CURRENT TURN path."
                        )
                        self.path_wpts = se2_apply_inverse_of_pose(residual, self.path_wpts)
                    # switch to IGN (GT) for the turn
                    self.active_pose_source = self.turn_pose_source
                    self.get_logger().info(f"[POSE] Switched active source → {self.turn_pose_source.upper()} (TURN)")
                    self._start_turn(global_with_residual=True)
                else:
                    self.get_logger().warn("Odom not zeroed yet, extending wait 0.20 s...")
                    self.reset_wait_started = now
                    self.reset_wait_sec = 0.20

        elif self.state == "TURN":
            done = self.turn_control_step_pid()
            if done:
                self.stop()
                # post-turn: reset odom and WAIT for ref before enabling MPC
                self.active_pose_source = 'odom'
                self.get_logger().info(f"[POSE] Switched active source → ODOM (POST_TURN_RESET)")
                self.heartbeat()  # trigger odom reset
                self._last_reset_walltime = self.get_clock().now().nanoseconds * 1e-9
                self.reset_wait_started = self._last_reset_walltime
                self._post_reset_t0 = self._last_reset_walltime
                self._post_reset_checks = 0
                self.state = "POST_TURN_RESET"

        elif self.state == "POST_TURN_RESET":
            now = self.get_clock().now().nanoseconds * 1e-9
            soft_xy = max(self.odom_zero_xy, 0.12)
            soft_th = max(self.odom_zero_th, 0.05)
            max_wait = 1.5
            if now - self.reset_wait_started >= self.reset_wait_sec:
                self._post_reset_checks += 1
                self.get_logger().info(f"Post-turn odom check: pose=({self.x:+.3f},{self.y:+.3f},{self.th:+.3f})")
                strict_ok = (abs(self.x) <= self.odom_zero_xy and abs(self.y) <= self.odom_zero_xy and abs(self.th) <= self.odom_zero_th)
                soft_ok   = (abs(self.x) <= soft_xy and abs(self.y) <= soft_xy and abs(self.th) <= soft_th)
                waited = now - self._post_reset_t0
                if strict_ok or soft_ok or waited >= max_wait:
                    # odom settled → enable AI ONLY and go wait for reference
                    self.set_ai(True)
                    self._ref_first_seen_after_reset = False
                    self._ref_latest = None
                    self._ref_wait_started = now
                    self.state = "REF_WAIT"
                    self.get_logger().info("POST_TURN_RESET ok → AI ENABLED → REF_WAIT (gate MPC until ref is nearby & fresh)")
                else:
                    self.get_logger().warn("Odom not zeroed yet (post-turn), extending wait 0.20 s...")
                    self.reset_wait_started = now
                    self.reset_wait_sec = 0.20

        elif self.state == "REF_WAIT":
            # require: (1) a Path message stamped after reset, (2) nearest point within gate distance
            now = self.get_clock().now().nanoseconds * 1e-9
            if self._ref_latest is not None:
                stamp_sec, min_d = self._ref_latest
                if stamp_sec > self._last_reset_walltime:
                    if not self._ref_first_seen_after_reset:
                        self._ref_first_seen_after_reset = True
                        self.get_logger().info(f"[REF_WAIT] first ref after reset: min_dist={min_d:.2f} m")
                    if min_d <= self.ref_gate_dist:
                        self.set_mpc(True)
                        self.seg_start_xy = None
                        self.idx += 1
                        self.state = "ROW"
                        self.get_logger().info(f"[REF_WAIT] reference within {self.ref_gate_dist} m (min={min_d:.2f}) → MPC ENABLED → ROW")
                        return
            # optional timeout: if no good ref for too long, still enable to avoid deadlock (tune as needed)
            if self.ref_timeout_s > 0 and (now - self._ref_wait_started) >= self.ref_timeout_s:
                self.get_logger().warn("[REF_WAIT] timeout waiting for fresh/nearby reference → enabling MPC anyway")
                self.set_mpc(True)
                self.seg_start_xy = None
                self.idx += 1
                self.state = "ROW"

    # ------------ TURN helpers ------------
    def _start_turn(self, global_with_residual: bool):
        self.wp_i = 0
        self._yaw_err_prev = 0.0
        self._yaw_I = 0.0
        self.final_reached_t = None
        if self.path_wpts:
            tag = "GLOBAL (residual handled)" if global_with_residual else "GLOBAL"
            self.get_logger().info(
                f"TURN start {tag}; wp0={tuple(round(v,3) for v in self.path_wpts[0])}, "
                f"wplast={tuple(round(v,3) for v in self.path_wpts[-1])}"
            )
        self.state = "TURN"
        self.seg_start_xy = None

    # ------------ PID follower (GLOBAL) ------------
    def turn_control_step_pid(self) -> bool:
        if self.wp_i >= len(self.path_wpts):
            self.get_logger().info("[TURN] Completed all waypoints.")
            return True

        gx, gy, gyaw = self.path_wpts[self.wp_i]
        dx = gx - self.x; dy = gy - self.y
        dist = math.hypot(dx, dy)
        is_final = (self.wp_i == len(self.path_wpts) - 1)

        # Use TURN-specific caps/gains
        v_max = self.v_max_turn
        kp_lin = self.kp_lin_turn
        w_max = self.w_max_turn
        w_min_spin = self.w_min_spin_turn

        bearing = unwrap_to_ref(math.atan2(dy, dx), self.th)
        yaw_ref = bearing if (not is_final or dist > 0.15) else closest_angle_to_ref(gyaw, self.th)
        yaw_err = normalize_angle(yaw_ref - self.th)
        pos_tol = self.pos_tol_final if is_final else self.pos_tol_mid

        v_cmd = kp_lin * dist
        if dist < 0.25:
            v_cmd = min(v_cmd, self.v_cap_near)

        if is_final:
            kp = self.kp_yaw_final; kd = self.kd_yaw_final; ki = self.ki_yaw_final
        else:
            kp = self.kp_yaw;       kd = self.kd_yaw;       ki = self.ki_yaw

        d_err = (yaw_err - self._yaw_err_prev) / max(1e-6, self.dt)
        self._yaw_err_prev = yaw_err
        self._yaw_I += ki * yaw_err * self.dt
        self._yaw_I = float(max(-0.6, min(0.6, self._yaw_I)))
        w_pid = kp * yaw_err + kd * d_err + self._yaw_I

        w_cmd = float(max(-w_max, min(w_max, w_pid)))
        if abs(w_pid - w_cmd) > 1e-6:
            self._yaw_I *= 0.9
            self.get_logger().info("[TURN] ω clipped at w_max (TURN).")

        turn_slow = max(self.turn_lin_floor, 1.0 - self.turn_lin_gain * (abs(w_cmd)/w_max))
        v_cmd *= turn_slow

        v_cmd = float(max(self.v_min, min(v_max, v_cmd)))

        if not is_final and dist <= pos_tol:
            self.get_logger().info(f"[TURN] reached wp {self.wp_i}/{len(self.path_wpts)-1} at ({gx:+.3f},{gy:+.3f}); advancing.")
            self.wp_i += 1
            self._yaw_err_prev = 0.0
            self._yaw_I = 0.0
            self.pub_cmd.publish(Twist())
            return False

        if is_final:
            pos_ok = (dist <= self.pos_tol_final)
            yaw_ok = (abs(yaw_err) <= self.yaw_tol_final)
            now = self.get_clock().now().nanoseconds * 1e-9

            if dist < 0.10:
                v_cmd = min(v_cmd, 0.04)

            if self.final_stop_lin and not yaw_ok and abs(yaw_err) > 0.18:
                v_cmd = 0.0
                if abs(w_cmd) < w_min_spin:
                    w_cmd = math.copysign(w_min_spin, (w_cmd if abs(w_cmd) > 1e-3 else yaw_err))

            if pos_ok and yaw_ok:
                if self.final_reached_t is None:
                    self.final_reached_t = now
                    self.get_logger().info("[FINAL] pose reached — settling…")
                self.pub_cmd.publish(Twist())
                if (now - self.final_reached_t) >= self.final_settle_time:
                    self.get_logger().info("[FINAL] pose settled. TURN complete.")
                    self.wp_i += 1
                    return True
                return False
            else:
                self.final_reached_t = None

        cmd = Twist()
        cmd.linear.x  = float(v_cmd)
        cmd.angular.z = float(w_cmd)
        self.pub_cmd.publish(cmd)
        return False

# ------------ main ------------
def main(args=None):
    rclpy.init(args=args)
    node = RowManagerFSM()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if hasattr(node, "_stop_ign"):
                node._stop_ign = True
        except: pass
        node.stop(); node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
