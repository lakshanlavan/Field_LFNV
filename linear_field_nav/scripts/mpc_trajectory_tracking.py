#!/usr/bin/env python3
"""
Advanced MPC Tracking Controller Node (ROS 2 / Python 3)
========================================================

Implements an optimized Model Predictive Control (MPC) for a differential-drive robot to track
reference waypoints. Features an advanced dynamic model with wheel slip, enhanced cost function
with contouring error, oscillation suppression, and a recovery mode for handling large deviations.
Uses a finite-state machine (FSM) to switch between MPC and BYPASS modes based on straight/curve
batch detection with hysteresis.

Subscribes:
  - /reference_waypoints (geometry_msgs/PoseArray): Reference path.
  - /odom_local (nav_msgs/Odometry): Robot odometry.
  - /mpc_enabled (std_msgs/Bool): MPC enable/disable (TRANSIENT_LOCAL).

Publishes:
  - /a200_0000/cmd_vel (geometry_msgs/Twist): Robot velocity commands.
  - /mpc/reference_path (nav_msgs/Path): Reference trajectory (frame_id='odom_local').
  - /mpc/actual_path (nav_msgs/Path): Actual robot path (frame_id='odom_local').
  - /mpc/predicted_path (nav_msgs/Path): Predicted trajectory (frame_id='odom_local').
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry, Path as NavPath
from geometry_msgs.msg import Twist, PoseStamped, PoseArray
import numpy as np
from math import atan2, sin, cos, sqrt, pi
import casadi as ca
import os
import csv
import time
import signal
from scipy.ndimage import uniform_filter1d
from typing import Optional, Tuple, List
from enum import Enum
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import CubicSpline

from std_msgs.msg import String                     # NEW
import json                                         # (you already have these; ensure `json` is present)
from datetime import datetime                       # NEW

FRAME_ID =  "odom_local" 

# Define FSM modes
class Mode(Enum):
    MPC = 0    # Use full solver
    BYPASS = 1 # Constant-velocity straight drive

def yaw_from_quat(q) -> float:
    """Extract yaw angle from quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return atan2(siny_cosp, cosy_cosp)

def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return atan2(sin(angle), cos(angle))

class DynamicSystem:
    """Advanced kinematic model for differential-drive robot with wheel slip."""
    def __init__(self, dt: float):
        self.dt = dt
        self.state_dim = 3  # [x, y, theta]
        self.control_dim = 2  # [v, w]

        # Symbolic variables
        x = ca.MX.sym('x', self.state_dim)
        u = ca.MX.sym('u', self.control_dim)
        slip_factor = ca.MX.sym('slip', 1)

        # Kinematic model with slip
        v = u[0] * (1 - 0.1 * slip_factor)  # Linear velocity with slip
        w = u[1] * (1 - 0.05 * slip_factor)  # Angular velocity with slip
        x_dot = ca.vertcat(
            v * ca.cos(x[2]),
            v * ca.sin(x[2]),
            w
        )

        # ── Correct RK-4 integration 
        # Build a helper that re-evaluates the ODE f(x,u,slip) at each stage
        f = ca.Function('f', [x, u, slip_factor], [x_dot])

        k1 = f(x,               u, slip_factor)
        k2 = f(x + 0.5*self.dt*k1, u, slip_factor)
        k3 = f(x + 0.5*self.dt*k2, u, slip_factor)
        k4 = f(x +     self.dt*k3, u, slip_factor)

        x_next = x + (self.dt/6) * (k1 + 2*k2 + 2*k3 + k4)

        self.f_dyn = ca.Function('f_dyn', [x, u, slip_factor], [x_next])
        self.f_dot = ca.Function('f_dot', [x, u, slip_factor], [x_dot])

    def dynamics(self, x: np.ndarray, u: np.ndarray, slip: float = 0.0) -> np.ndarray:
        """Compute next state with given slip factor."""
        return np.array(self.f_dyn(x, u, slip)).flatten()

class MPCSolver:
    """Optimized MPC solver with advanced cost function and oscillation suppression."""
    def __init__(
        self,
        N: int,
        dt: float,
        max_v: float,
        max_w: float,
        min_v: float,
        node: Node
    ):
        self.node = node
        self.N = N
        self.dt = dt
        self.max_v = max_v
        self.max_w = max_w
        self.min_v = min_v
        self.dyn = DynamicSystem(dt)
        self.Q = np.diag([30.0, 30.0, 15.0])  # State error weights
        self.Q_terminal = 10 * self.Q  # Terminal cost weight
        self.R = np.diag([0.1, 0.1])  # Control effort weights
        self.R_delta = np.diag([1.0, 1.0])  # Control rate weights
        self.Q_v = 5.0  # Linear velocity tracking weight
        self.Q_w = 4.0  # Angular velocity tracking weight
        self.prev_X = None
        self.prev_U = None
        self.last_solver_init = 0.0
        self.setup_solver()

    def setup_solver(self) -> None:
        """Configure the MPC optimization problem."""
        opti = ca.Opti()
        self.X = opti.variable(self.dyn.state_dim, self.N + 1)  # States: [x, y, theta]
        self.U = opti.variable(self.dyn.control_dim, self.N)  # Controls: [v, w]
        self.P = opti.parameter(self.dyn.state_dim, self.N + 1)  # Reference states
        self.V_ref = opti.parameter(1, self.N)  # Reference linear velocities
        self.W_ref = opti.parameter(1, self.N)  # Reference angular velocities
        self.X0 = opti.parameter(self.dyn.state_dim)  # Initial state
        self.U_prev = opti.parameter(self.dyn.control_dim)  # Previous control
        self.Slip = opti.parameter(1)  # Slip factor

        J = 0.0  # Cost function
        for k in range(self.N):
            # Contouring error
            err_pos = self.X[:2, k] - self.P[:2, k]
            theta_err = ca.atan2(
                ca.sin(self.X[2, k] - self.P[2, k]),
                ca.cos(self.X[2, k] - self.P[2, k])
            )
            err = ca.vertcat(err_pos, theta_err)
            J += ca.mtimes([err.T, self.Q, err])

            # Velocity tracking
            J += self.Q_v * (self.U[0, k] - self.V_ref[0, k])**2
            J += self.Q_w * (self.U[1, k] - self.W_ref[0, k])**2

            # Control effort
            J += ca.mtimes([self.U[:, k].T, self.R, self.U[:, k]])

            # Control smoothness
            delta_u = self.U[:, k] - (self.U_prev if k == 0 else self.U[:, k-1])
            J += ca.mtimes([delta_u.T, self.R_delta, delta_u])

        # Terminal cost
        err_pos = self.X[:2, self.N] - self.P[:2, self.N]
        theta_err = ca.atan2(
            ca.sin(self.X[2, self.N] - self.P[2, self.N]),
            ca.cos(self.X[2, self.N] - self.P[2, self.N])
        )
        err = ca.vertcat(err_pos, theta_err)
        J += ca.mtimes([err.T, self.Q_terminal, err])

        opti.minimize(J)
        opti.subject_to(self.X[:, 0] == self.X0)
        for k in range(self.N):
            opti.subject_to(self.X[:, k + 1] == self.dyn.f_dyn(self.X[:, k], self.U[:, k], self.Slip))
            opti.subject_to(self.U[0, k] >= self.min_v)
            opti.subject_to(self.U[0, k] <= self.max_v)
            opti.subject_to(self.U[1, k] >= -self.max_w)
            opti.subject_to(self.U[1, k] <= self.max_w)

        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 100,
            'ipopt.tol': 1e-3,
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.linear_solver': 'mumps',
            'ipopt.warm_start_init_point': 'yes',
        }
        opti.solver('ipopt', opts)
        self.opti = opti
        self.last_solver_init = time.time()
        self.node.get_logger().info("MPC solver initialized.")

    def solve(
        self,
        x0: np.ndarray,
        ref_traj: np.ndarray,
        v_ref: np.ndarray,
        w_ref: np.ndarray,
        slip: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve the MPC problem."""
        if (ref_traj.shape != (3, self.N + 1) or
            v_ref.shape != (self.N,) or
            w_ref.shape != (self.N,) or
            np.any(np.isnan(x0)) or
            np.any(np.isnan(ref_traj)) or
            np.any(np.isnan(v_ref)) or
            np.any(np.isnan(w_ref))):
            self.node.get_logger().error(
                f"Invalid MPC inputs: x0={x0}, ref_traj.shape={ref_traj.shape}, "
                f"v_ref.shape={v_ref.shape}, w_ref.shape={w_ref.shape}"
            )
            return self.compute_fallback(x0, ref_traj), np.zeros((3, self.N + 1))

        try:
            self.node.get_logger().debug(f"Solving MPC with x0={x0}")
            self.opti.set_value(self.X0, x0)
            self.opti.set_value(self.P, ref_traj)
            self.opti.set_value(self.V_ref, v_ref.reshape(1, -1))
            self.opti.set_value(self.W_ref, w_ref.reshape(1, -1))
            self.opti.set_value(self.U_prev, self.prev_U[:, 0] if self.prev_U is not None else np.zeros(2))
            self.opti.set_value(self.Slip, slip)

            init_X = self.prev_X if self.prev_X is not None else np.tile(x0.reshape(-1, 1), (1, self.N + 1))
            init_U = self.prev_U if self.prev_U is not None else np.zeros((2, self.N))
            self.opti.set_initial(self.X, init_X)
            self.opti.set_initial(self.U, init_U)

            sol = self.opti.solve()
            u_opt = sol.value(self.U[:, 0])
            X_pred = sol.value(self.X)
            self.prev_X = X_pred
            self.prev_U = sol.value(self.U)
            return u_opt, X_pred
        except Exception as e:
            self.node.get_logger().error(f"MPC solver failed: {str(e)}")
            if time.time() - self.last_solver_init > 1.0:
                self.setup_solver()
            return self.compute_fallback(x0, ref_traj), np.zeros((3, self.N + 1))

    def compute_fallback(self, x0: np.ndarray, ref_traj: np.ndarray) -> np.ndarray:
        """Compute fallback control if solver fails."""
        self.node.get_logger().warn("Using fallback control due to solver failure.")
        dx = ref_traj[0, 0] - x0[0]
        dy = ref_traj[1, 0] - x0[1]
        dist = sqrt(dx**2 + dy**2)
        target_yaw = atan2(dy, dx)
        yaw_err = normalize_angle(target_yaw - x0[2])
        v_cmd = np.clip(0.3 * dist, self.min_v, self.max_v)
        w_cmd = np.clip(0.5 * yaw_err, -self.max_w, self.max_w)
        return np.array([v_cmd, w_cmd])

class MPCTrackingController(Node):
    """ROS 2 node for MPC-based trajectory tracking with FSM for curve detection."""
    def __init__(self):
        super().__init__('mpc_tracking_controller')
        self.get_logger().info("Initializing MPC Tracking Controller.")

        # Robot parameters
        self.max_v = 0.4  # m/s
        self.min_v = 0.0
        self.max_w = 0.8  # rad/s
        self.wheelbase = 0.512  # m
        self.N = 20  # Prediction horizon
        self.dt = 0.1  # s
        self.waypoint_threshold = 0.3  # m
        self.num_waypoints = 80
        self.min_waypoint_spacing = 0.05  # m
        self.enable_logging = True
        self._armed_for_first_cycle = False
        self._print_once            = False

        # FSM state & hysteresis counters
        self.mode = Mode.MPC
        self._straight_batches = 0
        self._curve_batches = 0
        self._straight_needed = 3  # Need 3 straight batches to stay/return to MPC
        self._curve_needed = 2     # Need 2 curve batches to switch to BYPASS
        self._heading_ref = 0.0    # To hold heading when bypassing

        # MPC setup
        self.mpc_solver = MPCSolver(self.N, self.dt, self.max_v, self.max_w, self.min_v, self)
        self.dyn = DynamicSystem(self.dt)

        # State variables
        self.x = self.y = self.th = 0.0
        self.pose_ready = False
        self.path_ready = False
        self.waypoints: Optional[np.ndarray] = None
        self.current_index = 0
        self.mpc_enabled = True
        self.last_waypoint_time: Optional[float] = None
        self.last_odom_time: Optional[float] = None
        self.timeout = 5.0  # s
        self.path_act = NavPath()
        self.path_act.header.frame_id = FRAME_ID

        # Recovery mode variables
        self.recovery_mode = False
        self.recovery_start_pos = None
        self.cte_threshold = 0.9

        # Odometry smoothing
        self.last_th = None

        # QoS profiles
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers and subscriptions
        self.cmd_pub = self.create_publisher(Twist, '/a200_0000/cmd_vel', 10)
        self.ref_pub = self.create_publisher(NavPath, '/mpc/reference_path', 10)
        self.act_pub = self.create_publisher(NavPath, '/mpc/actual_path', 10)
        self.pred_pub = self.create_publisher(NavPath, '/mpc/predicted_path', 10)
        self.control_group = ReentrantCallbackGroup()
        self.create_subscription(Bool, '/mpc_enabled', self.mpc_enabled_cb, qos_latched, callback_group=self.control_group)
        self.create_subscription(PoseArray, '/reference_waypoints', self.path_cb, qos, callback_group=self.control_group)
        self.create_subscription(Odometry, '/odom_local', self.odom_cb, qos, callback_group=self.control_group)

        # ---- runtime trace intake ----
        self._ctx_mid = None
        self.create_subscription(String, '/rt_ctx_mid', self.ctx_mid_cb, 10)

        # Timers
        self.create_timer(self.dt, self.control_loop, callback_group=self.control_group)
        self.create_timer(1.0, self.diagnostic_cb)

        # CSV logging
        if self.enable_logging:
            self.csv_path = os.path.join(os.path.expanduser('~'), 'mpc_log.csv')
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['time', 'theta_err', 'cross_track_err', 'v_cmd', 'w_cmd'])

        # ---- end-to-end runtime CSV ----
        os.makedirs('runs', exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        self.rt_csv_path = os.path.join('runs', f"e2e_runtime-{ts}-0.csv")
        self.rt_csv = open(self.rt_csv_path, 'w', newline='', buffering=1)
        self.rt_writer = csv.writer(self.rt_csv)
        self.rt_writer.writerow([
            "frame_uuid","seq",
            "recv_t_ns","ai_end_ns","p2w_end_ns","mpc_end_ns",
            "dt_ai_ns","dt_p2w_ns","dt_mpc_ns","dt_total_ns","fps_estimate"
        ])
        self.get_logger().info(f"[runtime] writing: {self.rt_csv_path}")

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def ctx_mid_cb(self, msg: String) -> None:
        self._ctx_mid = json.loads(msg.data)

    def mpc_enabled_cb(self, msg: Bool) -> None:
        """Handle MPC enable/disable."""
        self.mpc_enabled = msg.data
        if msg.data:                 # we were just enabled
            self._armed_for_first_cycle = True
            self._print_once = True   # tell control_loop to emit the dump once
        self.get_logger().info(f"MPC {'enabled' if msg.data else 'paused'}.")

    def signal_handler(self, sig: int, frame) -> None:
        """Handle shutdown signals."""
        self.get_logger().info("Shutting down.")
        self.destroy_node()
        rclpy.shutdown()

    def diagnostic_cb(self) -> None:
        """Check for timeouts and publish stop commands if needed."""
        t_now = self.get_clock().now().nanoseconds * 1e-9
        if self.mpc_enabled:
            if self.last_odom_time and (t_now - self.last_odom_time > self.timeout):
                self.get_logger().warn("Odometry timeout. Stopping.")
                self.cmd_pub.publish(Twist())
            if self.last_waypoint_time and (t_now - self.last_waypoint_time > self.timeout):
                self.get_logger().warn("Waypoint timeout. Stopping.")
                self.cmd_pub.publish(Twist())

    def is_straight_batch(self, poses: List, angle_thresh_deg: float = 10.0) -> bool:
        """Determine if a batch of waypoints is straight based on position vectors."""
        if len(poses) < 10:
            return False  # Don’t classify undersized batches
        head = [atan2(poses[i+1].position.y - poses[i].position.y,
                      poses[i+1].position.x - poses[i].position.x)
                for i in range(9)]
        dtheta = np.array([normalize_angle(head[i+1] - head[i]) for i in range(8)])
        return np.max(np.abs(dtheta)) < np.deg2rad(angle_thresh_deg)

    def resample_waypoints(self, waypoints: List[List[float]], num_points: int) -> np.ndarray:
        """Resample waypoints with minimal smoothing to preserve sharp turns."""
        wps = np.asarray(waypoints)
        if len(wps) < 2:
            self.get_logger().warn("Insufficient waypoints for resampling.")
            return np.zeros((num_points, 3))
        t_old = np.linspace(0, 1, len(wps))
        t_new = np.linspace(0, 1, num_points)
        x = np.interp(t_new, t_old, wps[:, 0])  # No smoothing
        y = np.interp(t_new, t_old, wps[:, 1])  # No smoothing
        theta = np.zeros(num_points)
        for i in range(num_points - 1):
            theta[i] = atan2(y[i + 1] - y[i], x[i + 1] - x[i])
        theta[-1] = theta[-2]  # Copy previous heading
        return np.stack([x, y, theta], axis=-1)

    def create_reference_trajectory(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate reference trajectory, linear velocities, and angular velocities."""
        if self.waypoints is None or len(self.waypoints) == 0:
            ref_traj = np.tile(np.array([self.x, self.y, self.th]), (self.N + 1, 1)).T
            v_ref = np.zeros(self.N)
            w_ref = np.zeros(self.N)
            return ref_traj, v_ref, w_ref

        traj = self.waypoints[idx: min(idx + self.N + 1, len(self.waypoints))]
        if traj.shape[0] < self.N + 1:
            traj = np.vstack([traj, np.tile(traj[-1], (self.N + 1 - traj.shape[0], 1))])
        ref_traj = traj.T[:, :self.N + 1]

        # Compute v_ref and w_ref based on path geometry
        dx = np.diff(ref_traj[0, :])
        dy = np.diff(ref_traj[1, :])
        distances = np.sqrt(dx**2 + dy**2)
        v_ref = distances / self.dt
        v_ref = np.clip(v_ref, self.min_v, self.max_v)

        # Apply smoothing to the linear velocity reference (v_ref)
        v_ref = uniform_filter1d(v_ref, size=5)  # Smooth the velocity reference

        dtheta = np.diff(ref_traj[2, :])
        dtheta = np.array([normalize_angle(a) for a in dtheta])
        w_ref = dtheta / self.dt

        if np.any(np.isnan(ref_traj)) or np.any(np.isnan(v_ref)) or np.any(np.isnan(w_ref)):
            self.get_logger().error("NaN detected in reference trajectory or velocities.")
            ref_traj = np.tile(np.array([self.x, self.y, self.th]), (self.N + 1, 1)).T
            v_ref = np.zeros(self.N)
            w_ref = np.zeros(self.N)

        return ref_traj, v_ref, w_ref


    def compute_cross_track_error(self, x: float, y: float, ref_traj: np.ndarray) -> float:
        """Calculate minimum distance to reference trajectory."""
        ref_points = ref_traj[:2, :].T
        robot_pos = np.array([x, y])
        return float(np.min(np.sqrt(np.sum((ref_points - robot_pos)**2, axis=1))))

    def path_cb(self, msg: PoseArray) -> None:
        """Process incoming waypoints and manage FSM mode switching."""
        if not msg.poses:
            self.path_ready = False
            self.get_logger().warn("Received empty waypoint message.")
            return

        # Classify batch and update FSM
        if len(msg.poses) >= 10:
            straight = self.is_straight_batch(msg.poses, angle_thresh_deg=10.0)
            if straight:
                self._straight_batches += 1
                self._curve_batches = 0
            else:
                self._curve_batches += 1
                self._straight_batches = 0

            # Transitions with hysteresis
            if self.mode == Mode.MPC and self._curve_batches >= self._curve_needed:
                self.mode = Mode.BYPASS
                self._heading_ref = self.th
                self._straight_batches = 0
                self._curve_batches = 0
                self.get_logger().info("[FSM] ➜ BYPASS mode (v=0.4, w=0)")
            elif self.mode == Mode.BYPASS and self._straight_batches >= self._straight_needed:
                self.mode = Mode.MPC
                self._straight_batches = 0
                self._curve_batches = 0
                self.get_logger().info("[FSM] ➜ MPC mode")

        # Process waypoints
        t_now = self.get_clock().now().nanoseconds * 1e-9
        pts = [[p.position.x, p.position.y, yaw_from_quat(p.orientation)] for p in msg.poses]
        if len(pts) < 2:
            self.get_logger().warn("Insufficient waypoints received.")
            return
        self.waypoints = self.resample_waypoints(pts, self.num_waypoints)
        self.current_index = 0
        self.path_ready = True
        self.last_waypoint_time = t_now

        path_msg = NavPath()
        path_msg.header.frame_id = FRAME_ID
        path_msg.header.stamp = self.get_clock().now().to_msg()
        for wp in self.waypoints:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = wp[0]
            ps.pose.position.y = wp[1]
            ps.pose.orientation.z = sin(wp[2] / 2)
            ps.pose.orientation.w = cos(wp[2] / 2)
            path_msg.poses.append(ps)
        self.ref_pub.publish(path_msg)

    def odom_cb(self, msg: Odometry) -> None:
        """Update robot state from odometry with smoothing."""
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        raw_th = yaw_from_quat(msg.pose.pose.orientation)
        if self.last_th is not None:
            self.th = 0.9 * self.th + 0.1 * normalize_angle(raw_th)
        else:
            self.th = normalize_angle(raw_th)
        self.last_th = self.th
        self.pose_ready = True
        self.last_odom_time = self.get_clock().now().nanoseconds * 1e-9

    def publish_predicted_path(self, X_pred: np.ndarray) -> None:
        """Publish predicted trajectory."""
        path_msg = NavPath()
        path_msg.header.frame_id = FRAME_ID
        path_msg.header.stamp = self.get_clock().now().to_msg()
        for k in range(X_pred.shape[1]):
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = float(X_pred[0, k])
            ps.pose.position.y = float(X_pred[1, k])
            path_msg.poses.append(ps)
        self.pred_pub.publish(path_msg)

    def control_loop(self) -> None:
        """Execute control loop with MPC or BYPASS based on FSM mode."""
        if not (self.mpc_enabled and self.pose_ready and self.path_ready):
            self.get_logger().warn("Control loop skipped: MPC or inputs not ready.")
            self.cmd_pub.publish(Twist())  # Stop if conditions not met
            return

        # Bypass MPC when in curve-detected batches
        if self.mode == Mode.BYPASS:
            heading_err = normalize_angle(self._heading_ref - self.th)
            cmd = Twist()
            cmd.linear.x = 0.4
            cmd.angular.z = 0.0         
            # cmd.angular.z = np.clip(0.8 * heading_err, -0.2, 0.2)
            self.cmd_pub.publish(cmd)
            return  # Skip all MPC, recovery, logging, etc.

        x0 = np.array([self.x, self.y, self.th])
        ref_traj, v_ref, w_ref = self.create_reference_trajectory(self.current_index)
        # >>> add the one-shot diagnostic dump here <<< ------------------------
        if self._armed_for_first_cycle and self._print_once:
            d = sqrt((ref_traj[0, 0] - self.x) ** 2 +
                    (ref_traj[1, 0] - self.y) ** 2)
            self.get_logger().info(
                "[FIRST-CYCLE] θ_robot={:+7.2f}°  θ_ref0={:+7.2f}°  "
                "w_ref0={:+6.3f}  dist_ref0={:4.2f} m  "
                "mode={}  wp_idx={}".format(
                    np.degrees(self.th),
                    np.degrees(ref_traj[2, 0]),
                    w_ref[0] if len(w_ref) else 0.0,
                    d,
                    self.mode.name,
                    self.current_index)
            )
            self._print_once = False           # emit only once
        # ----------------------------------------------------------------------

        cmd = Twist()

        # Proceed with normal MPC or recovery mode
        cte = self.compute_cross_track_error(self.x, self.y, ref_traj)
        if not self.recovery_mode:
            if abs(cte) <= self.cte_threshold:
                # Normal MPC operation
                self.get_logger().debug(f"Control loop: x0={x0}, ref_traj shape={ref_traj.shape}, v_ref shape={v_ref.shape}, w_ref shape={w_ref.shape}")
                u_opt, X_pred = self.mpc_solver.solve(x0, ref_traj, v_ref, w_ref)
                self._armed_for_first_cycle = False  
                cmd.linear.x = float(np.clip(u_opt[0], self.min_v, self.max_v))
                cmd.angular.z = float(np.clip(u_opt[1], -self.max_w, self.max_w))
                self.publish_predicted_path(X_pred)
            else:
                self.get_logger().warn(f"Cross-track error {cte:.2f} exceeds threshold {self.cte_threshold}. Entering recovery mode.")
                self.recovery_mode = True
                self.recovery_start_pos = [self.x, self.y]
                cmd.linear.x = 0.5
                cmd.angular.z = 0.0
        else:
            # In recovery mode
            cmd.linear.x = 0.5
            cmd.angular.z = 0.0
            distance_traveled = sqrt((self.x - self.recovery_start_pos[0])**2 + (self.y - self.recovery_start_pos[1])**2)
            if distance_traveled >= 0.5:
                cte = self.compute_cross_track_error(self.x, self.y, ref_traj)
                if abs(cte) <= self.cte_threshold:
                    self.get_logger().info(f"Cross-track error {cte:.2f} within threshold {self.cte_threshold}. Exiting recovery mode.")
                    self.recovery_mode = False
                else:
                    self.get_logger().warn(f"Cross-track error {cte:.2f} still exceeds threshold {self.cte_threshold}. Continuing recovery mode.")
                    self.recovery_start_pos = [self.x, self.y]

        self.cmd_pub.publish(cmd)
        # ---- runtime trace: mark MPC end and write CSV row ----
        if self._ctx_mid is not None:
            try:
                ctx = dict(self._ctx_mid)
                ctx["mpc_end_ns"] = time.monotonic_ns()
                r = int(ctx.get("recv_t_ns", 0))
                a = int(ctx.get("ai_end_ns", r))
                p = int(ctx.get("p2w_end_ns", a))
                m = int(ctx.get("mpc_end_ns", p))
                total = max(1, m - r)
                self.rt_writer.writerow([
                    ctx.get("frame_uuid",""), ctx.get("seq", 0),
                    r, a, p, m,
                    a - r, p - a, m - p, total, f"{1e9/total:.3f}"
                ])
            except Exception as e:
                self.get_logger().warn(f"runtime csv write error: {e}")
            finally:
                self._ctx_mid = None

        # Update waypoint index
        dist = sqrt((ref_traj[0, 0] - self.x)**2 + (ref_traj[1, 0] - self.y)**2)
        if dist < self.waypoint_threshold and self.current_index < len(self.waypoints) - 1:
            self.current_index += 1

        # Publish actual path
        p = PoseStamped()
        p.header.frame_id = FRAME_ID
        p.header.stamp = self.get_clock().now().to_msg()
        p.pose.position.x = self.x
        p.pose.position.y = self.y
        self.path_act.poses.append(p)
        if len(self.path_act.poses) > 300:
            self.path_act.poses = self.path_act.poses[-300:]
        self.path_act.header.stamp = p.header.stamp
        self.act_pub.publish(self.path_act)

        # Logging
        if self.enable_logging:
            th_err = normalize_angle(ref_traj[2, 0] - self.th)
            cte = self.compute_cross_track_error(self.x, self.y, ref_traj)
            t_now = time.time()
            self.csv_writer.writerow([t_now, th_err, cte, cmd.linear.x, cmd.angular.z])

    def destroy_node(self) -> None:
        """Clean up resources."""
        self.get_logger().info("Shutting down MPC controller.")
        if self.enable_logging:
            self.csv_file.close()
        try:
            self.rt_csv.close()
        except Exception:
            pass
        super().destroy_node()

def main(args=None) -> None:
    """Run the node."""
    rclpy.init(args=args)
    node = MPCTrackingController()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()