#!/usr/bin/env python3
"""
Pixel-to-Odom Converter (ROS 2)
───────────────────────────────
* Subscribes
    • /predicted_pixels        (std_msgs/Float32MultiArray  – [u0,v0,u1,v1,…])
    • /a200_0000/platform/odom/filtered (nav_msgs/Odometry – zero-reset per segment)
    • /ai_enabled              (std_msgs/Bool – enables processing)
    • /rt_ctx_in               (std_msgs/String – AI timing context)        <-- NEW
* Publishes
    • /reference_waypoints     (geometry_msgs/PoseArray – frame_id matches odom topic)
    • /rt_ctx_mid              (std_msgs/String – ctx with p2w_end_ns)      <-- NEW
The ground is assumed perfectly flat (z = 0).
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from std_msgs.msg import Float32MultiArray, Bool, String   # <-- NEW String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose, Quaternion

import numpy as np
from math import sin, cos, atan2, pi
from scipy.interpolate import CubicSpline
import threading

# NEW:
import json, time


class PixelToOdomConverter(Node):
    # Constructor
    def __init__(self):
        super().__init__('pixel_to_odom_converter')

        # ── Fixed parameters (override here if needed) 
        fx, fy = 440.45, 440.45     # focal lengths   (px)
        cx, cy = 320.0, 240.0       # principal point (px)
        camera_height = 2.175        # metres
        pitch_deg = -45.0            # camera tilt (negative = down)
        self.num_waypoints = 10       # resampled output count

        # –– Intrinsics
        self.K_inv = np.linalg.inv(np.array([[fx, 0, cx],
                                             [0, fy, cy],
                                             [0,  0,  1.0]]))

        # –– Extrinsics (pitch then align camera Z→robot X)
        pitch = np.deg2rad(pitch_deg)
        R_pitch = np.array([[1, 0, 0],
                            [0,  cos(pitch), -sin(pitch)],
                            [0,  sin(pitch),  cos(pitch)]])
        R_align = np.array([[ 0, 0, 1],
                            [-1, 0, 0],
                            [ 0, -1, 0]])
        self.R_cam_robot = R_align @ R_pitch
        self.t_cam_robot = np.array([0.0, 0.0, camera_height])

        # –– Robot pose variables (updated from odom)
        self.robot_position = np.zeros(3)
        self.robot_yaw = 0.0
        self.robot_pose_received = False

        # –– AI enable flag & pixel buffer
        self.ai_enabled = False
        self.latest_pixels = None
        self.pixel_lock = threading.Lock()

        # –– QoS for /ai_enabled (latched flag)
        qos_transient = QoSProfile(depth=1,
                                   durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                   reliability=ReliabilityPolicy.RELIABLE)

        # Subscriptions
        self.create_subscription(Float32MultiArray,
                                 '/predicted_pixels',
                                 self.pixel_callback, 10)

        self.create_subscription(Odometry,
                                 'odom_local',
                                 self.odom_callback, 10)

        self.create_subscription(Bool,
                                 '/ai_enabled',
                                 self.ai_enabled_callback,
                                 qos_transient)

        # NEW: subscribe to AI timing context
        self.create_subscription(String,
                                 '/rt_ctx_in',
                                 self.ctx_in_cb, 10)
        self._ctx_latest = None

        # Publishers
        self.pub_waypoints = self.create_publisher(PoseArray,
                                                   '/reference_waypoints', 10)
        # NEW: republish context with p2w timestamp
        self.pub_ctx_mid = self.create_publisher(String,
                                                 '/rt_ctx_mid', 10)

        # Timer: convert & publish at 5 Hz
        self.create_timer(0.2, self.timer_callback)

        self.get_logger().info("🟢 Pixel-to-Odom Converter started.")

    # ---- NEW: context intake ----
    def ctx_in_cb(self, msg: String):
        try:
            self._ctx_latest = json.loads(msg.data)
        except Exception:
            self._ctx_latest = None

    #Callbacks 
    def ai_enabled_callback(self, msg: Bool):
        self.ai_enabled = msg.data
        state = "ENABLED" if self.ai_enabled else "DISABLED"
        self.get_logger().info(f"AI processing {state}")
        if not self.ai_enabled:                 # purge stale pixels
            with self.pixel_lock:
                self.latest_pixels = None

    def odom_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self.robot_position[:] = [pos.x, pos.y, pos.z]
        self.robot_yaw = self.quaternion_to_yaw(ori)
        self.robot_pose_received = True

    def pixel_callback(self, msg: Float32MultiArray):
        with self.pixel_lock:
            self.latest_pixels = msg

    #Main timer 
    def timer_callback(self):
        if not (self.ai_enabled and self.robot_pose_received):
            return

        # fetch and clear latest pixel batch
        with self.pixel_lock:
            msg = self.latest_pixels
            self.latest_pixels = None

        if msg is None or len(msg.data) % 2 != 0:
            return

        pixels = np.asarray(msg.data, dtype=float).reshape(-1, 2)

        # transform matrix robot->odom
        R_or = np.array([[cos(self.robot_yaw), -sin(self.robot_yaw), 0],
                         [sin(self.robot_yaw),  cos(self.robot_yaw), 0],
                         [0, 0, 1]])
        t_or = self.robot_position
        waypoints = []

        for idx, (u, v) in enumerate(pixels, start=1):
            ray_cam   = self.K_inv @ np.array([u, v, 1.0])
            ray_robot = self.R_cam_robot @ ray_cam
            if ray_robot[2] >= -1e-3:                       # above horizon – skip
                continue
            tau = -self.t_cam_robot[2] / ray_robot[2]       # ground intersection
            p_robot = self.t_cam_robot + tau * ray_robot
            p_odom  = R_or @ p_robot + t_or
            waypoints.append(p_odom)

        if not waypoints:
            self.get_logger().warn("No valid ground points in pixel batch.")
            return

        # Resample path
        waypoints = self.resample_waypoints(waypoints, self.num_waypoints)

        # Publish PoseArray
        pa = PoseArray()
        pa.header.frame_id = 'odom_local'
        pa.header.stamp = self.get_clock().now().to_msg()

        for i in range(len(waypoints)):
            pose = Pose()
            pose.position.x = float(waypoints[i][0])
            pose.position.y = float(waypoints[i][1])
            pose.position.z = 0.0

            if i < len(waypoints) - 1:                     # heading to next
                dx = waypoints[i+1][0] - waypoints[i][0]
                dy = waypoints[i+1][1] - waypoints[i][1]
                yaw = atan2(dy, dx)
            else:                                          # keep robot yaw
                yaw = self.robot_yaw
            pose.orientation = self.yaw_to_quaternion(yaw)
            pa.poses.append(pose)

        self.pub_waypoints.publish(pa)

        # ---- NEW: stamp pixel→odom end and forward context ----
        if self._ctx_latest is not None:
            ctx = dict(self._ctx_latest)  # shallow copy
            ctx["p2w_end_ns"] = time.monotonic_ns()
            out = String(); out.data = json.dumps(ctx)
            self.pub_ctx_mid.publish(out)
            self._ctx_latest = None  # consume once

    # Utilities 
    def resample_waypoints(self, pts, n):
        pts = np.array(pts)
        if len(pts) < 2:
            return pts

        d = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
        s = np.insert(np.cumsum(d), 0, 0.0)
        if s[-1] < 1e-6:
            return np.repeat(pts[:1], n, axis=0)

        t = s / s[-1]
        t_new = np.linspace(0, 1, n)
        csx, csy = CubicSpline(t, pts[:, 0]), CubicSpline(t, pts[:, 1])
        x, y = csx(t_new), csy(t_new)
        return np.column_stack([x, y, np.zeros_like(x)])

    @staticmethod
    def quaternion_to_yaw(q):
        return atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

    @staticmethod
    def yaw_to_quaternion(yaw):
        q = Quaternion()
        q.z = sin(yaw / 2.0)
        q.w = cos(yaw / 2.0)
        return q


# Main 
def main(args=None):
    rclpy.init(args=args)
    node = PixelToOdomConverter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
