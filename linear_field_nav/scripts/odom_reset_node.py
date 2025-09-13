#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster

class OdomResetNode(Node):
    def __init__(self):
        super().__init__('odom_reset_node')

        # Initialize TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribe to raw odometry
        self.sub_raw_odom = self.create_subscription(
            Odometry,
            '/a200_0000/platform/odom/filtered',
            self.odom_callback,
            10
        )

        # Subscribe to reset trigger
        self.sub_heartbeat = self.create_subscription(
            Bool,
            '/segment_heartbeat',
            self.heartbeat_callback,
            10
        )

        # Publisher for local odometry
        self.pub_odom_local = self.create_publisher(Odometry, '/odom_local', 10)

        # Timer for TF broadcasting (10 Hz)
        self.static_transform = None
        self.create_timer(0.1, self.broadcast_tf_timer_cb)

        # Internal state
        self._latest_raw_odom = None
        self._origin_x = 0.0
        self._origin_y = 0.0
        self._origin_yaw = 0.0
        self._have_origin = False
        self._last_reset_time = None

        self.get_logger().info("odom_reset_node started. Waiting for /segment_heartbeat...")

    def broadcast_tf_timer_cb(self):
        if self.static_transform is not None:
            self.static_transform.header.stamp = self.get_clock().now().to_msg()
            self.tf_broadcaster.sendTransform(self.static_transform)

    def heartbeat_callback(self, msg: Bool):
        """Handle reset trigger"""
        if msg.data and self._latest_raw_odom is not None:
            t_now = self.get_clock().now().nanoseconds * 1e-9
            if self._last_reset_time is None or (t_now - self._last_reset_time) > 1.0:
                raw = self._latest_raw_odom
                self._origin_x = raw.pose.pose.position.x
                self._origin_y = raw.pose.pose.position.y
                q = raw.pose.pose.orientation
                self._origin_yaw = self.quaternion_to_yaw(q)
                self._have_origin = True
                self._last_reset_time = t_now

                self.get_logger().info(
                    f"Reset triggered → new origin: x={self._origin_x:.3f}, y={self._origin_y:.3f}, yaw={self._origin_yaw:.3f}"
                )

    def odom_callback(self, raw_msg: Odometry):
        """Process raw odometry and publish local odometry"""
        self._latest_raw_odom = raw_msg

        # If no origin is set, pass through raw odometry
        if not self._have_origin:
            local = Odometry()
            local.header = raw_msg.header
            local.child_frame_id = raw_msg.child_frame_id
            local.pose = raw_msg.pose
            local.twist = raw_msg.twist
            self.pub_odom_local.publish(local)
            return

        # Extract raw pose
        px = raw_msg.pose.pose.position.x
        py = raw_msg.pose.pose.position.y
        q = raw_msg.pose.pose.orientation
        yaw = self.quaternion_to_yaw(q)

        # Compute local position (rotate by -origin_yaw)
        dx = px - self._origin_x
        dy = py - self._origin_y
        cos0 = math.cos(-self._origin_yaw)
        sin0 = math.sin(-self._origin_yaw)
        x_loc = dx * cos0 - dy * sin0
        y_loc = dx * sin0 + dy * cos0
        yaw_loc = self.normalize_angle(yaw - self._origin_yaw)

        # Create local odometry message
        local = Odometry()
        local.header.stamp = raw_msg.header.stamp
        local.header.frame_id = 'odom_local'
        local.child_frame_id = raw_msg.child_frame_id

        # Set local pose
        local.pose.pose.position.x = x_loc
        local.pose.pose.position.y = y_loc
        local.pose.pose.position.z = raw_msg.pose.pose.position.z
        local.pose.pose.orientation = self.yaw_to_quaternion(yaw_loc)

        # Copy twist directly (in child frame)
        local.twist = raw_msg.twist

        self.pub_odom_local.publish(local)

        # Update static transform from map to odom_local
        t = TransformStamped()
        t.header.stamp = raw_msg.header.stamp
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom_local'
        t.transform.translation.x = self._origin_x
        t.transform.translation.y = self._origin_y
        t.transform.translation.z = 0.0
        t.transform.rotation = self.yaw_to_quaternion(self._origin_yaw)
        self.static_transform = t

    @staticmethod
    def quaternion_to_yaw(q: Quaternion) -> float:
        """Convert quaternion to yaw angle"""
        norm = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
        if abs(norm - 1.0) > 1e-6:  # Fallback for invalid quaternions
            q.x, q.y, q.z, q.w = 0.0, 0.0, 0.0, 1.0
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def yaw_to_quaternion(yaw: float) -> Quaternion:
        """Convert yaw angle to quaternion"""
        q = Quaternion()
        half_yaw = yaw / 2.0
        q.z = math.sin(half_yaw)
        q.w = math.cos(half_yaw)
        q.x = 0.0
        q.y = 0.0
        return q

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    node = OdomResetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()