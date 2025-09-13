#!/usr/bin/env python3
"""
ROS 2 node: ResNet-18 pixel-waypoint predictor (Husky A200)
• Subscribes  /a200_0000/sensors/camera_0/color/image   sensor_msgs/Image
              /ai_enabled                               std_msgs/Bool (latched)
• Publishes   /predicted_pixels                         std_msgs/Float32MultiArray
              /debug/predicted_image                    sensor_msgs/Image
"""

import os, cv2, torch, torch.nn as nn, torchvision.transforms as T
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Bool, Float32MultiArray, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import json, time, uuid


# 1. Model (identical to training)
class CNNStateEstimator(nn.Module):
    """ResNet-18 backbone → conv+BN → shared FC → {way-points, scene-type}"""
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        res = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=None)
        self.features = nn.Sequential(*list(res.children())[:-2])          # layer4 output
        self.conv2d   = nn.Conv2d(512, 64, 3, 2, 1)
        self.bn       = nn.BatchNorm2d(64)
        self.relu     = nn.ReLU(inplace=True)
        self.flat     = nn.Flatten()

        with torch.no_grad():
            flat_sz = self._flat(torch.zeros(1, 3, 240, 320)).shape[1]

        self.fc_hidden = nn.Linear(flat_sz, 128)
        self.dropout   = nn.Dropout(dropout)
        self.fc_wp     = nn.Linear(128, 20)   # 10 × (u,v)
        self.fc_type   = nn.Linear(128, 3)    # 3-class (unused at runtime)

    def _flat(self, x):
        x = self.features(x)
        x = self.relu(self.bn(self.conv2d(x)))
        return self.flat(x)

    def forward(self, x):
        x = self._flat(x)
        x = self.dropout(self.fc_hidden(x))
        return self.fc_wp(x), self.fc_type(x)


# 2. Predictor node
class ResNetPixelPredictor(Node):
    ORIG_W, ORIG_H = 640, 480
    TGT_W,  TGT_H  = 320, 240
    SCALE_UP = torch.tensor([ORIG_W / TGT_W, ORIG_H / TGT_H] * 10,
                            dtype=torch.float32)

    def __init__(self):
        super().__init__('resnet_pixel_predictor')
        self.get_logger().info("Loading ResNet-18 waypoint regressor…")
        self.ai_enabled = False

        # QoS (latched) for /ai_enabled
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)

        self.create_subscription(Bool, '/ai_enabled',
                                 self._ai_cb, qos_profile=qos)

        # Model + weights
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = CNNStateEstimator().to(self.device)

        ckpt = os.path.join(get_package_share_directory('husky_navigation'),
                            'config', 'weights', 'best_model.pth')
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.model.eval()
        self.get_logger().info(f"Weights loaded: {ckpt}")

        # Pre-processing
        self.bridge = CvBridge()
        self.tfm = T.Compose([
            T.Resize((self.TGT_H, self.TGT_W)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        # I/O
        self.pix_pub  = self.create_publisher(Float32MultiArray,
                                              '/predicted_pixels', 10)
        self.dbg_pub  = self.create_publisher(Image,
                                              '/debug/predicted_image', 10)

        # Trace-context publisher (for runtime timing)
        self.pub_ctx = self.create_publisher(String, '/rt_ctx_in', 10)
        self._seq = 0

        self.create_subscription(Image,
                                 '/a200_0000/sensors/camera_0/color/image',
                                 self._img_cb, 10)

        self.get_logger().info("Node ready – wait for /ai_enabled == True")

    # callbacks
    def _ai_cb(self, msg: Bool):
        self.ai_enabled = msg.data
        self.get_logger().info(f"AI {'ENABLED' if msg.data else 'DISABLED'}")

    def _img_cb(self, msg: Image):
        if not self.ai_enabled:
            return
        try:
            # ---- time zero at image receipt & per-frame id ----
            t0 = time.monotonic_ns()
            self._seq += 1
            frame_uuid = uuid.uuid4().hex

            cv_bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            img_pil = T.ToPILImage()(cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB))
            inp = self.tfm(img_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred_wp, _ = self.model(inp)                       # discard logits
            arr = (pred_wp.cpu().squeeze() * self.SCALE_UP).numpy()

            self.pix_pub.publish(Float32MultiArray(data=arr.tolist()))

            # ---- stamp AI stage end & publish trace context ----
            ai_end = time.monotonic_ns()
            ctx = {
                "frame_uuid": frame_uuid,
                "seq": self._seq,
                "recv_t_ns": t0,
                "ai_end_ns": ai_end,
                "p2w_end_ns": 0,
                "mpc_end_ns": 0
            }
            ctx_msg = String(); ctx_msg.data = json.dumps(ctx)
            self.pub_ctx.publish(ctx_msg)

            # overlay for visual-debugging
            for i in range(10):
                u, v = map(int, arr[2*i:2*i+2])
                if 0 <= u < self.ORIG_W and 0 <= v < self.ORIG_H:
                    cv2.circle(cv_bgr, (u, v), 5, (0,0,255), -1)
                    cv2.putText(cv_bgr, str(i+1), (u+5, v-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255,255,255), 1)
            self.dbg_pub.publish(self.bridge.cv2_to_imgmsg(cv_bgr, 'bgr8'))

        except Exception as e:
            self.get_logger().error(f"_img_cb error: {e}")


# 3. main
def main(args=None):
    rclpy.init(args=args)
    node = ResNetPixelPredictor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
