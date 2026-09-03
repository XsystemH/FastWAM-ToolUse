#!/usr/bin/env python3
"""Continuous UR10e client matching the preserved reference control loop.

The ROS control behavior intentionally follows
``references/ur_rtde/ros_control_client_gello.py``.  Only the TCP request and
response fields are adapted to ``fastwam_server.py``.
"""

import argparse
import pickle
import socket
import struct
import time
from collections import deque

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from robotiq_2f_gripper_control.msg import (
    Robotiq2FGripper_robot_input,
    Robotiq2FGripper_robot_output,
)
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class URFastWAMControlClient:
    UR_ARM_JOINTS = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    def __init__(
        self,
        server_ip,
        port,
        prompt,
        image_topic,
        hz,
        history_size=3,
        jpeg_quality=80,
        speed_scale=1.0,
    ):
        rospy.init_node("ur_fast_wam_client")
        self.bridge = CvBridge()
        self.server_ip = server_ip
        self.port = int(port)
        self.prompt = prompt
        self.image_topic = image_topic
        self.hz_target = hz
        self.history_size = history_size
        self.jpeg_quality = jpeg_quality
        self.speed_scale = float(np.clip(speed_scale, 0.01, 1.0))
        if self.speed_scale != speed_scale:
            rospy.logwarn(
                f"Clipped --speed-scale from {speed_scale} to {self.speed_scale}. "
                "Expected a value in (0, 1]."
            )

        self.img_history = deque(maxlen=history_size)
        self.qpos_history = deque(maxlen=history_size)
        self.frame_times = deque(maxlen=10)

        self.live_frame = None
        self.latest_joints = None
        self.initial_pose = None
        self.current_gripper = 0.0
        self.state = "IDLE"

        self.last_rtt = 0.0
        self.actual_hz = 0.0

        self.joint_pub = rospy.Publisher(
            "/scaled_pos_joint_traj_controller/command",
            JointTrajectory,
            queue_size=1,
        )
        self.gripper_pub = rospy.Publisher(
            "/Robotiq2FGripperRobotOutput",
            Robotiq2FGripper_robot_output,
            queue_size=10,
        )

        rospy.Subscriber(image_topic, Image, self._cb_rgb)
        rospy.Subscriber("/joint_states", JointState, self._cb_joints)
        rospy.Subscriber(
            "/Robotiq2FGripperRobotInput",
            Robotiq2FGripper_robot_input,
            self._cb_gripper,
        )

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.header_struct = struct.Struct("Q")
        try:
            self.sock.connect((server_ip, self.port))
        except Exception as exc:
            rospy.logerr(f"Link failed: {exc}")
            raise SystemExit(1)

    def _cb_rgb(self, data):
        self.live_frame = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def _cb_joints(self, data):
        if len(data.name) == 0:
            return

        name_to_pos = dict(zip(data.name, data.position))
        if not all(name in name_to_pos for name in self.UR_ARM_JOINTS):
            return

        joints = [name_to_pos[name] for name in self.UR_ARM_JOINTS]
        self.latest_joints = joints
        if self.initial_pose is None:
            self.initial_pose = list(joints)

    def _cb_gripper(self, msg):
        self.current_gripper = float(msg.gPO) / 255.0

    def send_gripper(self, value):
        cmd = Robotiq2FGripper_robot_output()
        cmd.rACT = 1
        cmd.rGTO = 1
        cmd.rSP = int(np.clip(round(255 * self.speed_scale), 1, 255))
        cmd.rFR = 150
        cmd.rPR = int(np.clip(value, 0.0, 1.0) * 255)
        self.gripper_pub.publish(cmd)

    def draw_clean_ui(self, frame):
        h, _ = frame.shape[:2]
        panel = frame.copy()
        cv2.rectangle(panel, (0, 0), (420, 145), (15, 15, 15), -1)
        cv2.addWeighted(panel, 0.7, frame, 0.3, 0, frame)

        font = cv2.FONT_HERSHEY_DUPLEX
        status_color = (0, 255, 0) if self.state == "RUNNING" else (0, 200, 255)

        cv2.putText(frame, f"SYSTEM: {self.state}", (15, 30), font, 0.7, status_color, 1)
        cv2.putText(
            frame,
            f"FREQ:   {self.actual_hz:.1f} Hz",
            (15, 60),
            font,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            f"SPEED:  {self.speed_scale * 100:.0f}%",
            (15, 85),
            font,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            f"DELAY:  {self.last_rtt:.1f} ms",
            (15, 110),
            font,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            f"CAMERA: {self.image_topic}",
            (15, 135),
            font,
            0.45,
            (255, 255, 255),
            1,
        )

        buf_w = int((len(self.img_history) / float(self.history_size)) * 230)
        cv2.rectangle(frame, (15, 140), (245, 145), (50, 50, 50), -1)
        cv2.rectangle(frame, (15, 140), (15 + buf_w, 145), (0, 255, 0), -1)

        cv2.rectangle(frame, (0, h - 35), (frame.shape[1], h), (10, 10, 10), -1)
        cv2.putText(
            frame,
            "[S] START   [R] RESET   [Q] QUIT",
            (20, h - 12),
            font,
            0.5,
            (180, 180, 180),
            1,
        )

    def perform_reset(self, quitting=False):
        self.send_gripper(0.0)
        rospy.sleep(1.0)

        if self.initial_pose is not None:
            msg = JointTrajectory()
            msg.joint_names = list(self.UR_ARM_JOINTS)
            point = JointTrajectoryPoint()
            point.positions = list(self.initial_pose)
            point.time_from_start = rospy.Duration(4.0)
            msg.points.append(point)
            self.joint_pub.publish(msg)
            rospy.sleep(4.2)

        self.img_history.clear()
        self.qpos_history.clear()
        self.state = "IDLE"

    def _recv_exact(self, size):
        chunks = []
        remaining = size
        while remaining > 0:
            chunk = self.sock.recv(min(4096, remaining))
            if not chunk:
                raise ConnectionError("Socket closed while receiving response.")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _extract_action(self, response):
        if not isinstance(response, dict):
            raise TypeError(f"Expected FastWAM response dict, got {type(response).__name__}.")
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Unknown FastWAM server error."))
        if response.get("execute") is not False:
            raise RuntimeError("FastWAM server response must declare execute=false.")

        action = np.asarray(response.get("predicted_action"), dtype=np.float32)
        if action.ndim == 1:
            action = action[None, :]
        if action.ndim != 2 or action.shape[0] < 1 or action.shape[1] != 7:
            raise ValueError(f"Expected predicted_action shape [T,7], got {action.shape}.")
        if not np.isfinite(action).all():
            raise ValueError("predicted_action contains NaN or infinity.")

        first = action[0]
        return first[:6].astype(np.float32), float(np.clip(first[6], 0.0, 1.0))

    def _send_observation_and_get_action(self):
        payload = pickle.dumps(
            {
                "mode": "inference-dry-run",
                # Bytes avoid NumPy 2.x pickle module names on a NumPy 1.x ROS
                # workstation while representing the same encoded JPEG.
                "img_history": list(self.img_history),
                "qpos_history": list(self.qpos_history),
                "prompt": self.prompt,
                "requested_action_steps": 1,
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        self.sock.sendall(self.header_struct.pack(len(payload)) + payload)

        header = self._recv_exact(self.header_struct.size)
        msg_len = self.header_struct.unpack(header)[0]
        response = pickle.loads(self._recv_exact(msg_len))
        return self._extract_action(response)

    def _publish_arm_command(self, arm_action):
        msg = JointTrajectory()
        msg.joint_names = list(self.UR_ARM_JOINTS)
        point = JointTrajectoryPoint()
        point.positions = arm_action.tolist()
        point.time_from_start = rospy.Duration((1.5 / self.hz_target) / self.speed_scale)
        msg.points.append(point)
        self.joint_pub.publish(msg)

    def run(self):
        rate = rospy.Rate(self.hz_target)
        rospy.on_shutdown(lambda: self.perform_reset(quitting=True))

        while not rospy.is_shutdown():
            if self.live_frame is None or self.latest_joints is None:
                rate.sleep()
                continue

            cycle_start = time.perf_counter()

            ok, img_enc = cv2.imencode(
                ".jpg",
                self.live_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not ok:
                rate.sleep()
                continue

            self.img_history.append(img_enc.tobytes())
            qpos = list(self.latest_joints) + [self.current_gripper]
            self.qpos_history.append(qpos)

            if self.state == "RUNNING" and len(self.img_history) == self.history_size:
                try:
                    arm_action, gripper_action = self._send_observation_and_get_action()
                    self._publish_arm_command(arm_action)
                    self.send_gripper(gripper_action)
                    self.last_rtt = (time.perf_counter() - cycle_start) * 1000.0
                except Exception as exc:
                    rospy.logerr_throttle(1.0, f"Inference/execute failed: {exc}")
                    self.state = "IDLE"

            elif self.state == "RESETTING":
                self.perform_reset()

            self.frame_times.append(time.perf_counter())
            if len(self.frame_times) > 1:
                self.actual_hz = len(self.frame_times) / (
                    self.frame_times[-1] - self.frame_times[0]
                )

            display = self.live_frame.copy()
            self.draw_clean_ui(display)
            cv2.imshow("UR FastWAM Control Client", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                self.state = "RUNNING"
            elif key == ord("r"):
                self.state = "RESETTING"
            elif key == ord("q"):
                break

            rate.sleep()

        cv2.destroyAllWindows()
        self.sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.1.109")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--prompt", default="pick up the cup")
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--hz", type=int, default=30)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument(
        "--speed-scale",
        type=float,
        default=1.0,
        help=(
            "Execution speed multiplier for each full predicted action. "
            "Use 0.2 for 20%% speed; default 1.0 uses the original trajectory duration."
        ),
    )
    args = parser.parse_args()

    URFastWAMControlClient(
        args.ip,
        args.port,
        args.prompt,
        args.image_topic,
        args.hz,
        history_size=args.history_size,
        jpeg_quality=args.jpeg_quality,
        speed_scale=args.speed_scale,
    ).run()
