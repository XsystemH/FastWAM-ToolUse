#!/usr/bin/env python3
"""Live ROS camera/state probe that cannot publish commands to the UR10e."""

import argparse
import pickle
import socket
import struct
import time
from collections import deque

print("FASTWAM_SAFE_CLIENT_STAGE standard_imports_ok", flush=True)

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

print("FASTWAM_SAFE_CLIENT_STAGE reference_imports_ok", flush=True)


JOINT_ORDER = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
HEADER_STRUCT = struct.Struct("Q")
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


def recv_exact(conn, size):
    """Receive exactly ``size`` bytes or fail on a closed connection."""
    chunks = []
    remaining = int(size)
    while remaining > 0:
        chunk = conn.recv(min(65536, remaining))
        if not chunk:
            raise ConnectionError("Socket closed while receiving payload.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(conn):
    """Receive one response using the reference client's native-Q framing."""
    header = recv_exact(conn, HEADER_STRUCT.size)
    payload_size = int(HEADER_STRUCT.unpack(header)[0])
    if payload_size <= 0 or payload_size > MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"Invalid payload size {payload_size}; allowed range is "
            f"1..{MAX_PAYLOAD_BYTES} bytes."
        )
    return pickle.loads(recv_exact(conn, payload_size))


def send_message(conn, value):
    """Send one request using the reference client's native-Q framing."""
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(HEADER_STRUCT.pack(len(payload)) + payload)


class UR10eSafeProbeClient:
    """Read observations and predictions without creating command publishers."""

    def __init__(
        self,
        *,
        server_ip,
        port,
        mode,
        prompt,
        hz,
        history_size,
        jpeg_quality,
        image_topic,
        joint_topic,
        gripper_topic,
    ):
        print("FASTWAM_SAFE_CLIENT_STAGE ros_init_start", flush=True)
        rospy.init_node("fastwam_ur10e_safe_probe", anonymous=True)
        print("FASTWAM_SAFE_CLIENT_STAGE ros_init_ok", flush=True)
        self.bridge = CvBridge()
        self.mode = mode
        self.prompt = prompt
        self.hz = float(hz)
        self.history_size = int(history_size)
        self.jpeg_quality = int(np.clip(jpeg_quality, 1, 100))

        self.live_frame = None
        self.preview_frame = None
        self.latest_joints = None
        self.current_gripper = 0.0
        self.img_history = deque(maxlen=self.history_size)
        self.qpos_history = deque(maxlen=self.history_size)
        self.last_rtt_ms = 0.0
        self.last_status = "WAITING"
        self.image_topic = image_topic
        self.joint_topic = joint_topic
        self.gripper_topic = gripper_topic
        self._received_first_image = False
        self._waiting_for_image_logged = False

        # Deliberately no rospy.Publisher is constructed in this class.
        rospy.Subscriber(self.image_topic, Image, self._on_image)
        rospy.Subscriber(self.joint_topic, JointState, self._on_joints)
        rospy.Subscriber(
            self.gripper_topic,
            Robotiq2FGripper_robot_input,
            self._on_gripper,
        )
        print(
            "FASTWAM_SAFE_CLIENT_STAGE subscriptions_ok "
            f"image={self.image_topic} joints={self.joint_topic} "
            f"gripper={self.gripper_topic}",
            flush=True,
        )

        print(
            f"FASTWAM_SAFE_CLIENT_STAGE tcp_connect_start server={server_ip}:{int(port)}",
            flush=True,
        )
        self.sock = socket.create_connection((server_ip, int(port)), timeout=30.0)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("FASTWAM_SAFE_CLIENT_STAGE tcp_connect_ok", flush=True)

    def _on_image(self, message):
        self.live_frame = self.bridge.imgmsg_to_cv2(message, "bgr8")
        if not self._received_first_image:
            self._received_first_image = True
            print(
                "FASTWAM_SAFE_CLIENT_STAGE first_image_ok "
                f"shape={tuple(self.live_frame.shape)} encoding=bgr8",
                flush=True,
            )

    def _on_joints(self, message):
        positions = dict(zip(message.name, message.position))
        if all(name in positions for name in JOINT_ORDER):
            self.latest_joints = [float(positions[name]) for name in JOINT_ORDER]

    def _on_gripper(self, message):
        self.current_gripper = float(message.gPO) / 255.0

    @staticmethod
    def _validate_response(response):
        if not isinstance(response, dict):
            raise RuntimeError(f"Unexpected response type: {type(response).__name__}")
        if response.get("execute") is not False:
            raise RuntimeError("Server response did not explicitly declare execute=false.")
        forbidden = {"action", "arm", "gripper"}.intersection(response)
        if forbidden:
            raise RuntimeError(f"Unsafe legacy action keys were returned: {sorted(forbidden)}")
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error", "Unknown server error")))
        return response

    def _request(self):
        request = {
            "mode": self.mode,
            "img_history": list(self.img_history),
            "qpos_history": list(self.qpos_history),
            "prompt": self.prompt,
        }
        started = time.perf_counter()
        send_message(self.sock, request)
        response = self._validate_response(recv_message(self.sock))
        self.last_rtt_ms = (time.perf_counter() - started) * 1000.0
        print(
            "FASTWAM_SAFE_CLIENT_REQUEST_OK "
            f"mode={self.mode} rtt_ms={self.last_rtt_ms:.1f} execute=false",
            flush=True,
        )
        return response

    def _update_preview(self, response):
        encoded = response.get("preview_jpeg")
        if encoded:
            self.preview_frame = cv2.imdecode(
                np.frombuffer(encoded, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
        if self.mode == "inference-dry-run":
            action = np.asarray(response.get("predicted_action"), dtype=np.float32)
            if action.ndim != 2 or action.shape[1] != 7 or action.shape[0] < 1:
                raise RuntimeError(
                    f"Unexpected predicted_action shape: {tuple(action.shape)}"
                )
            if not np.isfinite(action).all():
                raise RuntimeError("predicted_action contains NaN or infinity")
            safety = response.get("safety")
            if not isinstance(safety, dict):
                raise RuntimeError("Inference response is missing safety metadata")
            current_qpos = (
                list(self.qpos_history[-1]) if self.qpos_history else None
            )
            print(
                "FASTWAM_SAFE_CLIENT_INFERENCE_OK "
                f"execute=false current_qpos={current_qpos} "
                f"predicted_action={action.round(6).tolist()} "
                f"returned_steps={safety.get('returned_steps')} "
                f"clipped_joints={safety.get('clipped_joint_values')} "
                f"clipped_gripper={safety.get('clipped_gripper_values')} "
                f"infer_ms={response.get('server_timing', {}).get('infer_ms')}",
                flush=True,
            )
            self.last_status = f"DRY RUN action={tuple(action.shape)} finite=True"
        else:
            print(
                "FASTWAM_SAFE_CLIENT_IMAGE_CHECK_OK "
                f"execute=false shape={response.get('image_shape')} "
                f"color_space={response.get('color_space')}",
                flush=True,
            )
            self.last_status = f"IMAGE CHECK shape={response.get('image_shape')} RGB"

    def _draw(self):
        if self.live_frame is None:
            return
        local = self.live_frame.copy()
        cv2.putText(
            local,
            "LOCAL BGR (camera)",
            (15, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (0, 255, 255),
            1,
        )
        display = local
        if self.preview_frame is not None:
            preview = cv2.resize(self.preview_frame, (local.shape[1], local.shape[0]))
            cv2.putText(
                preview,
                "SERVER RGB ROUND-TRIP",
                (15, 30),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (0, 255, 0),
                1,
            )
            display = np.concatenate([local, preview], axis=1)
        cv2.putText(
            display,
            f"{self.last_status} | RTT {self.last_rtt_ms:.1f} ms | EXECUTE DISABLED | Q quit",
            (15, display.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.imshow("FastWAM UR10e Safe Probe", display)

    def run(self):
        rate = rospy.Rate(self.hz)
        try:
            while not rospy.is_shutdown():
                if self.live_frame is None:
                    if not self._waiting_for_image_logged:
                        self._waiting_for_image_logged = True
                        print(
                            "FASTWAM_SAFE_CLIENT_WAITING "
                            f"reason=no_image topic={self.image_topic}",
                            flush=True,
                        )
                    rate.sleep()
                    continue
                if self.mode == "inference-dry-run" and self.latest_joints is None:
                    rospy.logwarn_throttle(
                        5.0,
                        "FASTWAM_SAFE_CLIENT_WAITING reason=no_joint_state "
                        f"topic={self.joint_topic}",
                    )
                    rate.sleep()
                    continue

                ok, encoded = cv2.imencode(
                    ".jpg",
                    self.live_frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                )
                if ok:
                    self.img_history.append(encoded)
                    if self.latest_joints is not None:
                        self.qpos_history.append(
                            list(self.latest_joints) + [self.current_gripper]
                        )
                    if len(self.img_history) == self.history_size:
                        try:
                            self._update_preview(self._request())
                        except Exception as exc:
                            self.last_status = f"ERROR: {exc}"
                            rospy.logerr_throttle(1.0, self.last_status)

                self._draw()
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                rate.sleep()
        finally:
            cv2.destroyAllWindows()
            self.sock.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", required=True)
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument(
        "--mode",
        choices=("image-check", "inference-dry-run"),
        default="image-check",
    )
    parser.add_argument("--prompt", default="pick up the cup")
    parser.add_argument("--hz", type=float, default=2.0)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--joint-topic", default="/joint_states")
    parser.add_argument(
        "--gripper-topic",
        default="/Robotiq2FGripperRobotInput",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    UR10eSafeProbeClient(
        server_ip=args.ip,
        port=args.port,
        mode=args.mode,
        prompt=args.prompt,
        hz=args.hz,
        history_size=args.history_size,
        jpeg_quality=args.jpeg_quality,
        image_topic=args.image_topic,
        joint_topic=args.joint_topic,
        gripper_topic=args.gripper_topic,
    ).run()


if __name__ == "__main__":
    main()
