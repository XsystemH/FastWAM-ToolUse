#!/usr/bin/env python3
"""Live ROS camera/state probe that cannot publish commands to the UR10e."""

from __future__ import annotations

import argparse
import pickle
import socket
import struct
import time
from collections import deque
from typing import Any

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from robotiq_2f_gripper_control.msg import Robotiq2FGripper_robot_input
from sensor_msgs.msg import Image, JointState


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


def recv_exact(conn: socket.socket, size: int) -> bytes:
    """Receive exactly ``size`` bytes or fail on a closed connection."""
    chunks: list[bytes] = []
    remaining = int(size)
    while remaining > 0:
        chunk = conn.recv(min(65536, remaining))
        if not chunk:
            raise ConnectionError("Socket closed while receiving payload.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(conn: socket.socket) -> Any:
    """Receive one response using the reference client's native-Q framing."""
    header = recv_exact(conn, HEADER_STRUCT.size)
    payload_size = int(HEADER_STRUCT.unpack(header)[0])
    if payload_size <= 0 or payload_size > MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"Invalid payload size {payload_size}; allowed range is "
            f"1..{MAX_PAYLOAD_BYTES} bytes."
        )
    return pickle.loads(recv_exact(conn, payload_size))


def send_message(conn: socket.socket, value: Any) -> None:
    """Send one request using the reference client's native-Q framing."""
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(HEADER_STRUCT.pack(len(payload)) + payload)


class UR10eSafeProbeClient:
    """Read observations and predictions without creating command publishers."""

    def __init__(
        self,
        *,
        server_ip: str,
        port: int,
        mode: str,
        prompt: str,
        hz: float,
        history_size: int,
        jpeg_quality: int,
    ) -> None:
        rospy.init_node("fastwam_ur10e_safe_probe")
        self.bridge = CvBridge()
        self.mode = mode
        self.prompt = prompt
        self.hz = float(hz)
        self.history_size = int(history_size)
        self.jpeg_quality = int(np.clip(jpeg_quality, 1, 100))

        self.live_frame: np.ndarray | None = None
        self.preview_frame: np.ndarray | None = None
        self.latest_joints: list[float] | None = None
        self.current_gripper = 0.0
        self.img_history: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.qpos_history: deque[list[float]] = deque(maxlen=self.history_size)
        self.last_rtt_ms = 0.0
        self.last_status = "WAITING"

        # Deliberately no rospy.Publisher is constructed in this class.
        rospy.Subscriber("/camera/color/image_raw", Image, self._on_image)
        rospy.Subscriber("/joint_states", JointState, self._on_joints)
        rospy.Subscriber(
            "/Robotiq2FGripperRobotInput",
            Robotiq2FGripper_robot_input,
            self._on_gripper,
        )

        self.sock = socket.create_connection((server_ip, int(port)), timeout=30.0)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _on_image(self, message: Image) -> None:
        self.live_frame = self.bridge.imgmsg_to_cv2(message, "bgr8")

    def _on_joints(self, message: JointState) -> None:
        positions = dict(zip(message.name, message.position))
        if all(name in positions for name in JOINT_ORDER):
            self.latest_joints = [float(positions[name]) for name in JOINT_ORDER]

    def _on_gripper(self, message: Robotiq2FGripper_robot_input) -> None:
        self.current_gripper = float(message.gPO) / 255.0

    @staticmethod
    def _validate_response(response: Any) -> dict[str, Any]:
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

    def _request(self) -> dict[str, Any]:
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
        return response

    def _update_preview(self, response: dict[str, Any]) -> None:
        encoded = response.get("preview_jpeg")
        if encoded:
            self.preview_frame = cv2.imdecode(
                np.frombuffer(encoded, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
        if self.mode == "inference-dry-run":
            action = np.asarray(response.get("predicted_action"), dtype=np.float32)
            finite = bool(action.size and np.isfinite(action).all())
            self.last_status = f"DRY RUN action={tuple(action.shape)} finite={finite}"
        else:
            self.last_status = f"IMAGE CHECK shape={response.get('image_shape')} RGB"

    def _draw(self) -> None:
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

    def run(self) -> None:
        rate = rospy.Rate(self.hz)
        try:
            while not rospy.is_shutdown():
                if self.live_frame is None:
                    rate.sleep()
                    continue
                if self.mode == "inference-dry-run" and self.latest_joints is None:
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


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    UR10eSafeProbeClient(
        server_ip=args.ip,
        port=args.port,
        mode=args.mode,
        prompt=args.prompt,
        hz=args.hz,
        history_size=args.history_size,
        jpeg_quality=args.jpeg_quality,
    ).run()


if __name__ == "__main__":
    main()
