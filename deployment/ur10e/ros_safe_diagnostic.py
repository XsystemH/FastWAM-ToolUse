#!/usr/bin/env python3
"""One-shot ROS/TCP/image diagnostic with no robot command publishers."""

import argparse
import pickle
import socket
import struct
import time
from collections import deque

print("FASTWAM_DIAGNOSTIC_STAGE standard_imports_ok", flush=True)

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

print("FASTWAM_DIAGNOSTIC_STAGE reference_imports_ok", flush=True)


HEADER_STRUCT = struct.Struct("Q")
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


def recv_exact(conn, size):
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
    header = recv_exact(conn, HEADER_STRUCT.size)
    payload_size = int(HEADER_STRUCT.unpack(header)[0])
    if payload_size <= 0 or payload_size > MAX_PAYLOAD_BYTES:
        raise ValueError(f"Invalid response payload size: {payload_size}")
    return pickle.loads(recv_exact(conn, payload_size))


def send_message(conn, value):
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(HEADER_STRUCT.pack(len(payload)) + payload)


class SafeDiagnostic:
    def __init__(self):
        self.bridge = None
        self.image = None
        self.joints = None
        self.gripper = None

    def on_image(self, message):
        if self.image is None:
            self.image = self.bridge.imgmsg_to_cv2(message, "bgr8")
            print(
                "FASTWAM_DIAGNOSTIC_STAGE first_image_ok "
                f"shape={tuple(self.image.shape)} encoding=bgr8",
                flush=True,
            )

    def on_joints(self, message):
        if self.joints is None:
            self.joints = (list(message.name), list(message.position))
            print(
                "FASTWAM_DIAGNOSTIC_STAGE first_joint_state_ok "
                f"names={len(message.name)} positions={len(message.position)}",
                flush=True,
            )

    def on_gripper(self, message):
        if self.gripper is None:
            self.gripper = float(message.gPO) / 255.0
            print(
                "FASTWAM_DIAGNOSTIC_STAGE first_gripper_state_ok "
                f"value={self.gripper:.4f}",
                flush=True,
            )


def validate_response(response):
    if not isinstance(response, dict):
        raise RuntimeError(f"Unexpected response type: {type(response).__name__}")
    if response.get("execute") is not False:
        raise RuntimeError("Server response did not explicitly declare execute=false.")
    forbidden = {"action", "arm", "gripper"}.intersection(response)
    if forbidden:
        raise RuntimeError(f"Unsafe legacy action keys returned: {sorted(forbidden)}")
    if not response.get("ok"):
        raise RuntimeError(str(response.get("error", "Unknown server error")))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", required=True)
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--joint-topic", default="/joint_states")
    parser.add_argument(
        "--gripper-topic",
        default="/Robotiq2FGripperRobotInput",
    )
    parser.add_argument("--preview-out", default="fastwam_diagnostic_preview.jpg")
    return parser.parse_args()


def main():
    args = parse_args()
    runtime_os = __import__("os")
    runtime_sys = __import__("sys")
    fault_handler = __import__("faulthandler")
    fault_handler.enable()
    fault_handler.dump_traceback_later(8.0, repeat=False, exit=True)

    master_uri = runtime_os.environ.get("ROS_MASTER_URI", "http://localhost:11311")
    print(
        "FASTWAM_DIAGNOSTIC_STAGE runtime_ok "
        f"python={runtime_sys.executable} version={runtime_sys.version.split()[0]} "
        f"master={master_uri} ros_ip={runtime_os.environ.get('ROS_IP', '<unset>')} "
        f"ros_hostname={runtime_os.environ.get('ROS_HOSTNAME', '<unset>')}",
        flush=True,
    )

    hostname = socket.gethostname()
    print(
        f"FASTWAM_DIAGNOSTIC_STAGE hostname_resolve_start hostname={hostname}",
        flush=True,
    )
    try:
        addresses = sorted(
            {
                item[4][0]
                for item in socket.getaddrinfo(
                    hostname,
                    0,
                    socket.AF_UNSPEC,
                    socket.SOCK_STREAM,
                )
            }
        )
        print(
            f"FASTWAM_DIAGNOSTIC_STAGE hostname_resolve_ok addresses={addresses}",
            flush=True,
        )
    except Exception as exc:
        print(
            "FASTWAM_DIAGNOSTIC_STAGE hostname_resolve_error "
            f"type={type(exc).__name__} error={exc}",
            flush=True,
        )

    print(
        f"FASTWAM_DIAGNOSTIC_STAGE master_getpid_start uri={master_uri}",
        flush=True,
    )
    master_pid = rospy.get_master().getPid()
    print(
        f"FASTWAM_DIAGNOSTIC_STAGE master_getpid_ok pid={master_pid}",
        flush=True,
    )

    print("FASTWAM_DIAGNOSTIC_STAGE ros_init_start", flush=True)
    rospy.init_node("fastwam_safe_diagnostic", anonymous=True)
    print("FASTWAM_DIAGNOSTIC_STAGE ros_init_ok", flush=True)
    fault_handler.cancel_dump_traceback_later()

    diagnostic = SafeDiagnostic()
    diagnostic.bridge = CvBridge()
    rospy.Subscriber(args.image_topic, Image, diagnostic.on_image)
    rospy.Subscriber(args.joint_topic, JointState, diagnostic.on_joints)
    rospy.Subscriber(
        args.gripper_topic,
        Robotiq2FGripper_robot_input,
        diagnostic.on_gripper,
    )
    print(
        "FASTWAM_DIAGNOSTIC_STAGE subscriptions_ok "
        f"image={args.image_topic} joints={args.joint_topic} "
        f"gripper={args.gripper_topic}",
        flush=True,
    )

    print(
        f"FASTWAM_DIAGNOSTIC_STAGE tcp_connect_start server={args.ip}:{args.port}",
        flush=True,
    )
    conn = socket.create_connection((args.ip, args.port), timeout=args.timeout)
    conn.settimeout(args.timeout)
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print("FASTWAM_DIAGNOSTIC_STAGE tcp_connect_ok", flush=True)

    try:
        deadline = time.monotonic() + args.timeout
        while diagnostic.image is None and not rospy.is_shutdown():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"No image received from {args.image_topic} within {args.timeout}s."
                )
            time.sleep(0.1)

        ok, encoded = cv2.imencode(".jpg", diagnostic.image)
        if not ok:
            raise RuntimeError("OpenCV failed to encode the first camera image.")
        print(
            f"FASTWAM_DIAGNOSTIC_STAGE jpeg_encode_ok bytes={encoded.size}",
            flush=True,
        )

        send_message(
            conn,
            {
                "mode": "image-check",
                # Normalize OpenCV's usual [N, 1] output for wire compatibility.
                "image": encoded.reshape(-1),
                "color_space": "rgb",
            },
        )
        print("FASTWAM_DIAGNOSTIC_STAGE request_sent", flush=True)
        response = recv_message(conn)
        validate_response(response)
        print(
            "FASTWAM_DIAGNOSTIC_STAGE response_ok "
            f"mode={response.get('mode')} execute=false "
            f"image_shape={response.get('image_shape')}",
            flush=True,
        )

        preview_bytes = response.get("preview_jpeg")
        if preview_bytes:
            preview = cv2.imdecode(
                np.frombuffer(preview_bytes, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if preview is None or not cv2.imwrite(args.preview_out, preview):
                raise RuntimeError(f"Failed to save preview to {args.preview_out}")
            print(
                f"FASTWAM_DIAGNOSTIC_STAGE preview_saved path={args.preview_out}",
                flush=True,
            )
    finally:
        conn.close()

    print("FASTWAM_ROS_SAFE_DIAGNOSTIC_OK execute=false", flush=True)


if __name__ == "__main__":
    main()
