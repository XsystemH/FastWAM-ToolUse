#!/usr/bin/env python3
"""ROS client that requires a fresh human confirmation for every action chunk.

The server remains non-authoritative and returns ``execute=false``.  This client
may locally publish one rate-limited action chunk only when execution was
explicitly enabled at startup and the operator presses E after reviewing the
returned image and candidate actions.  The requested chunk length is
configurable.  The candidate is consumed before publication, so key repeat
cannot execute another chunk.
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
        raise ValueError(
            f"Invalid payload size {payload_size}; allowed range is "
            f"1..{MAX_PAYLOAD_BYTES} bytes."
        )
    return pickle.loads(recv_exact(conn, payload_size))


def send_message(conn, value):
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(HEADER_STRUCT.pack(len(payload)) + payload)


class UR10eConfirmedStepClient:
    def __init__(
        self,
        *,
        server_ip,
        port,
        prompt,
        hz,
        history_size,
        jpeg_quality,
        image_topic,
        joint_topic,
        gripper_topic,
        action_steps,
        enable_step_execution,
        max_joint_delta_rad,
        max_gripper_delta,
        max_state_drift_rad,
        max_pending_age,
        trajectory_duration,
        speed_scale,
    ):
        print("FASTWAM_CONFIRMED_STEP_STAGE ros_init_start", flush=True)
        rospy.init_node("fastwam_ur10e_confirmed_step", anonymous=True)
        print("FASTWAM_CONFIRMED_STEP_STAGE ros_init_ok", flush=True)

        self.bridge = CvBridge()
        self.prompt = prompt
        self.hz = float(hz)
        self.history_size = int(history_size)
        self.jpeg_quality = int(np.clip(jpeg_quality, 1, 100))
        self.action_steps = int(action_steps)
        self.execution_enabled = bool(enable_step_execution)
        self.max_joint_delta_rad = float(max_joint_delta_rad)
        self.max_gripper_delta = float(max_gripper_delta)
        self.max_state_drift_rad = float(max_state_drift_rad)
        self.max_pending_age = float(max_pending_age)
        self.trajectory_duration = float(trajectory_duration)
        self.speed_scale = float(speed_scale)

        self.live_frame = None
        self.preview_frame = None
        self.latest_joints = None
        self.current_gripper = 0.0
        self.img_history = deque(maxlen=self.history_size)
        self.qpos_history = deque(maxlen=self.history_size)
        self.pending_action = None
        self.pending_qpos = None
        self.pending_at = None
        self.last_rtt_ms = 0.0
        self.last_status = "WAITING FOR IMAGE/STATE"
        self.executing_until = 0.0
        self.gripper_timers = []

        rospy.Subscriber(image_topic, Image, self._on_image)
        rospy.Subscriber(joint_topic, JointState, self._on_joints)
        rospy.Subscriber(
            gripper_topic,
            Robotiq2FGripper_robot_input,
            self._on_gripper,
        )

        self.joint_pub = None
        self.gripper_pub = None
        if self.execution_enabled:
            self.joint_pub = rospy.Publisher(
                "/scaled_pos_joint_traj_controller/command",
                JointTrajectory,
                queue_size=1,
            )
            self.gripper_pub = rospy.Publisher(
                "/Robotiq2FGripperRobotOutput",
                Robotiq2FGripper_robot_output,
                queue_size=1,
            )

        print(
            "FASTWAM_CONFIRMED_STEP_STAGE execution_gate "
            f"enabled={str(self.execution_enabled).lower()} "
            f"action_steps={self.action_steps} "
            "prepare_key=I execute_key=E discard_key=X",
            flush=True,
        )
        print(
            f"FASTWAM_CONFIRMED_STEP_STAGE tcp_connect_start server={server_ip}:{int(port)}",
            flush=True,
        )
        self.sock = socket.create_connection((server_ip, int(port)), timeout=30.0)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("FASTWAM_CONFIRMED_STEP_STAGE tcp_connect_ok", flush=True)

    def _on_image(self, message):
        self.live_frame = self.bridge.imgmsg_to_cv2(message, "bgr8")

    def _on_joints(self, message):
        positions = dict(zip(message.name, message.position))
        if all(name in positions for name in JOINT_ORDER):
            self.latest_joints = np.asarray(
                [positions[name] for name in JOINT_ORDER], dtype=np.float32
            )

    def _on_gripper(self, message):
        self.current_gripper = float(message.gPO) / 255.0

    def _current_qpos(self):
        if self.latest_joints is None:
            return None
        return np.concatenate(
            [self.latest_joints, np.asarray([self.current_gripper], dtype=np.float32)]
        )

    def _validate_response(self, response):
        if not isinstance(response, dict):
            raise RuntimeError(f"Unexpected response type: {type(response).__name__}")
        if response.get("execute") is not False:
            raise RuntimeError("Server response must explicitly declare execute=false.")
        if {"action", "arm", "gripper"}.intersection(response):
            raise RuntimeError("Server returned unsafe legacy action keys.")
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error", "Unknown server error")))

        action = np.asarray(response.get("predicted_action"), dtype=np.float32)
        expected_shape = (self.action_steps, 7)
        if action.ndim != 2 or action.shape != expected_shape:
            raise RuntimeError(
                f"Confirmed-chunk mode requires exactly {expected_shape} actions; "
                f"got {tuple(action.shape)}."
            )
        if not np.isfinite(action).all():
            raise RuntimeError("predicted_action contains NaN or infinity.")
        safety = response.get("safety")
        if (
            not isinstance(safety, dict)
            or int(safety.get("returned_steps", 0)) != self.action_steps
            or int(safety.get("requested_action_steps", 0)) != self.action_steps
        ):
            raise RuntimeError(
                f"Server did not attest exactly {self.action_steps} returned actions."
            )
        return action, safety

    def _client_limit(self, targets, current):
        limited = np.asarray(targets, dtype=np.float32).copy()
        if limited.ndim != 2 or limited.shape[1] != 7:
            raise RuntimeError(f"Expected client action chunk [N,7], got {limited.shape}.")
        previous = np.asarray(current, dtype=np.float32).copy()
        for index in range(len(limited)):
            arm_delta = limited[index, :6] - previous[:6]
            limited[index, :6] = previous[:6] + np.clip(
                arm_delta,
                -self.max_joint_delta_rad,
                self.max_joint_delta_rad,
            )
            gripper_delta = float(limited[index, 6] - previous[6])
            limited[index, 6] = np.clip(
                previous[6]
                + np.clip(
                    gripper_delta,
                    -self.max_gripper_delta,
                    self.max_gripper_delta,
                ),
                0.0,
                1.0,
            )
            previous = limited[index]
        if not np.isfinite(limited).all():
            raise RuntimeError("Client-limited action chunk contains NaN or infinity.")
        return limited

    def _prepare_chunk(self):
        if self._chunk_is_executing():
            raise RuntimeError("The previously confirmed action chunk is still executing.")
        current = self._current_qpos()
        if self.live_frame is None or current is None:
            raise RuntimeError("Image and joint state must both be available.")
        if len(self.img_history) != self.history_size:
            raise RuntimeError(
                f"Observation history is not full ({len(self.img_history)}/{self.history_size})."
            )

        request = {
            "mode": "inference-dry-run",
            "img_history": list(self.img_history),
            "qpos_history": list(self.qpos_history),
            "prompt": self.prompt,
            "requested_action_steps": self.action_steps,
        }
        started = time.perf_counter()
        send_message(self.sock, request)
        response = recv_message(self.sock)
        self.last_rtt_ms = (time.perf_counter() - started) * 1000.0
        targets, safety = self._validate_response(response)

        encoded = response.get("preview_jpeg")
        if not encoded:
            raise RuntimeError("Inference response has no image preview for review.")
        preview = cv2.imdecode(
            np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if preview is None:
            raise RuntimeError("Server image preview could not be decoded.")

        self.preview_frame = preview
        self.pending_qpos = current.copy()
        self.pending_action = self._client_limit(targets, current)
        self.pending_at = time.monotonic()
        self.last_status = "PENDING: inspect image/action, then E execute or X discard"
        print(
            "FASTWAM_CONFIRMED_CHUNK_PREPARED execute=false "
            f"steps={self.action_steps} "
            f"current_qpos={current.round(6).tolist()} "
            f"pending_actions={self.pending_action.round(6).tolist()} "
            f"server_clipped_joints={safety.get('clipped_joint_values')} "
            f"server_clipped_gripper={safety.get('clipped_gripper_values')} "
            f"rtt_ms={self.last_rtt_ms:.1f}",
            flush=True,
        )

    def _discard_pending(self, reason):
        had_pending = self.pending_action is not None
        self.pending_action = None
        self.pending_qpos = None
        self.pending_at = None
        self.last_status = f"DISCARDED: {reason}"
        if had_pending:
            print(
                f"FASTWAM_CONFIRMED_STEP_DISCARDED reason={reason}", flush=True
            )

    def _chunk_is_executing(self):
        return time.monotonic() < self.executing_until

    def _publish_gripper_once(self, value):
        command = Robotiq2FGripper_robot_output()
        command.rACT = 1
        command.rGTO = 1
        command.rSP = int(np.clip(round(255 * self.speed_scale), 1, 255))
        command.rFR = 150
        command.rPR = int(round(np.clip(value, 0.0, 1.0) * 255))
        self.gripper_pub.publish(command)

    def _schedule_gripper_chunk(self, targets, step_duration):
        self.gripper_timers = []
        total_steps = len(targets)
        for index, value in enumerate(targets[:, 6], start=1):
            def publish_gripper(_event, target=float(value), step=index):
                self._publish_gripper_once(target)
                print(
                    "FASTWAM_CONFIRMED_CHUNK_GRIPPER "
                    f"step={step}/{total_steps} target={target:.6f}",
                    flush=True,
                )

            timer = rospy.Timer(
                rospy.Duration(index * step_duration),
                publish_gripper,
                oneshot=True,
            )
            self.gripper_timers.append(timer)

    def _execute_pending_chunk(self):
        if not self.execution_enabled:
            raise RuntimeError(
                "Execution is disabled; restart with --enable-step-execution."
            )
        if self.pending_action is None or self.pending_qpos is None:
            raise RuntimeError("No pending action; press I to infer first.")
        age = time.monotonic() - self.pending_at
        if age > self.max_pending_age:
            self._discard_pending("stale")
            raise RuntimeError(
                f"Pending action expired after {age:.2f}s; press I again."
            )

        current = self._current_qpos()
        if current is None:
            self._discard_pending("joint_state_missing")
            raise RuntimeError("Current joint state is unavailable.")
        drift = float(np.max(np.abs(current[:6] - self.pending_qpos[:6])))
        if drift > self.max_state_drift_rad:
            self._discard_pending("robot_moved_since_inference")
            raise RuntimeError(
                f"Robot drifted {drift:.6f} rad since inference; press I again."
            )
        if self.joint_pub.get_num_connections() < 1:
            raise RuntimeError("Arm command publisher has no controller subscriber.")
        if self.gripper_pub.get_num_connections() < 1:
            raise RuntimeError("Gripper publisher has no controller subscriber.")

        # Consume the candidate before the first publish.  A repeated E key has
        # nothing to execute, even if ROS publication below raises an exception.
        targets = self._client_limit(self.pending_action, current)
        self.pending_action = None
        self.pending_qpos = None
        self.pending_at = None

        trajectory = JointTrajectory()
        trajectory.joint_names = list(JOINT_ORDER)
        step_duration = self.trajectory_duration / self.speed_scale
        for index, target in enumerate(targets, start=1):
            point = JointTrajectoryPoint()
            point.positions = target[:6].tolist()
            point.time_from_start = rospy.Duration(index * step_duration)
            trajectory.points.append(point)
        self.joint_pub.publish(trajectory)
        self._schedule_gripper_chunk(targets, step_duration)
        total_duration = len(targets) * step_duration
        self.executing_until = time.monotonic() + total_duration

        self.last_status = (
            f"EXECUTING {len(targets)} ACTIONS; inference remains locked until complete"
        )
        print(
            "FASTWAM_CONFIRMED_CHUNK_EXECUTED "
            f"steps={len(targets)} targets={targets.round(6).tolist()} "
            f"step_duration_s={step_duration:.3f} total_duration_s={total_duration:.3f}",
            flush=True,
        )

    def _update_history(self):
        if self.live_frame is None:
            return
        current = self._current_qpos()
        if current is None:
            return
        ok, encoded = cv2.imencode(
            ".jpg",
            self.live_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if ok:
            self.img_history.append(encoded)
            self.qpos_history.append(current.tolist())

    def _draw(self):
        if self.live_frame is None:
            return
        local = self.live_frame.copy()
        cv2.putText(
            local,
            "LOCAL BGR CAMERA",
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
                "SERVER RGB PREVIEW",
                (15, 30),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (0, 255, 0),
                1,
            )
            display = np.concatenate([local, preview], axis=1)

        gate = "ENABLED" if self.execution_enabled else "DISABLED"
        cv2.rectangle(
            display,
            (0, display.shape[0] - 62),
            (display.shape[1], display.shape[0]),
            (10, 10, 10),
            -1,
        )
        cv2.putText(
            display,
            f"{self.last_status} | gate={gate} | RTT={self.last_rtt_ms:.1f}ms",
            (15, display.shape[0] - 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            display,
            f"[I] INFER/PREPARE   [E] EXECUTE {self.action_steps} ACTIONS   "
            "[X] DISCARD   [Q] QUIT",
            (15, display.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
        )
        cv2.imshow("FastWAM UR10e Confirmed Step", display)

    def run(self):
        rate = rospy.Rate(self.hz)
        try:
            while not rospy.is_shutdown():
                self._update_history()
                self._draw()
                key = cv2.waitKey(1) & 0xFF
                try:
                    if key == ord("i"):
                        self._discard_pending("superseded")
                        self._prepare_chunk()
                    elif key == ord("e"):
                        self._execute_pending_chunk()
                    elif key == ord("x"):
                        if self._chunk_is_executing():
                            raise RuntimeError(
                                "The already-published chunk cannot be discarded; "
                                "wait for it to finish."
                            )
                        self._discard_pending("operator")
                    elif key == ord("q"):
                        if self._chunk_is_executing():
                            raise RuntimeError(
                                "Quit is locked until the confirmed chunk finishes."
                            )
                        break
                except Exception as exc:
                    self.last_status = f"ERROR: {exc}"
                    rospy.logerr(str(exc))
                rate.sleep()
        finally:
            self._discard_pending("shutdown")
            for timer in self.gripper_timers:
                timer.shutdown()
            cv2.destroyAllWindows()
            self.sock.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", required=True)
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--prompt", default="pick up the cup")
    parser.add_argument("--hz", type=float, default=2.0)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--joint-topic", default="/joint_states")
    parser.add_argument(
        "--gripper-topic", default="/Robotiq2FGripperRobotInput"
    )
    parser.add_argument(
        "--action-steps",
        type=int,
        default=5,
        help="Number of actions in the one chunk authorized by each E press.",
    )
    parser.add_argument(
        "--enable-step-execution",
        action="store_true",
        help="Create command publishers; every step still requires I then E.",
    )
    parser.add_argument("--max-joint-delta-rad", type=float, default=0.05)
    parser.add_argument("--max-gripper-delta", type=float, default=0.05)
    parser.add_argument("--max-state-drift-rad", type=float, default=0.02)
    parser.add_argument("--max-pending-age", type=float, default=5.0)
    parser.add_argument("--trajectory-duration", type=float, default=0.75)
    parser.add_argument("--speed-scale", type=float, default=0.2)
    args = parser.parse_args()
    for name in (
        "hz",
        "max_joint_delta_rad",
        "max_gripper_delta",
        "max_state_drift_rad",
        "max_pending_age",
        "trajectory_duration",
        "speed_scale",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.speed_scale > 1.0:
        parser.error("--speed-scale must be at most 1.0")
    if args.history_size <= 0:
        parser.error("--history-size must be positive")
    if args.action_steps <= 0:
        parser.error("--action-steps must be positive")
    return args


def main():
    args = parse_args()
    UR10eConfirmedStepClient(
        server_ip=args.ip,
        port=args.port,
        prompt=args.prompt,
        hz=args.hz,
        history_size=args.history_size,
        jpeg_quality=args.jpeg_quality,
        image_topic=args.image_topic,
        joint_topic=args.joint_topic,
        gripper_topic=args.gripper_topic,
        action_steps=args.action_steps,
        enable_step_execution=args.enable_step_execution,
        max_joint_delta_rad=args.max_joint_delta_rad,
        max_gripper_delta=args.max_gripper_delta,
        max_state_drift_rad=args.max_state_drift_rad,
        max_pending_age=args.max_pending_age,
        trajectory_duration=args.trajectory_duration,
        speed_scale=args.speed_scale,
    ).run()


if __name__ == "__main__":
    main()
