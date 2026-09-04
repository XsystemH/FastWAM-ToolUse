"""Small standard-library test suite for the staged UR10e protocol."""

from __future__ import annotations

import ast
import io
import socket
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image

from deployment.ur10e.protocol import (
    decode_image,
    format_qpos,
    limit_absolute_actions,
    recv_message,
    send_message,
)
from deployment.ur10e.fastwam_server import StagedUR10eServer


def make_png() -> bytes:
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    output = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(output, format="PNG")
    return output.getvalue()


def make_args(preview_dir: str, mode: str) -> Namespace:
    return Namespace(
        mode=mode,
        preview_dir=preview_dir,
        prompt="pick up the cup",
        max_response_steps=1,
        max_stream_replan_steps=10,
        max_joint_delta_rad=0.05,
    )


def import_signature(parsed_tree):
    result = []
    for node in parsed_tree.body:
        if isinstance(node, ast.Import):
            result.append(("import", tuple((alias.name, alias.asname) for alias in node.names)))
        elif isinstance(node, ast.ImportFrom):
            result.append(
                (
                    "from",
                    node.level,
                    node.module,
                    tuple((alias.name, alias.asname) for alias in node.names),
                )
            )
    return result


class FakeEngine:
    def infer(self, image_rgb, qpos, prompt):
        del image_rgb, qpos, prompt
        return np.ones((32, 7), dtype=np.float32)


class ProtocolTest(unittest.TestCase):
    def test_joint_runtime_accepts_training_compile_override(self):
        runtime_path = Path(__file__).parents[2] / "src/fastwam/runtime.py"
        runtime_tree = ast.parse(runtime_path.read_text(encoding="utf-8"))
        factory = next(
            node
            for node in runtime_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "create_fastwam_joint"
        )
        argument_names = {argument.arg for argument in factory.args.args}
        self.assertIn("compile_training_denoise", argument_names)
        self.assertIn("vae_path", argument_names)
        self.assertIn("text_encoder_path", argument_names)
        self.assertIn("tokenizer_path", argument_names)

    def test_ros_client_matches_reference_imports_and_is_non_actuating(self):
        source_path = Path(__file__).with_name("ros_safe_client.py")
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        reference_path = Path(__file__).parents[2] / "references/ur_rtde/ros_control_client_gello.py"
        reference_tree = ast.parse(reference_path.read_text(encoding="utf-8"))

        self.assertEqual(import_signature(tree), import_signature(reference_tree))

        relative_imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level > 0
        ]
        self.assertEqual(relative_imports, [])

        function_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        self.assertTrue({"recv_exact", "recv_message", "send_message"} <= function_names)

        source = source_path.read_text(encoding="utf-8")
        for marker in (
            "standard_imports_ok",
            "reference_imports_ok",
            "ros_init_ok",
            "subscriptions_ok",
            "tcp_connect_ok",
            "first_image_ok",
            "FASTWAM_SAFE_CLIENT_REQUEST_OK",
            "FASTWAM_SAFE_CLIENT_IMAGE_CHECK_OK",
            "FASTWAM_SAFE_CLIENT_INFERENCE_OK",
        ):
            self.assertIn(marker, source)

        publisher_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Publisher"
        ]
        self.assertEqual(publisher_calls, [])

    def test_ros_diagnostic_matches_reference_imports_and_is_non_actuating(self):
        source_path = Path(__file__).with_name("ros_safe_diagnostic.py")
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        reference_path = Path(__file__).parents[2] / "references/ur_rtde/ros_control_client_gello.py"
        reference_tree = ast.parse(reference_path.read_text(encoding="utf-8"))

        self.assertEqual(import_signature(tree), import_signature(reference_tree))
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "Publisher"
                for node in ast.walk(tree)
            )
        )
        for marker in (
            "master_getpid_start",
            "master_getpid_ok",
            "ros_init_start",
            "ros_init_ok",
            "subscriptions_ok",
            "tcp_connect_ok",
            "first_image_ok",
            "request_sent",
            "response_ok",
            "FASTWAM_ROS_SAFE_DIAGNOSTIC_OK",
        ):
            self.assertIn(marker, source)

    def test_confirmed_client_has_two_gates_and_configurable_chunk(self):
        source_path = Path(__file__).with_name("ros_confirmed_step_client.py")
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        reference_path = Path(__file__).parents[2] / "references/ur_rtde/ros_control_client_gello.py"
        reference_tree = ast.parse(reference_path.read_text(encoding="utf-8"))

        self.assertEqual(import_signature(tree), import_signature(reference_tree))
        self.assertIn("--enable-step-execution", source)
        self.assertIn("--action-steps", source)
        self.assertIn('key == ord("i")', source)
        self.assertIn('key == ord("e")', source)
        self.assertIn('key == ord("x")', source)
        self.assertIn("FASTWAM_CONFIRMED_CHUNK_PREPARED execute=false", source)
        self.assertIn("FASTWAM_CONFIRMED_CHUNK_EXECUTED", source)
        self.assertIn("FASTWAM_CONFIRMED_CHUNK_STEP", source)
        self.assertIn("(1.5 / self.hz) / self.speed_scale", source)
        self.assertIn("self._execute_one_reference_cycle()", source)
        self.assertNotIn("rospy.Timer", source)
        self.assertNotIn("list_controllers", source)
        self.assertNotIn("robot_program_running", source)

        publisher_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Publisher"
        ]
        self.assertEqual(len(publisher_calls), 2)

        arm = source.index("self.execution_queue.extend", source.index("def _execute_pending_chunk"))
        consume = source.index("self.pending_action = None", source.index("def _execute_pending_chunk"))
        self.assertLess(consume, arm)

    def test_continuous_client_matches_reference_loop_and_fastwam_protocol(self):
        source_path = Path(__file__).with_name("ros_fast_wam.py")
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        reference_path = Path(__file__).parents[2] / "references/ur_rtde/ros_control_client_gello.py"
        reference_tree = ast.parse(reference_path.read_text(encoding="utf-8"))

        self.assertEqual(import_signature(tree), import_signature(reference_tree))
        for marker in (
            'self.state == "RUNNING"',
            'key == ord("s")',
            'key == ord("r")',
            'key == ord("q")',
            '"mode": "inference-dry-run"',
            '"requested_action_steps": 1',
            '"action_stream": True',
            '"stream_replan_steps": self.replan_steps',
            'response.get("predicted_action")',
            'response.get("execute") is not False',
            '(1.5 / self.hz_target) / self.speed_scale',
            'default=30',
            'default=10',
        ):
            self.assertIn(marker, source)
        self.assertNotIn("rospy.Timer", source)
        self.assertNotIn("list_controllers", source)
        self.assertNotIn("robot_program_running", source)

        publisher_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Publisher"
        ]
        self.assertEqual(len(publisher_calls), 2)

    def test_wire_round_trip(self):
        left, right = socket.socketpair()
        try:
            value = {"image": b"jpeg", "qpos": np.arange(7, dtype=np.float32)}
            send_message(left, value)
            received = recv_message(right)
            np.testing.assert_array_equal(received["qpos"], value["qpos"])
            self.assertEqual(received["image"], b"jpeg")
        finally:
            left.close()
            right.close()

    def test_encoded_image_is_rgb(self):
        decoded = decode_image(make_png())
        rgb = np.zeros((2, 3, 3), dtype=np.uint8)
        rgb[..., 0] = 255
        np.testing.assert_array_equal(decoded, rgb)

    def test_bgr_array_is_converted(self):
        bgr = np.asarray([[[3, 2, 1]]], dtype=np.uint8)
        decoded = decode_image(bgr, color_space="bgr")
        np.testing.assert_array_equal(decoded, np.asarray([[[1, 2, 3]]], dtype=np.uint8))

    def test_six_joint_qpos_adds_gripper(self):
        qpos = format_qpos(np.arange(6, dtype=np.float32))
        np.testing.assert_array_equal(qpos, np.asarray([0, 1, 2, 3, 4, 5, 0], dtype=np.float32))

    def test_action_is_truncated_and_rate_limited(self):
        current = np.zeros(7, dtype=np.float32)
        predicted = np.asarray(
            [[1.0, -1.0, 0.01, 0.0, 0.0, 0.0, 2.0], [2.0] * 7],
            dtype=np.float32,
        )
        safe = limit_absolute_actions(
            predicted,
            current_qpos=current,
            max_response_steps=1,
            max_joint_delta_rad=0.05,
        )
        self.assertEqual(safe.actions.shape, (1, 7))
        np.testing.assert_allclose(safe.actions[0, :6], [0.05, -0.05, 0.01, 0, 0, 0])
        self.assertEqual(float(safe.actions[0, -1]), 1.0)

    def test_action_chunk_is_rate_limited_sequentially(self):
        safe = limit_absolute_actions(
            np.ones((3, 7), dtype=np.float32),
            current_qpos=np.zeros(7, dtype=np.float32),
            max_response_steps=3,
            max_joint_delta_rad=0.05,
        )
        np.testing.assert_allclose(safe.actions[:, 0], [0.05, 0.10, 0.15])

    def test_nonfinite_action_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            limit_absolute_actions(
                np.asarray([[np.nan] * 7], dtype=np.float32),
                current_qpos=np.zeros(7, dtype=np.float32),
            )

    def test_image_check_response_cannot_execute(self):
        with tempfile.TemporaryDirectory() as preview_dir:
            server = StagedUR10eServer(make_args(preview_dir, "image-check"), None)
            response = server.handle_request({"image": make_png()})
            self.assertIs(response["execute"], False)
            self.assertFalse({"action", "arm", "gripper"}.intersection(response))
            self.assertTrue(Path(response["preview_path"]).is_file())
            self.assertGreater(len(response["preview_jpeg"]), 0)

    def test_dry_run_uses_non_executable_prediction_keys(self):
        with tempfile.TemporaryDirectory() as preview_dir:
            server = StagedUR10eServer(
                make_args(preview_dir, "inference-dry-run"),
                FakeEngine(),
            )
            response = server.handle_request(
                {
                    "image": make_png(),
                    "qpos": np.zeros(7, dtype=np.float32),
                    "prompt": "pick up the cup",
                }
            )
            self.assertIs(response["execute"], False)
            self.assertFalse({"action", "arm", "gripper"}.intersection(response))
            self.assertIsInstance(response["predicted_action"], list)
            self.assertEqual(np.asarray(response["predicted_action"]).shape, (1, 7))

    def test_dry_run_honors_requested_action_steps_under_server_cap(self):
        with tempfile.TemporaryDirectory() as preview_dir:
            args = make_args(preview_dir, "inference-dry-run")
            args.max_response_steps = 10
            server = StagedUR10eServer(args, FakeEngine())
            response = server.handle_request(
                {
                    "image": make_png(),
                    "qpos": np.zeros(7, dtype=np.float32),
                    "prompt": "pick up the cup",
                    "requested_action_steps": 5,
                }
            )
            self.assertIsInstance(response["predicted_action"], list)
            self.assertIsInstance(response["predicted_arm"], list)
            self.assertIsInstance(response["predicted_gripper"], list)
            self.assertEqual(np.asarray(response["predicted_action"]).shape, (5, 7))
            self.assertEqual(response["safety"]["returned_steps"], 5)
            self.assertEqual(response["safety"]["requested_action_steps"], 5)

    def test_dry_run_rejects_requested_action_steps_above_server_cap(self):
        with tempfile.TemporaryDirectory() as preview_dir:
            server = StagedUR10eServer(
                make_args(preview_dir, "inference-dry-run"),
                FakeEngine(),
            )
            with self.assertRaisesRegex(ValueError, "exceeds the server cap"):
                server.handle_request(
                    {
                        "image": make_png(),
                        "qpos": np.zeros(7, dtype=np.float32),
                        "prompt": "pick up the cup",
                        "requested_action_steps": 5,
                    }
                )

    def test_action_stream_infers_once_then_pops_cached_actions(self):
        with tempfile.TemporaryDirectory() as preview_dir:
            engine = FakeEngine()
            engine.calls = 0
            original_infer = engine.infer

            def counted_infer(*args):
                engine.calls += 1
                return original_infer(*args)

            engine.infer = counted_infer
            server = StagedUR10eServer(
                make_args(preview_dir, "inference-dry-run"),
                engine,
            )
            request = {
                "image": make_png(),
                "qpos": np.zeros(7, dtype=np.float32),
                "prompt": "pick up the cup",
                "requested_action_steps": 1,
                "action_stream": True,
                "stream_replan_steps": 5,
            }

            first = server.handle_request(request)
            second = server.handle_request(request)

            self.assertEqual(engine.calls, 1)
            self.assertTrue(first["action_stream"]["inference_performed"])
            self.assertFalse(second["action_stream"]["inference_performed"])
            self.assertEqual(first["action_stream"]["queue_remaining"], 4)
            self.assertEqual(second["action_stream"]["queue_remaining"], 3)
            np.testing.assert_allclose(first["predicted_action"][0][:6], [0.05] * 6)
            np.testing.assert_allclose(second["predicted_action"][0][:6], [0.10] * 6)

            for _ in range(3):
                server.handle_request(request)
            refilled = server.handle_request(request)
            self.assertEqual(engine.calls, 2)
            self.assertTrue(refilled["action_stream"]["inference_performed"])
            self.assertEqual(refilled["action_stream"]["chunk_id"], 2)

    def test_action_stream_replan_limit_and_connection_reset(self):
        with tempfile.TemporaryDirectory() as preview_dir:
            args = make_args(preview_dir, "inference-dry-run")
            args.max_stream_replan_steps = 2
            server = StagedUR10eServer(args, FakeEngine())
            request = {
                "image": make_png(),
                "qpos": np.zeros(7, dtype=np.float32),
                "prompt": "pick up the cup",
                "action_stream": True,
            }

            first = server.handle_request(request)
            self.assertEqual(first["action_stream"]["chunk_steps"], 2)
            self.assertEqual(first["action_stream"]["queue_remaining"], 1)
            server.reset_action_stream()
            restarted = server.handle_request(request)
            self.assertTrue(restarted["action_stream"]["inference_performed"])
            self.assertEqual(restarted["action_stream"]["chunk_id"], 2)

    def test_action_stream_request_can_select_ten_replan_steps(self):
        with tempfile.TemporaryDirectory() as preview_dir:
            server = StagedUR10eServer(
                make_args(preview_dir, "inference-dry-run"),
                FakeEngine(),
            )
            response = server.handle_request(
                {
                    "image": make_png(),
                    "qpos": np.zeros(7, dtype=np.float32),
                    "prompt": "pick up the cup",
                    "action_stream": True,
                    "stream_replan_steps": 10,
                }
            )
            self.assertEqual(response["action_stream"]["chunk_steps"], 10)
            self.assertEqual(response["action_stream"]["queue_remaining"], 9)

    def test_action_stream_rejects_replan_steps_above_server_cap(self):
        with tempfile.TemporaryDirectory() as preview_dir:
            server = StagedUR10eServer(
                make_args(preview_dir, "inference-dry-run"),
                FakeEngine(),
            )
            with self.assertRaisesRegex(ValueError, "exceeds the server cap"):
                server.handle_request(
                    {
                        "image": make_png(),
                        "qpos": np.zeros(7, dtype=np.float32),
                        "prompt": "pick up the cup",
                        "action_stream": True,
                        "stream_replan_steps": 11,
                    }
                )


if __name__ == "__main__":
    unittest.main()
