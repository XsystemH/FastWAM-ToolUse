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
        max_joint_delta_rad=0.05,
    )


class FakeEngine:
    def infer(self, image_rgb, qpos, prompt):
        del image_rgb, qpos, prompt
        return np.ones((32, 7), dtype=np.float32)


class ProtocolTest(unittest.TestCase):
    def test_ros_client_matches_reference_imports_and_is_non_actuating(self):
        source_path = Path(__file__).with_name("ros_safe_client.py")
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        reference_path = Path(__file__).parents[2] / "references/ur_rtde/ros_control_client_gello.py"
        reference_tree = ast.parse(reference_path.read_text(encoding="utf-8"))

        def import_signature(parsed_tree):
            result = []
            for node in parsed_tree.body:
                if isinstance(node, ast.Import):
                    result.append(
                        ("import", tuple((alias.name, alias.asname) for alias in node.names))
                    )
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

        publisher_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Publisher"
        ]
        self.assertEqual(publisher_calls, [])

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
            self.assertEqual(response["predicted_action"].shape, (1, 7))


if __name__ == "__main__":
    unittest.main()
