"""Wire-compatible helpers for the existing UR10e policy TCP protocol.

The protocol is intentionally compatible with the two preserved reference
scripts: a native ``Q`` length prefix followed by a pickled Python object.
Pickle is unsafe for untrusted peers, so this module must only be used on the
isolated robot-control network.
"""

from __future__ import annotations

import dataclasses
import io
import pickle
import socket
import struct
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ARM_DOF = 6
ACTION_DIM = 7
LEGACY_ACTION_DIM = 8
JOINT_ORDER = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
HEADER_STRUCT = struct.Struct("Q")
DEFAULT_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


@dataclasses.dataclass(frozen=True)
class ParsedObservation:
    image_rgb: np.ndarray
    qpos: np.ndarray | None
    prompt: str | None


@dataclasses.dataclass(frozen=True)
class SafeActionResult:
    actions: np.ndarray
    source_shape: tuple[int, ...]
    returned_steps: int
    clipped_joint_values: int
    clipped_gripper_values: int


def recv_exact(conn: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(size)
    while remaining > 0:
        chunk = conn.recv(min(65536, remaining))
        if not chunk:
            raise ConnectionError("Socket closed while receiving payload.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(
    conn: socket.socket,
    *,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
) -> Any:
    header = recv_exact(conn, HEADER_STRUCT.size)
    payload_size = int(HEADER_STRUCT.unpack(header)[0])
    if payload_size <= 0 or payload_size > max_payload_bytes:
        raise ValueError(
            f"Invalid payload size {payload_size}; allowed range is "
            f"1..{max_payload_bytes} bytes."
        )
    return pickle.loads(recv_exact(conn, payload_size))


def send_message(conn: socket.socket, value: Any) -> None:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(HEADER_STRUCT.pack(len(payload)) + payload)


def decode_image(value: Any, *, color_space: str = "rgb") -> np.ndarray:
    """Decode an encoded image or validate an HWC uint8 array as RGB."""
    if value is None:
        raise ValueError("Image payload is missing.")

    encoded: bytes | None = None
    if isinstance(value, (bytes, bytearray, memoryview)):
        encoded = bytes(value)
    elif isinstance(value, np.ndarray) and value.ndim == 1:
        encoded = value.astype(np.uint8, copy=False).tobytes()

    if encoded is not None:
        try:
            with Image.open(io.BytesIO(encoded)) as image:
                return np.asarray(image.convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            raise ValueError("Failed to decode encoded image payload.") from exc

    if not isinstance(value, np.ndarray) or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(
            "Image array must have shape [H,W,3], got "
            f"{type(value).__name__} {getattr(value, 'shape', None)}."
        )
    if value.dtype != np.uint8:
        raise ValueError(f"Image array must be uint8, got {value.dtype}.")

    normalized_color_space = str(color_space).strip().lower()
    if normalized_color_space == "rgb":
        return np.ascontiguousarray(value)
    if normalized_color_space == "bgr":
        return np.ascontiguousarray(value[..., ::-1])
    raise ValueError(f"Unsupported color_space={color_space!r}; expected 'rgb' or 'bgr'.")


def extract_latest_image(data: dict[str, Any]) -> np.ndarray:
    """Match the reference server's image-key precedence and select the latest."""
    raw_images: list[Any] = []
    if "img_history" in data:
        raw_images = list(data["img_history"])
    elif "images" in data:
        images = data["images"]
        if isinstance(images, dict):
            for key in sorted(images):
                value = images[key]
                raw_images.extend(value if isinstance(value, (list, tuple)) else [value])
        else:
            raw_images = list(images)
    elif "image" in data:
        raw_images = [data["image"]]

    raw_images = [item for item in raw_images if item is not None]
    if not raw_images:
        raise ValueError("Expected `img_history`, `images`, or `image`.")
    return decode_image(raw_images[-1], color_space=data.get("color_space", "rgb"))


def format_qpos(qpos: Any) -> np.ndarray:
    """Return [six UR joints in GELLO/RTDE order, gripper]."""
    array = np.asarray(qpos, dtype=np.float32).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError("qpos contains NaN or infinity.")
    if array.size == ACTION_DIM:
        return array
    if array.size == LEGACY_ACTION_DIM:
        return np.concatenate([array[:ARM_DOF], array[-1:]]).astype(np.float32)
    if array.size == ARM_DOF:
        return np.concatenate([array, np.zeros(1, dtype=np.float32)])
    raise ValueError(f"Expected 6, 7, or 8 qpos values, got shape {array.shape}.")


def extract_qpos(data: dict[str, Any]) -> np.ndarray:
    if "qpos_history" in data:
        history = np.asarray(data["qpos_history"], dtype=np.float32)
        if history.size == 0:
            raise ValueError("qpos_history is empty.")
        return format_qpos(history[-1])
    if "qpos" in data:
        return format_qpos(data["qpos"])
    if "arm_history" in data and "gripper_history" in data:
        arm = np.asarray(data["arm_history"], dtype=np.float32)[-1].reshape(-1)
        gripper = np.asarray(data["gripper_history"], dtype=np.float32)[-1].reshape(-1)
        return format_qpos(np.concatenate([arm, gripper]))
    if "arm" in data:
        arm = np.asarray(data["arm"], dtype=np.float32).reshape(-1)
        gripper = np.asarray(data.get("gripper", [0.0]), dtype=np.float32).reshape(-1)
        return format_qpos(np.concatenate([arm, gripper]))
    raise ValueError("Expected qpos/qpos_history or arm/gripper state fields.")


def parse_observation(
    data: Any,
    *,
    default_prompt: str | None = None,
    require_state: bool = True,
    require_prompt: bool = True,
) -> ParsedObservation:
    if not isinstance(data, dict):
        raise ValueError(f"Request must be a dict, got {type(data).__name__}.")
    image_rgb = extract_latest_image(data)
    qpos = extract_qpos(data) if require_state else None
    prompt = data.get("prompt") or data.get("instruction") or default_prompt
    if require_prompt and not prompt:
        raise ValueError("No prompt/instruction was provided.")
    return ParsedObservation(
        image_rgb=image_rgb,
        qpos=qpos,
        prompt=None if prompt is None else str(prompt),
    )


def encode_preview_jpeg(image_rgb: np.ndarray, *, quality: int = 90) -> bytes:
    output = io.BytesIO()
    Image.fromarray(image_rgb, mode="RGB").save(
        output,
        format="JPEG",
        quality=int(np.clip(quality, 1, 100)),
    )
    return output.getvalue()


def save_preview(image_rgb: np.ndarray, path: str | Path, *, quality: int = 95) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb, mode="RGB").save(
        output_path,
        format="JPEG",
        quality=int(np.clip(quality, 1, 100)),
    )
    return output_path


def limit_absolute_actions(
    actions: Any,
    *,
    current_qpos: Any,
    max_response_steps: int = 1,
    max_joint_delta_rad: float = 0.05,
) -> SafeActionResult:
    """Limit an absolute UR10e action chunk without making it executable.

    The six arm values are clipped sequentially around the current/previous
    joint target. The gripper value is clipped to [0, 1]. The caller must keep
    the result under non-executable response keys during the review phase.
    """
    raw = np.asarray(actions, dtype=np.float32)
    source_shape = tuple(raw.shape)
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.ndim != 2 or raw.shape[1] < ACTION_DIM:
        raise ValueError(f"Expected action shape [T,D>=7], got {source_shape}.")
    if not np.isfinite(raw).all():
        raise ValueError("Predicted actions contain NaN or infinity.")
    if max_response_steps <= 0:
        raise ValueError("max_response_steps must be positive.")
    if max_joint_delta_rad <= 0:
        raise ValueError("max_joint_delta_rad must be positive.")

    deploy = np.concatenate([raw[:, :ARM_DOF], raw[:, -1:]], axis=1)
    deploy = deploy[: min(max_response_steps, len(deploy))].copy()
    previous = format_qpos(current_qpos)[:ARM_DOF]
    clipped_joint_values = 0
    for index in range(len(deploy)):
        lower = previous - max_joint_delta_rad
        upper = previous + max_joint_delta_rad
        original = deploy[index, :ARM_DOF].copy()
        deploy[index, :ARM_DOF] = np.clip(original, lower, upper)
        clipped_joint_values += int(np.count_nonzero(deploy[index, :ARM_DOF] != original))
        previous = deploy[index, :ARM_DOF]

    original_gripper = deploy[:, -1].copy()
    deploy[:, -1] = np.clip(original_gripper, 0.0, 1.0)
    clipped_gripper_values = int(np.count_nonzero(deploy[:, -1] != original_gripper))
    return SafeActionResult(
        actions=deploy.astype(np.float32, copy=False),
        source_shape=source_shape,
        returned_steps=len(deploy),
        clipped_joint_values=clipped_joint_values,
        clipped_gripper_values=clipped_gripper_values,
    )
