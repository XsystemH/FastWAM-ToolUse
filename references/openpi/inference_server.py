import argparse
import dataclasses
import logging
import pickle
import socket
import struct
from collections import deque
from pathlib import Path

import cv2
import jax.numpy as jnp
import numpy as np

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.policies import policy as _policy
from openpi.shared import download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config

UR10E_ARM_DOF = 6
UR10E_DEPLOY_ACTION_DIM = 7
UR10E_LEGACY_ACTION_DIM = 8
UR10E_JOINT_ORDER = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def _broadcast_joint_state(state, actions):
    joint_state = state[..., :UR10E_ARM_DOF]
    while joint_state.ndim < actions[..., :UR10E_ARM_DOF].ndim:
        joint_state = np.expand_dims(joint_state, axis=-2)
    return joint_state


@dataclasses.dataclass(frozen=True)
class UR10eChunkOutputs(_transforms.DataTransformFn):
    """Convert padded OpenPI chunks to deployable UR10e chunks."""

    use_delta_joint_actions: bool = True

    def __call__(self, data):
        # OpenPI/pi0 uses a padded model action_dim, commonly 32. That is the
        # per-step action width, not the chunk length. Keep the full horizon and
        # slice only the deployable UR10e action width.
        actions = np.asarray(data["actions"][..., :UR10E_DEPLOY_ACTION_DIM], dtype=np.float32)
        if actions.shape[-1] != UR10E_DEPLOY_ACTION_DIM:
            raise ValueError(f"actions must have last dimension 7, got shape {actions.shape}")

        if self.use_delta_joint_actions:
            state = np.asarray(data["state"][..., :UR10E_DEPLOY_ACTION_DIM], dtype=np.float32)
            if state.shape[-1] != UR10E_DEPLOY_ACTION_DIM:
                raise ValueError(f"state must provide at least 7 values, got shape {data['state'].shape}")
            actions = actions.copy()
            actions[..., :UR10E_ARM_DOF] += _broadcast_joint_state(state, actions)

        return {"actions": actions}


def recv_exact(conn, size):
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = conn.recv(min(65536, remaining))
        if not chunk:
            raise ConnectionError("Socket closed while receiving payload.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def decode_image(encoded_or_array):
    if encoded_or_array is None:
        return None
    if isinstance(encoded_or_array, bytes):
        encoded_or_array = np.frombuffer(encoded_or_array, dtype=np.uint8)
    if isinstance(encoded_or_array, np.ndarray) and encoded_or_array.ndim == 1:
        image = cv2.imdecode(encoded_or_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image bytes.")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if isinstance(encoded_or_array, np.ndarray):
        image = encoded_or_array
        if image.ndim == 3 and image.shape[-1] == 3:
            return image.astype(np.uint8, copy=False)
    raise ValueError(f"Unsupported image payload type/shape: {type(encoded_or_array)}")


def as_float_array(value, name):
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        raise ValueError(f"`{name}` is empty.")
    return array


def format_ur10e_qpos(qpos):
    """Return [6 UR joints in GELLO/RTDE order, gripper]."""
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.shape[-1] == UR10E_DEPLOY_ACTION_DIM:
        return qpos
    if qpos.shape[-1] == UR10E_LEGACY_ACTION_DIM:
        return np.concatenate([qpos[..., :UR10E_ARM_DOF], qpos[..., -1:]], axis=-1)
    if qpos.shape[-1] == UR10E_ARM_DOF:
        gripper = np.zeros((*qpos.shape[:-1], 1), dtype=np.float32)
        return np.concatenate([qpos, gripper], axis=-1)
    raise ValueError(f"Unsupported UR10e qpos shape {qpos.shape}")


def format_ur10e_action(action):
    """Return [6 UR joint targets in GELLO/RTDE order, gripper]."""
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] < UR10E_DEPLOY_ACTION_DIM:
        raise ValueError(f"Expected at least 7 UR10e action values, got shape {action.shape}")
    return np.concatenate([action[..., :UR10E_ARM_DOF], action[..., -1:]], axis=-1)


def create_chunk_policy(train_config, checkpoint_dir, prompt=None, pytorch_device=None):
    """Create an OpenPI policy whose outputs are UR10e action chunks."""
    checkpoint_dir = Path(download.maybe_download(str(checkpoint_dir)))
    weight_path = checkpoint_dir / "model.safetensors"
    is_pytorch = weight_path.exists()

    logging.info("Loading OpenPI model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, str(weight_path))
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    use_delta = getattr(train_config.data, "use_delta_joint_actions", True)
    return _policy.Policy(
        model,
        transforms=[
            _transforms.InjectDefaultPrompt(prompt),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            UR10eChunkOutputs(use_delta_joint_actions=use_delta),
        ],
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )


def build_observation(data, default_prompt):
    images = []
    if "img_history" in data:
        images = [decode_image(item) for item in data["img_history"]]
    elif "images" in data:
        raw_images = data["images"]
        if isinstance(raw_images, dict):
            for key in sorted(raw_images):
                value = raw_images[key]
                if isinstance(value, (list, tuple)):
                    images.extend(decode_image(item) for item in value)
                else:
                    images.append(decode_image(value))
        else:
            images = [decode_image(item) for item in raw_images]
    elif "image" in data:
        images = [decode_image(data["image"])]

    images = [image for image in images if image is not None]
    if not images:
        raise ValueError("No image was provided. Expected `img_history`, `images`, or `image`.")

    if "qpos_history" in data:
        qpos = as_float_array(data["qpos_history"], "qpos_history")[-1]
    elif "qpos" in data:
        qpos = as_float_array(data["qpos"], "qpos")
    elif "arm_history" in data and "gripper_history" in data:
        arm = as_float_array(data["arm_history"], "arm_history")[-1].reshape(-1)
        gripper = as_float_array(data["gripper_history"], "gripper_history")[-1].reshape(-1)
        qpos = np.concatenate([arm, gripper], axis=0)
    else:
        arm = as_float_array(data.get("arm"), "arm").reshape(-1)
        gripper = as_float_array(data.get("gripper"), "gripper").reshape(-1)
        qpos = np.concatenate([arm, gripper], axis=0)

    prompt = data.get("prompt") or data.get("instruction") or default_prompt
    if not prompt:
        raise ValueError("No prompt/instruction provided and --prompt is empty.")

    # OpenPI UR10eInputs expects the original LeRobot-style keys. The training
    # data was converted from GELLO as qpos/actions in this order:
    # shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, gripper.
    return {
        "observation/image": images[-1],
        "observation/state": format_ur10e_qpos(qpos),
        "prompt": str(prompt),
    }


class OpenPIInferenceEngine:
    def __init__(
        self,
        config_name,
        checkpoint_dir,
        prompt=None,
        pytorch_device=None,
        return_chunk=False,
        replan_steps=None,
    ):
        train_config = _config.get_config(config_name)
        self.model_action_dim = int(train_config.model.action_dim)
        self.action_horizon = int(train_config.model.action_horizon)
        self.policy = create_chunk_policy(
            train_config,
            Path(checkpoint_dir),
            prompt=prompt,
            pytorch_device=pytorch_device,
        )
        self.prompt = prompt
        self.return_chunk = return_chunk
        self.replan_steps = None if replan_steps is None else max(1, int(replan_steps))
        self.action_queue = deque()

        print("--- OpenPI UR10e Inference Server ---")
        print(f"Config: {config_name}")
        print(f"Checkpoint: {checkpoint_dir}")
        print("UR10e joint order:", ", ".join(UR10E_JOINT_ORDER))
        print(
            f"OpenPI model chunk: action_horizon={self.action_horizon}, "
            f"padded action_dim={self.model_action_dim}"
        )
        print(f"Deploy chunk shape: ({self.action_horizon}, {UR10E_DEPLOY_ACTION_DIM})")
        print("Action format: [first 6 dims = joint targets, last dim = gripper]")
        if not return_chunk:
            if self.replan_steps is None:
                print("Chunk deployment: execute the full predicted action chunk before replanning")
            else:
                print(f"Chunk deployment: execute {self.replan_steps} actions before replanning")

    def infer(self, obs):
        if not self.action_queue:
            result = self.policy.infer(obs)
            actions = format_ur10e_action(result["actions"])
            if self.return_chunk:
                return actions
            chunk_steps = len(actions) if self.replan_steps is None else self.replan_steps
            if chunk_steps > len(actions):
                raise ValueError(
                    f"--replan-steps={chunk_steps} exceeds predicted chunk length {len(actions)}. "
                    f"This config action_horizon is {self.action_horizon}."
                )
            self.action_queue.extend(actions[:chunk_steps])

        return np.asarray(self.action_queue.popleft(), dtype=np.float32)

    def reset(self):
        self.action_queue.clear()
        if hasattr(self.policy, "reset"):
            self.policy.reset()


def split_action(action):
    action = format_ur10e_action(action)
    return action[..., :UR10E_ARM_DOF], action[..., -1:]


def main():
    parser = argparse.ArgumentParser(description="Socket inference server for finetuned OpenPI UR10e policies.")
    parser.add_argument("--config-name", default="pi05_ur10e_hw335_lora")
    parser.add_argument("--ckpt", "--checkpoint-dir", dest="checkpoint_dir", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--pytorch-device", default=None)
    parser.add_argument(
        "--return-chunk",
        action="store_true",
        help="Return the full OpenPI action chunk. Default returns one action per request.",
    )
    parser.add_argument(
        "--replan-steps",
        type=int,
        default=None,
        help="Number of actions to execute from each OpenPI chunk before querying again. Default uses the full chunk.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    engine = OpenPIInferenceEngine(
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
        prompt=args.prompt,
        pytorch_device=args.pytorch_device,
        return_chunk=args.return_chunk,
        replan_steps=args.replan_steps,
    )

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"OpenPI server online. Listening on {args.host}:{args.port}")

    header_struct = struct.Struct("Q")
    while True:
        conn, addr = server.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        engine.reset()
        print(f"Inference client connected: {addr}")
        try:
            while True:
                header = recv_exact(conn, header_struct.size)
                msg_size = header_struct.unpack(header)[0]
                data = pickle.loads(recv_exact(conn, msg_size))
                obs = build_observation(data, engine.prompt)
                action = engine.infer(obs)
                arm_action, gripper_action = split_action(action)
                response = {
                    "action": np.asarray(action, dtype=np.float32),
                    "arm": np.asarray(arm_action, dtype=np.float32),
                    "gripper": np.asarray(gripper_action, dtype=np.float32),
                }
                payload = pickle.dumps(response)
                conn.sendall(header_struct.pack(len(payload)) + payload)
        except (ConnectionError, EOFError):
            print(f"Inference client disconnected: {addr}")
        except Exception as exc:
            print(f"Inference error for {addr}: {exc}")
            try:
                payload = pickle.dumps({"error": str(exc)})
                conn.sendall(header_struct.pack(len(payload)) + payload)
            except Exception:
                pass
        finally:
            conn.close()


if __name__ == "__main__":
    main()
