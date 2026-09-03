#!/usr/bin/env python3
"""Staged FastWAM TCP server for UR10e image review and dry-run inference.

This initial server deliberately has no action-serving mode. It never imports
ROS/RTDE and never returns the legacy ``action``, ``arm`` or ``gripper`` keys.
Consequently the preserved robot client cannot execute its responses.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

if __package__:
    from .protocol import (
        ACTION_DIM,
        JOINT_ORDER,
        encode_preview_jpeg,
        limit_absolute_actions,
        parse_observation,
        recv_message,
        save_preview,
        send_message,
    )
else:
    from protocol import (  # type: ignore[no-redef]
        ACTION_DIM,
        JOINT_ORDER,
        encode_preview_jpeg,
        limit_absolute_actions,
        parse_observation,
        recv_message,
        save_preview,
        send_message,
    )

LOGGER = logging.getLogger("fastwam.ur10e.server")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FastWAMEngine:
    """Load a trained FastWAMJoint checkpoint for action-only inference."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        dataset_stats_path: str | Path,
        task_config: str,
        model_config: str,
        device: str,
        num_inference_steps: int,
        sigma_shift: float,
        seed: int | None,
        vae_path: str | None = None,
        text_encoder_path: str | None = None,
        tokenizer_path: str | None = None,
    ) -> None:
        import hydra
        import torch
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        from fastwam.utils.config_resolvers import register_default_resolvers

        self._torch = torch
        self._default_prompt = DEFAULT_PROMPT
        self.device = str(device)
        self.model_dtype = torch.bfloat16
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = float(sigma_shift)
        self.seed = seed

        checkpoint = Path(checkpoint_path).expanduser().resolve()
        stats = Path(dataset_stats_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        if not stats.is_file():
            raise FileNotFoundError(f"Dataset stats not found: {stats}")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is required for {self.device}, but torch reports it unavailable.")

        register_default_resolvers()
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with hydra.initialize_config_dir(
            version_base="1.3",
            config_dir=str(PROJECT_ROOT / "configs"),
        ):
            cfg = hydra.compose(
                config_name="train",
                overrides=[f"task={task_config}", f"model={model_config}"],
            )

        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        model_cfg.load_text_encoder = True
        model_cfg.skip_dit_load_from_pretrain = True
        model_cfg.action_dit_pretrained_path = None
        if vae_path is not None:
            model_cfg.vae_path = self._require_path(vae_path, "VAE")
        if text_encoder_path is not None:
            model_cfg.text_encoder_path = self._require_path(text_encoder_path, "text encoder")
        if tokenizer_path is not None:
            model_cfg.tokenizer_path = self._require_path(tokenizer_path, "tokenizer")
        self.model = instantiate(
            model_cfg,
            model_dtype=self.model_dtype,
            device=self.device,
        )
        self.model.load_checkpoint(str(checkpoint))
        self.model = self.model.to(self.device).eval()

        self.processor = instantiate(cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats)))
        self.image_height = int(cfg.data.train.video_size[0])
        self.image_width = int(cfg.data.train.video_size[1])
        self.action_horizon = int(cfg.data.train.num_frames) - 1
        if self.action_horizon <= 0:
            raise ValueError(f"Invalid action horizon derived from config: {self.action_horizon}")

        LOGGER.info(
            "Loaded FastWAM | checkpoint=%s stats=%s task=%s model=%s horizon=%d",
            checkpoint,
            stats,
            task_config,
            model_config,
            self.action_horizon,
        )

    @staticmethod
    def _require_path(value: str, label: str) -> str:
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{label} path not found: {path}")
        return str(path)

    def _normalize_state(self, qpos: np.ndarray):
        torch = self._torch
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("UR10e deployment requires one merged state key.")
        state_key = state_meta[0]["key"]
        batch = {
            "state": {
                state_key: torch.as_tensor(qpos, dtype=torch.float32).unsqueeze(0)
            }
        }
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        return batch["state"][state_key]

    def _denormalize_action(self, action, proprio) -> np.ndarray:
        torch = self._torch
        action_btd = action.detach().to(device="cpu", dtype=torch.float32)
        if action_btd.ndim == 2:
            action_btd = action_btd.unsqueeze(0)
        state_btd = proprio.detach().to(device="cpu", dtype=torch.float32)
        if state_btd.ndim == 2:
            state_btd = state_btd.unsqueeze(1)
        batch = {"action": action_btd, "state": state_btd}
        batch = self.processor.action_state_merger.backward(batch)
        batch = self.processor.normalizer.backward(batch)
        if self.processor.action_state_transforms is not None:
            for transform in reversed(self.processor.action_state_transforms):
                batch = transform.backward(batch)

        action_meta = self.processor.shape_meta["action"]
        merged = {
            "action": {meta["key"]: batch["action"][meta["key"]][0] for meta in action_meta},
            "state": {
                meta["key"]: batch["state"][meta["key"]][0]
                for meta in self.processor.shape_meta["state"]
            },
        }
        result = self.processor.action_state_merger.forward(merged)["action"]
        if result.ndim != 2 or result.shape[-1] < ACTION_DIM:
            raise ValueError(f"Unexpected denormalized action shape: {tuple(result.shape)}")
        return result.numpy().astype(np.float32, copy=False)

    def _prepare_image(self, image_rgb: np.ndarray):
        torch = self._torch
        resized = Image.fromarray(image_rgb, mode="RGB").resize(
            (self.image_width, self.image_height),
            resample=Image.BILINEAR,
        )
        array = np.asarray(resized, dtype=np.uint8).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=self.model_dtype)
        return tensor * (2.0 / 255.0) - 1.0

    def infer(self, image_rgb: np.ndarray, qpos: np.ndarray, prompt: str) -> np.ndarray:
        torch = self._torch
        image = self._prepare_image(image_rgb)
        proprio = self._normalize_state(qpos)
        formatted_prompt = self._default_prompt.format(task=prompt)
        kwargs: dict[str, Any] = {
            "prompt": formatted_prompt,
            "input_image": image,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": "",
            "text_cfg_scale": 1.0,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": "cpu",
            "tiled": False,
            "compile_action_infer": False,
        }
        if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            kwargs["num_video_frames"] = 1
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.device.startswith("cuda"),
        ):
            result = self.model.infer_action(**kwargs)
        return self._denormalize_action(result["action"], proprio)


class StagedUR10eServer:
    def __init__(self, args: argparse.Namespace, engine: FastWAMEngine | None) -> None:
        self.args = args
        self.engine = engine
        self.preview_dir = Path(args.preview_dir).expanduser().resolve()
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.request_count = 0

    def _preview_path(self) -> Path:
        self.request_count += 1
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        return self.preview_dir / f"frame-{timestamp}-{self.request_count:06d}.jpg"

    def handle_request(self, data: Any) -> dict[str, Any]:
        if self.args.mode == "image-check":
            obs = parse_observation(data, require_state=False, require_prompt=False)
            preview_path = save_preview(obs.image_rgb, self._preview_path())
            LOGGER.info(
                "FASTWAM_IMAGE_CHECK_OK request=%d shape=%s preview=%s execute=false",
                self.request_count,
                tuple(int(v) for v in obs.image_rgb.shape),
                preview_path,
            )
            return {
                "ok": True,
                "mode": "image_check",
                "execute": False,
                "image_shape": tuple(int(v) for v in obs.image_rgb.shape),
                "image_dtype": str(obs.image_rgb.dtype),
                "color_space": "RGB",
                "preview_path": str(preview_path),
                "preview_jpeg": encode_preview_jpeg(obs.image_rgb),
            }

        if self.engine is None:
            raise RuntimeError("Inference engine is unavailable.")
        obs = parse_observation(data, default_prompt=self.args.prompt)
        requested_steps = data.get("requested_action_steps", self.args.max_response_steps)
        if isinstance(requested_steps, bool) or not isinstance(requested_steps, int):
            raise ValueError("requested_action_steps must be an integer.")
        if requested_steps <= 0:
            raise ValueError("requested_action_steps must be positive.")
        if requested_steps > self.args.max_response_steps:
            raise ValueError(
                f"requested_action_steps={requested_steps} exceeds the server cap "
                f"of {self.args.max_response_steps}."
            )
        started = time.perf_counter()
        predicted = self.engine.infer(obs.image_rgb, obs.qpos, obs.prompt)
        safe = limit_absolute_actions(
            predicted,
            current_qpos=obs.qpos,
            max_response_steps=requested_steps,
            max_joint_delta_rad=self.args.max_joint_delta_rad,
        )
        return {
            "ok": True,
            "mode": "inference_dry_run",
            "execute": False,
            "image_shape": tuple(int(v) for v in obs.image_rgb.shape),
            "color_space": "RGB",
            "preview_jpeg": encode_preview_jpeg(obs.image_rgb),
            "action_semantics": "absolute_joint_target",
            "joint_order": JOINT_ORDER,
            # Keep the wire payload NumPy-version-neutral.  In particular,
            # NumPy 2 pickles reference ``numpy._core``, which cannot be
            # imported by older ROS workstations running NumPy 1.x.
            "predicted_action": safe.actions.tolist(),
            "predicted_arm": safe.actions[:, :6].tolist(),
            "predicted_gripper": safe.actions[:, -1:].tolist(),
            "safety": {
                "source_shape": safe.source_shape,
                "returned_steps": safe.returned_steps,
                "requested_action_steps": requested_steps,
                "server_max_response_steps": self.args.max_response_steps,
                "max_joint_delta_rad": self.args.max_joint_delta_rad,
                "clipped_joint_values": safe.clipped_joint_values,
                "clipped_gripper_values": safe.clipped_gripper_values,
            },
            "server_timing": {"infer_ms": (time.perf_counter() - started) * 1000.0},
        }

    def serve_forever(self) -> None:
        max_payload = int(self.args.max_payload_mib * 1024 * 1024)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            server.bind((self.args.host, self.args.port))
            server.listen(1)
            LOGGER.info(
                "FastWAM staged server listening on %s:%d mode=%s execute=false",
                self.args.host,
                self.args.port,
                self.args.mode,
            )
            while True:
                conn, addr = server.accept()
                with conn:
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    LOGGER.info("Client connected: %s", addr)
                    try:
                        while True:
                            request = recv_message(conn, max_payload_bytes=max_payload)
                            try:
                                response = self.handle_request(request)
                            except Exception as exc:
                                LOGGER.exception("Request failed for %s", addr)
                                response = {
                                    "ok": False,
                                    "mode": self.args.mode.replace("-", "_"),
                                    "execute": False,
                                    "error": str(exc),
                                }
                            send_message(conn, response)
                    except (ConnectionError, EOFError):
                        LOGGER.info("Client disconnected: %s", addr)
                    except Exception:
                        # Framing/unpickling errors make the current stream unsafe to
                        # reuse, but must not terminate the image-check server.
                        LOGGER.exception("Closing malformed client stream: %s", addr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("image-check", "inference-dry-run"),
        default="image-check",
        help="No mode in this initial implementation can authorize execution.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--preview-dir", default="runs/ur10e-image-check")
    parser.add_argument("--max-payload-mib", type=int, default=64)
    parser.add_argument("--checkpoint")
    parser.add_argument("--dataset-stats")
    parser.add_argument("--vae-path")
    parser.add_argument("--text-encoder-path")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--task-config", default="ur_robotiq_uncond_1cam224")
    parser.add_argument("--model-config", default="fastwam_joint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default="pick up the cup")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--sigma-shift", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-response-steps", type=int, default=1)
    parser.add_argument("--max-joint-delta-rad", type=float, default=0.05)
    args = parser.parse_args()
    if args.max_payload_mib <= 0:
        parser.error("--max-payload-mib must be positive")
    if args.max_response_steps <= 0:
        parser.error("--max-response-steps must be positive")
    if args.mode == "inference-dry-run" and not (args.checkpoint and args.dataset_stats):
        parser.error("inference-dry-run requires --checkpoint and --dataset-stats")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    engine = None
    if args.mode == "inference-dry-run":
        engine = FastWAMEngine(
            checkpoint_path=args.checkpoint,
            dataset_stats_path=args.dataset_stats,
            task_config=args.task_config,
            model_config=args.model_config,
            device=args.device,
            num_inference_steps=args.num_inference_steps,
            sigma_shift=args.sigma_shift,
            seed=args.seed,
            vae_path=args.vae_path,
            text_encoder_path=args.text_encoder_path,
            tokenizer_path=args.tokenizer_path,
        )
    StagedUR10eServer(args, engine).serve_forever()


if __name__ == "__main__":
    main()
