#!/usr/bin/env python3
"""Evaluate one complete recorded UR10e episode without publishing actions.

This is a teacher-forced offline diagnostic: every replan uses the recorded
image and proprioception at that point.  Predictions are never fed to ROS or
used to synthesize the next observation.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/home/wbjsamuel/projects/FastWAM"))
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/media/wbjsamuel/data/lerobot-v21-ur-robotiq-6tasks/cup"),
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("/media/wbjsamuel/data/fastwam/ur-robotiq-6tasks/dataset_stats.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/home/wbjsamuel/Downloads/FastWAM-checkpoints/"
            "cup-270222/step_020000.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/wbjsamuel/Downloads/FastWAM-eval/cup-270222/episode-000"),
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--active-delta-threshold", type=float, default=1e-3)
    return parser.parse_args()


def require_inputs(args: argparse.Namespace) -> None:
    missing = [
        str(path)
        for path in (args.repo, args.dataset, args.stats, args.checkpoint)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing required offline inputs: {missing}")
    if args.episode < 0:
        raise ValueError("--episode must be non-negative.")
    if args.replan_steps <= 0:
        raise ValueError("--replan-steps must be positive.")
    if args.num_inference_steps <= 0:
        raise ValueError("--num-inference-steps must be positive.")


def build_config(args: argparse.Namespace):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(
        version_base="1.3", config_dir=str(args.repo / "configs")
    ):
        cfg = hydra.compose(
            config_name="train",
            overrides=["task=ur-robotiq-uncond-1cam224", "model=fastwam_joint"],
        )

    OmegaConf.update(cfg, "mixed_precision", "bf16", merge=False)
    OmegaConf.update(cfg, "data.train.dataset_dirs", [str(args.dataset)], merge=False)
    OmegaConf.update(cfg, "data.train.is_training_set", False, merge=False)
    OmegaConf.update(cfg, "data.train.pretrained_norm_stats", str(args.stats), merge=False)
    OmegaConf.update(cfg, "model.skip_dit_load_from_pretrain", True, merge=False)
    OmegaConf.update(cfg, "model.action_dit_pretrained_path", None, merge=False)
    return cfg


def add_batch_dim(sample: dict) -> dict:
    result = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.unsqueeze(0)
        elif key == "prompt" and isinstance(value, str):
            result[key] = [value]
        else:
            result[key] = value
    return result


def denormalize(processor, proprio_btd, pred_action, gt_action):
    action_meta = processor.shape_meta["action"]
    state_meta = processor.shape_meta["state"]
    state_btd = proprio_btd.detach().to(device="cpu", dtype=torch.float32)
    if state_btd.ndim == 2:
        state_btd = state_btd.unsqueeze(1)

    outputs = {}
    state_result = None
    for name, raw_action in (("pred", pred_action), ("gt", gt_action)):
        action_btd = raw_action.detach().to(device="cpu", dtype=torch.float32)
        if action_btd.ndim == 2:
            action_btd = action_btd.unsqueeze(0)
        batch = {"action": action_btd, "state": state_btd}
        batch = processor.action_state_merger.backward(batch)
        batch = processor.normalizer.backward(batch)
        merged = {
            "action": {
                meta["key"]: batch["action"][meta["key"]][0]
                for meta in action_meta
            },
            "state": {
                meta["key"]: batch["state"][meta["key"]][0]
                for meta in state_meta
            },
        }
        merged = processor.action_state_merger.forward(merged)
        outputs[name] = merged["action"].cpu().numpy()
        if state_result is None:
            state_result = merged["state"].cpu().numpy()
    return outputs["pred"], outputs["gt"], state_result


def episode_bounds(dataset, episode: int) -> tuple[int, int, int]:
    index = dataset.lerobot_dataset.episode_data_index
    episode_count = len(index["from"])
    if episode >= episode_count:
        raise ValueError(
            f"--episode={episode} is out of range; dataset has {episode_count} episodes."
        )
    return int(index["from"][episode]), int(index["to"][episode]), episode_count


def safe_mean(values: np.ndarray) -> float | None:
    return float(values.mean()) if values.size else None


def main() -> None:
    args = parse_args()
    require_inputs(args)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    print("FASTWAM_OFFLINE_EPISODE robot_io=false action_publish=false", flush=True)
    cfg = build_config(args)
    dataset = instantiate(cfg.data.train)
    processor = dataset.lerobot_dataset.processor
    start, stop, episode_count = episode_bounds(dataset, args.episode)
    episode_length = stop - start

    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda:0")
    model.load_checkpoint(str(args.checkpoint))
    model.eval()

    stitched_pred = []
    stitched_gt = []
    stitched_state = []
    rows = []
    started = time.perf_counter()
    indices = list(range(start, stop, args.replan_steps))

    print(
        "FASTWAM_OFFLINE_EPISODE_START "
        f"episode={args.episode}/{episode_count - 1} frames={episode_length} "
        f"replans={len(indices)} replan_steps={args.replan_steps}",
        flush=True,
    )

    for replan_index, dataset_index in enumerate(indices):
        sample = add_batch_dim(dataset[dataset_index])
        video = sample["video"][0]
        gt_action = sample["action"][0]
        proprio = sample["proprio"][0, 0].unsqueeze(0)
        kwargs = {
            "prompt": None,
            "input_image": video[:, 0].unsqueeze(0),
            "action_horizon": int(gt_action.shape[-2]),
            "num_video_frames": 1,
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "num_inference_steps": args.num_inference_steps,
            # Match the online server, which uses the configured fixed seed at
            # every replan rather than advancing it between chunks.
            "seed": args.seed,
            "rand_device": "cpu",
            "tiled": False,
            "compile_action_infer": False,
        }
        if sample.get("context") is not None:
            kwargs["context"] = sample["context"][0]
            kwargs["context_mask"] = sample["context_mask"][0]
        else:
            kwargs["prompt"] = sample["prompt"][0]

        one_started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            result = model.infer_action(**kwargs)
        pred, gt, state = denormalize(
            processor, proprio, result["action"], gt_action
        )
        take = min(args.replan_steps, stop - dataset_index, len(pred), len(gt))
        pred = pred[:take]
        gt = gt[:take]
        state0 = np.asarray(state[0], dtype=np.float32)
        diff = pred - gt
        stitched_pred.append(pred)
        stitched_gt.append(gt)
        stitched_state.append(np.repeat(state0[None, :], take, axis=0))
        row = {
            "replan_index": replan_index,
            "dataset_index": dataset_index,
            "steps": take,
            "mae": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(np.square(diff)))),
            "first_step_mae": float(np.mean(np.abs(diff[0]))),
            "seconds": time.perf_counter() - one_started,
        }
        rows.append(row)
        print("FASTWAM_OFFLINE_EPISODE_REPLAN " + json.dumps(row), flush=True)

    pred = np.concatenate(stitched_pred, axis=0)
    gt = np.concatenate(stitched_gt, axis=0)
    state = np.concatenate(stitched_state, axis=0)
    diff = pred - gt
    pred_delta = pred[:, :6] - state[:, :6]
    gt_delta = gt[:, :6] - state[:, :6]
    active = np.abs(gt_delta) >= args.active_delta_threshold
    direction_match = np.sign(pred_delta[active]) == np.sign(gt_delta[active])

    pred_motion = np.diff(pred[:, :6], axis=0)
    active_motion = np.abs(pred_motion) >= args.active_delta_threshold
    reversal_mask = (
        (np.sign(pred_motion[1:]) != np.sign(pred_motion[:-1]))
        & active_motion[1:]
        & active_motion[:-1]
    )
    reversal_denominator = active_motion[1:] & active_motion[:-1]

    summary = {
        "offline_only": True,
        "robot_io": False,
        "action_publish": False,
        "teacher_forced": True,
        "episode": args.episode,
        "episode_count": episode_count,
        "episode_start": start,
        "episode_stop": stop,
        "episode_length": episode_length,
        "evaluated_action_steps": int(len(pred)),
        "replan_steps": args.replan_steps,
        "replans": len(rows),
        "finite_ratio": float((np.isfinite(pred) & np.isfinite(gt)).mean()),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "mae_per_dim": np.mean(np.abs(diff), axis=0).tolist(),
        "direction_match_ratio": safe_mean(direction_match),
        "predicted_reversal_ratio": safe_mean(
            reversal_mask[reversal_denominator]
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "replan_metrics": rows,
    }

    np.savez_compressed(
        args.output / "trajectory.npz",
        predicted_action=pred,
        ground_truth_action=gt,
        recorded_state=state,
        difference=diff,
    )
    with (args.output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (args.output / "replans.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("FASTWAM_OFFLINE_EPISODE_OK " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
