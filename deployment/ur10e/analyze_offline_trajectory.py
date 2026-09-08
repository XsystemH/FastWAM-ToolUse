#!/usr/bin/env python3
"""Compare command reversals and chunk boundaries in an offline UR10e report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def reversal_ratio(actions: np.ndarray, threshold: float) -> tuple[float | None, list[float | None]]:
    motion = np.diff(actions[:, :6], axis=0)
    active = np.abs(motion) >= threshold
    reversals = (
        (np.sign(motion[1:]) != np.sign(motion[:-1]))
        & active[1:]
        & active[:-1]
    )
    eligible = active[1:] & active[:-1]

    def ratio(mask: np.ndarray, denominator: np.ndarray) -> float | None:
        count = int(np.count_nonzero(denominator))
        return float(np.count_nonzero(mask) / count) if count else None

    return ratio(reversals, eligible), [
        ratio(reversals[:, joint], eligible[:, joint]) for joint in range(6)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--active-threshold", type=float, default=1e-3)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    metrics_path = args.report / "metrics.json"
    trajectory_path = args.report / "trajectory.npz"
    with metrics_path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    with np.load(trajectory_path) as trajectory:
        predicted = trajectory["predicted_action"]
        ground_truth = trajectory["ground_truth_action"]

    pred_reversal, pred_per_joint = reversal_ratio(predicted, args.active_threshold)
    gt_reversal, gt_per_joint = reversal_ratio(ground_truth, args.active_threshold)
    replan_steps = int(metrics["replan_steps"])
    boundaries = np.arange(replan_steps, len(predicted), replan_steps)
    pred_boundary = np.max(
        np.abs(predicted[boundaries, :6] - predicted[boundaries - 1, :6]), axis=1
    )
    gt_boundary = np.max(
        np.abs(ground_truth[boundaries, :6] - ground_truth[boundaries - 1, :6]),
        axis=1,
    )
    top_indices = np.argsort(pred_boundary)[::-1][: args.top]
    top_boundaries = [
        {
            "stitched_step": int(boundaries[index]),
            "dataset_index": int(metrics["episode_start"] + boundaries[index]),
            "predicted_jump_maxabs": float(pred_boundary[index]),
            "ground_truth_jump_maxabs": float(gt_boundary[index]),
        }
        for index in top_indices
    ]

    output = {
        "report": str(args.report.resolve()),
        "active_threshold_rad": args.active_threshold,
        "predicted_reversal_ratio": pred_reversal,
        "ground_truth_reversal_ratio": gt_reversal,
        "excess_reversal_ratio": (
            None
            if pred_reversal is None or gt_reversal is None
            else pred_reversal - gt_reversal
        ),
        "predicted_reversal_ratio_per_arm_joint": pred_per_joint,
        "ground_truth_reversal_ratio_per_arm_joint": gt_per_joint,
        "predicted_boundary_jump_mean_maxabs": float(pred_boundary.mean()),
        "ground_truth_boundary_jump_mean_maxabs": float(gt_boundary.mean()),
        "predicted_boundary_jump_maxabs": float(pred_boundary.max()),
        "ground_truth_boundary_jump_maxabs": float(gt_boundary.max()),
        "top_predicted_boundaries": top_boundaries,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
