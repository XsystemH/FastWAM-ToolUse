#!/usr/bin/env python3
"""Print the compact, decision-relevant portion of an offline episode report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUMMARY_KEYS = (
    "episode",
    "episode_length",
    "evaluated_action_steps",
    "replan_steps",
    "replans",
    "finite_ratio",
    "mae",
    "rmse",
    "direction_match_ratio",
    "predicted_reversal_ratio",
    "large_first_step_count",
    "predicted_first_delta_maxabs",
    "ground_truth_first_delta_maxabs",
    "boundary_jump_maxabs",
    "unchanged_replan_image_count",
    "mean_replan_image_change_mae",
    "arm_value_maxabs",
    "arm_unit_diagnostic",
    "elapsed_seconds",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument(
        "--include-first-replan",
        action="store_true",
        help="Also include the full first-replan vectors.",
    )
    args = parser.parse_args()
    with args.metrics.open(encoding="utf-8") as handle:
        report = json.load(handle)
    compact = {key: report.get(key) for key in SUMMARY_KEYS}
    if args.include_first_replan:
        compact["first_replan"] = report.get("first_replan")
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
