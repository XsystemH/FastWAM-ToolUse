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
    "arm_value_maxabs",
    "arm_unit_diagnostic",
    "elapsed_seconds",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    args = parser.parse_args()
    with args.metrics.open(encoding="utf-8") as handle:
        report = json.load(handle)
    compact = {key: report.get(key) for key in SUMMARY_KEYS}
    compact["first_replan"] = report.get("first_replan")
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
