#!/usr/bin/env python3
"""Non-actuating client for the staged FastWAM UR10e TCP server."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import recv_message, send_message


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": bool(np.isfinite(value).all()),
        }
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", required=True)
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("image-check", "inference-dry-run"),
        default="image-check",
    )
    parser.add_argument("--qpos", nargs=7, type=float)
    parser.add_argument("--prompt", default="pick up the cup")
    parser.add_argument(
        "--preview-out",
        type=Path,
        default=Path("runs/ur10e-received-preview.jpg"),
    )
    args = parser.parse_args()
    if args.mode == "inference-dry-run" and args.qpos is None:
        parser.error("inference-dry-run requires seven values after --qpos")
    return args


def main() -> None:
    args = parse_args()
    request: dict[str, Any] = {
        "image": args.image.read_bytes(),
        "color_space": "rgb",
        "prompt": args.prompt,
        "mode": args.mode,
    }
    if args.qpos is not None:
        request["qpos"] = np.asarray(args.qpos, dtype=np.float32)

    with socket.create_connection((args.ip, args.port), timeout=30.0) as conn:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        send_message(conn, request)
        response = recv_message(conn)

    if not isinstance(response, dict):
        raise RuntimeError(f"Unexpected response type: {type(response).__name__}")
    if response.get("execute") is not False:
        raise RuntimeError("Server response did not explicitly declare execute=false.")
    forbidden = {"action", "arm", "gripper"}.intersection(response)
    if forbidden:
        raise RuntimeError(f"Unsafe legacy action keys were returned: {sorted(forbidden)}")
    if not response.get("ok"):
        raise RuntimeError(str(response.get("error", "Unknown server error")))

    preview = response.get("preview_jpeg")
    if preview is not None:
        args.preview_out.parent.mkdir(parents=True, exist_ok=True)
        args.preview_out.write_bytes(preview)
        print(f"Saved RGB preview to {args.preview_out.resolve()}")
    print(json.dumps(_jsonable(response), indent=2, ensure_ascii=False))
    print("FASTWAM_UR10E_SAFE_PROBE_OK execute=false")


if __name__ == "__main__":
    main()
