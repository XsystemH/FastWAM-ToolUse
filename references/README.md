# Reference implementations

This directory preserves the two deployment scripts used to define the initial
UR10e TCP interface. They are snapshots for comparison only; production code
lives under `deployment/ur10e/`.

| Snapshot | Source | Revision observed |
| --- | --- | --- |
| `openpi/inference_server.py` | [`wbjsamuel/openpi_tooluse`](https://github.com/wbjsamuel/openpi_tooluse/blob/main/inference_server.py) | `460b6a6` |
| `ur_rtde/ros_control_client_gello.py` | [`wbjsamuel/UR-RTDE`](https://github.com/wbjsamuel/UR-RTDE/blob/main/test_scripts/ros_control_client_gello.py) | `04dad3d` |

The reference protocol uses a native-width `struct.Struct("Q")` length prefix
followed by a pickled Python object. It is intended only for a trusted,
isolated robot-control network.
