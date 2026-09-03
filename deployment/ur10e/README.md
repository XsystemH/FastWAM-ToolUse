# UR10e staged TCP deployment

This directory implements the first, deliberately non-actuating deployment
stage for the `5090 -> industrial PC -> UR10e` setup. Its framing and request
field compatibility follow the scripts preserved under `references/`:

1. native `struct.Struct("Q")` payload length;
2. a pickled Python dictionary;
3. `image`/`images`/`img_history` and `qpos`/`qpos_history` request fields;
4. TCP port `9999` by default.

Because unpickling data can execute code, bind this service only on a trusted,
isolated robot-control network.

## Safety boundary

The initial implementation exposes only two modes:

- `image-check`: decode the received frame as RGB, save it on the 5090 and
  echo a JPEG preview to the probe client;
- `inference-dry-run`: run FastWAM, denormalize absolute 7-D actions, limit
  the response to one step by default and clip each joint target around the
  current qpos.

Both modes return `execute=false`. The server has no ROS/RTDE imports and does
not return the legacy `action`, `arm` or `gripper` keys. The included probe
client rejects any response that violates those conditions. There is no live
actuation mode in this commit.

## Image review first

Start the server from the repository root:

```bash
python -m deployment.ur10e.fastwam_server \
  --mode image-check \
  --host 0.0.0.0 \
  --port 9999 \
  --preview-dir runs/ur10e-image-check
```

Send one recorded or live-exported camera frame from a non-actuating machine:

```bash
python -m deployment.ur10e.safe_probe_client \
  --ip <5090-ip> \
  --image frame.jpg \
  --preview-out received-preview.jpg
```

Confirm the preview's camera selection, orientation and RGB colors before
starting dry-run model inference.

On the industrial PC, the live ROS probe subscribes to the same camera, joint
and gripper topics as the reference controller, but deliberately constructs no
command publisher:

```bash
python deployment/ur10e/ros_safe_client.py \
  --ip <5090-ip> \
  --mode image-check \
  --hz 2
```

Its window displays the local BGR camera frame beside the server's decoded RGB
round-trip. Press `q` to exit. It cannot command either the arm or the gripper.
Successful requests also emit `FASTWAM_SAFE_CLIENT_IMAGE_CHECK_OK execute=false`
for an auditable text-only acceptance record.
Like the preserved ROS reference client, `ros_safe_client.py` contains its TCP
framing and response handling directly, so that file can be copied and run by
itself in the industrial PC's existing ROS Python environment. It still needs
the ROS messages, `cv_bridge`, OpenCV and NumPy already used by the controller.

If ROS initialization or the first image request appears to stall, copy and run
the separate one-shot diagnostic instead:

```bash
python -u ros_safe_diagnostic.py --ip <5090-ip> --port 9999
```

It checks the runtime, hostname lookup, ROS master, node registration, the three
subscriptions, TCP connection, first image, JPEG encoding and image-check round
trip in order. An eight-second master/initialization stall prints its Python
thread stack and exits. It creates no publishers and never requests inference.

## Dry-run inference

```bash
python -m deployment.ur10e.fastwam_server \
  --mode inference-dry-run \
  --checkpoint /path/to/step_020000.pt \
  --dataset-stats /path/to/dataset_stats.json \
  --vae-path /existing/cache/Wan2.2_VAE.safetensors \
  --text-encoder-path /existing/cache/models_t5_umt5-xxl-enc-bf16.pth \
  --tokenizer-path /existing/cache/google/umt5-xxl \
  --task-config ur_robotiq_uncond_1cam224 \
  --model-config fastwam_joint \
  --max-response-steps 1 \
  --max-joint-delta-rad 0.05
```

The three component-path options are optional. When supplied, they are checked
before model construction and used directly, so a deployment host can reuse a
known local Wan cache without contacting ModelScope or Hugging Face.

Then submit a non-actuating probe with the current seven-value state:

```bash
python -m deployment.ur10e.safe_probe_client \
  --ip <5090-ip> \
  --mode inference-dry-run \
  --image frame.jpg \
  --qpos 0 0 0 0 0 0 0 \
  --prompt "pick up the cup"
```

The prediction is returned only as `predicted_action`, `predicted_arm` and
`predicted_gripper`. The live safe client prints the current qpos, finite
rate-limited prediction, clipping counts and inference latency under the marker
`FASTWAM_SAFE_CLIENT_INFERENCE_OK execute=false`. A future reviewed commit must
add the industrial-PC safety controller and any explicit actuation handoff.

## Static and protocol checks

```bash
python -m compileall -q deployment/ur10e
python -m deployment.ur10e.test_protocol
```
