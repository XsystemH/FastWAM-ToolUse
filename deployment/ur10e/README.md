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

## Human-confirmed action-chunk execution

Keep `ros_safe_client.py` for image and inference review.  The separate
`ros_confirmed_step_client.py` is the only client in this directory that can
create command publishers, and only when explicitly enabled at startup:

```bash
python -u ros_confirmed_step_client.py \
  --ip <5090-ip> \
  --port 9999 \
  --prompt "pick up the cup" \
  --image-topic <third-person-camera-image-topic> \
  --action-steps 5 \
  --enable-step-execution \
  --hz 30 \
  --max-joint-delta-rad 0.05 \
  --max-gripper-delta 0.05 \
  --speed-scale 1.0
```

`--action-steps` accepts any positive integer, subject to the number of actions
produced by the model and the 5090 server's explicit safety cap.  For example,
use `--action-steps 10` to review and authorize ten actions.  Start the 5090
server with a cap at least as large as the requested chunk, such as
`--max-response-steps 10`.  The operator workflow is deliberately two-stage:

1. Press **I** to freeze the current observation, run one inference, and show
   the returned RGB preview and the configured number of client-limited
   candidate actions.  No command is published.
2. Inspect the image and printed targets, then press **E** once to arm that one
   pending chunk.  The main ROS loop publishes one target per iteration, just
   like the preserved reference client.  Press **X** to discard it instead.
3. After either E or X, the candidate is cleared.  Another movement requires a
   fresh I followed by a fresh E.  Inference is locked until the configured
   number of action cycles has been sent or interrupted.

After confirmation, the client follows the reference controller's loop and
trajectory timing: it publishes one arm-and-gripper target every `1 / hz`
seconds and gives each arm target a nominal duration of
`(1.5 / hz) / speed_scale`.  The reference default is 30 Hz, so at speed scale
1.0 a target is published every 0.033 seconds with a 0.05-second nominal
duration.

Do not compensate for subtle motion by raising the joint-delta limit before
verifying the published targets and measured joint-state change.

Holding or repeating E cannot execute another action chunk because the pending
candidate is consumed before ROS publication.  Execution never requests a
second inference automatically.  Omitting
`--enable-step-execution` leaves the same interface available for rehearsal but
does not create command publishers.

The default `--image-topic /camera/color/image_raw` must not be assumed to be
the desired view.  On the industrial PC, enumerate the currently published
image topics and pass the verified third-person topic explicitly.  At startup
the client prints both the selected topic and the first ROS `frame_id`; inspect
the displayed local image before pressing I.  If the displayed image is still
the wrist view, quit without inference or execution and correct the topic.

Wire messages use JPEG `bytes` and ordinary Python action lists rather than
pickled NumPy arrays.  This keeps a NumPy 1.x ROS workstation compatible with a
NumPy 2.x inference server and avoids `No module named numpy._core` during
response deserialization.

Pressing X or Q while a confirmed chunk is being sent drops only the targets
that have not yet been published.  It does not claim to cancel a target that
the ROS controller already accepted.

## Continuous reference-style execution

`ros_fast_wam.py` is the continuous counterpart for final deployment.  Its ROS
loop, S/R/Q state machine, publishers, reset behavior and trajectory timing
match `references/ur_rtde/ros_control_client_gello.py`.  The only protocol
adaptations are the FastWAM request mode, prompt, `predicted_action` response
key and JPEG bytes needed for cross-version NumPy compatibility.  Like the
reference server, the 5090 now runs FastWAM only when its per-connection action
queue is empty, caches the predicted chunk, and returns one queued action per
client cycle.  This removes a full model-inference pause between every pair of
published targets.

The continuous client defaults to executing the first 10 actions from each
prediction.  Use `--replan-steps 5` or `--replan-steps 10` on the industrial PC
to select the desired replan interval.  The server defaults to a hard ceiling
of 10 and can set a different ceiling with `--max-stream-replan-steps`.
The queue is cleared whenever the TCP client disconnects and a new client
connects.  `--max-response-steps` continues to govern only non-streaming
confirmed-chunk requests.

```bash
python -u ros_fast_wam.py \
  --ip <5090-ip> \
  --port 9999 \
  --prompt "pick up the cup" \
  --image-topic <verified-third-person-rect-topic> \
  --hz 30 \
  --history-size 3 \
  --jpeg-quality 80 \
  --replan-steps 10 \
  --speed-scale 1.0
```

The client starts in `IDLE`.  Press S to enter the same continuous
inference-and-execution loop as the reference client, R to run the reference
reset procedure, or Q to quit.  Like the reference implementation, this client
does not query `robot_program_running` or the controller manager before
publishing.  It also preserves the reference shutdown reset, which opens the
gripper and commands the initially observed arm pose.  Use the confirmed-step
client for image/action review before starting this continuous client.

## Static and protocol checks

```bash
python -m compileall -q deployment/ur10e
python -m deployment.ur10e.test_protocol
```
