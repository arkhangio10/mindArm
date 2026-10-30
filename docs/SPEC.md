# MindArm specification — Step 2

Status: implemented walking skeleton plus its first real module, an optional Jetson-driven
arm (`arm: jetson`). Intent selection defaults to a keyboard-driven
simulation; an optional double-blink decoder selects one configured object and is eye
movement captured by electrodes, not brain control. This document
is the source of truth for interfaces, observable behavior, safety boundaries, and open
risks. Update it whenever a contract changes.

## Scope and acceptance

The laptop demo runs without hardware. `python -m mindarm` uses real Nemotron by default
and fake EEG/perception, keyboard intent selection, and a simulated arm. After **e** and
four displayed options, **2** selects index 1 and a validated motion plan executes.
`--planner fake` provides the same loop offline. Python 3.10+ is required.

With `arm: jetson`, the same loop moves the physical 3D-printed arm through a service on
the Jetson Nano, reached over SSH stdio. The simulator remains the configured default and
the only arm used by the default tests.

Out of scope: serial/Arduino integration, camera-based perception (the camera is only viewed
while teaching poses), object detectors, SSVEP/alpha decoding,
flickering UI, dashboards, web search, robot foundation models, motor imagery, inverse
kinematics, and servo position feedback.

## Architecture and files

`src/mindarm/interfaces.py` defines Pydantic v2 models and runtime-checkable Protocols.
`config.py` loads validated YAML. `factory.py` is the only implementation-selection point.
The orchestrator depends on Protocols, so future adapters replace fakes without changing
the orchestration contract. `session_log.py` owns JSONL writes; `eeg/recording.py` owns
recording validation and persistence. `__main__.py` handles setup and terminal cleanup.
`arm/jetson.py` is the laptop adapter for the real arm. `jetson/mindarm_arm.py` is the
service that runs on the Jetson. It is not part of the package and is deployed by copying
it to the Jetson's home directory. `teach.py` is the operator's pose teaching tool.
`bench.py` is the live planner benchmark; it calls Token Factory and no test runs it.

The default editable install locates `config/settings.yaml` relative to the source checkout.
A non-editable installation must use `--config` pointing to a retained checkout/configuration.
Relative fixture/log/recording paths resolve from the selected YAML file's directory.
`.env` is read from that directory's parent, without overriding exported environment values.

## Interfaces

All interfaces are synchronous. `Decoder.poll()` must return immediately. EEG reads are
bounded: in memory for the fake and replay sources, and at most the block duration plus 2 s
for the live stream source. `ArmController.execute()` may block for real motion (about
1–2 s per primitive). Perception/planner calls and arm execution run on one background
worker. `ArmController.cancel()` must return immediately. It makes an in-progress
`execute()` stop at the current position and return `success=False`; it is a no-op when idle.

```python
class EEGSource(Protocol):
    @property
    def sample_rate(self) -> float: ...
    @property
    def n_channels(self) -> int: ...
    def read(self, n_samples: int) -> np.ndarray: ...

class Decoder(Protocol):
    def poll(self) -> DecoderEvent | None: ...

class Perception(Protocol):
    def detect(self) -> Scene: ...

class Planner(Protocol):
    def propose_options(self, scene: Scene) -> list[Option]: ...
    def plan_motion(self, option: Option, scene: Scene) -> MotionPlan: ...

class ArmController(Protocol):
    def execute(self, plan: MotionPlan) -> ExecutionResult: ...
    def state(self) -> dict[str, float]: ...
    def cancel(self) -> None: ...

class ArmTransport(Protocol):
    def send(self, message: dict[str, Any]) -> None: ...
    def receive(self, timeout: float) -> dict[str, Any] | None: ...  # None on timeout
    def close(self) -> None: ...
```

### Shared data models

Models forbid extra fields, implicit coercion, NaN, and infinity. All indexes are zero-based;
displayed keys are one-based. Joint positions (`state()`, `joint_angles`, limits, and poses)
use the configured arm's unit: degrees for the simulator, and PWM pulse width in
microseconds (µs) for the Jetson arm. The YAML fields keep their `*_angle` names.

| Model | Fields and invariants |
| --- | --- |
| `DecoderEvent` | `kind`: SELECT / SELECT_TARGET / TOGGLE_ENGAGE / QUIT; `index`: 0..3 required only for SELECT |
| `DetectedObject` | Nonempty `label`; finite `x`, `y`, `confidence` in [0, 1] |
| `Scene` | `objects: list[DetectedObject]`; an empty scene is allowed |
| `Option` | Nonempty `label` (max 100 characters), `description` (max 500) |
| `OptionsResponse` | JSON envelope `options`; exactly four, distinct labels, exactly one spelled `Home` |
| `MotionPrimitive` | Closed `action`; optional `object_label` and `pose_name`, constrained as below |
| `MotionPlan` | `primitives`, an ordered list of 1..32 primitives |
| `ExecutionResult` | `success: bool`, `executed_primitives: int >= 0`, `joint_angles: dict[str, float]`, `error: str \| None` |

`QUIT` is the approved extension to the original two decoder event kinds so keyboard
shutdown is explicit within the same decoder contract.

### Motion vocabulary

| Primitive | Required arguments | Meaning in this step |
| --- | --- | --- |
| HOME | None | Return to local home pose |
| MOVE_ABOVE | `object_label` found in the current scene | Apply a local approach target; log the label |
| GRASP | None | Apply local grip target |
| LIFT | None | Apply local lift target |
| MOVE_TO_POSE | `pose_name`: `home` or `handover` | Apply a local named pose |
| RELEASE | None | Apply local release target |

Unrelated arguments and all other actions are rejected. Optional arguments may be omitted
or null where unused. Scene object membership is checked by `MotionPlan.validate_scene()`
at planner and orchestrator boundaries. Home must map to exactly one HOME primitive.

Example plan:

```json
{"primitives":[{"action":"MOVE_ABOVE","object_label":"bottle"},{"action":"GRASP"},{"action":"LIFT"},{"action":"MOVE_TO_POSE","pose_name":"handover"},{"action":"RELEASE"}]}
```

## Implementations

### EEG

`FakeEEGSource` defaults to one channel, 250 Hz sampling, a 12 Hz unit-amplitude sine, and
Gaussian noise with standard deviation 0.2. Rate, channels, frequency, noise, and seed are
configurable. Frequency must be below Nyquist. Phase and seeded noise remain continuous
across reads. Optional alpha bursts add amplitude-4, 10 Hz sine waves during seconds 4-6
of each six-second signal period. The units are synthetic and uncalibrated.

`read(n_samples)` requires a positive integer and returns finite float64 arrays shaped
`(n_channels, n_samples)`. Reads do not sleep. Acquisition is paced by the orchestrator
while engaged, using the source's actual sample rate and configured chunk size.

`ReplayEEGSource` uses `np.load(..., allow_pickle=False)`. Required NPZ entries:

- `data`: finite numeric, nonempty two-dimensional channels x samples array.
- `sample_rate`: finite positive numeric scalar.
- `n_channels`: positive integer scalar matching the first data dimension.

With looping enabled, reads wrap seamlessly even for requests longer than the recording.
With looping disabled, a read exceeding remaining samples raises `EOFError` without
advancing or returning a partial block. End-of-file disengages through the normal error path.

`StreamEEGSource` (`eeg: stream`) reads one channel in volts from the viewer's server-sent
event stream (`stream_url`, served by `eeg_view.py` outside this repository). It keeps at most
0.5 s of the newest samples, so reengaging never reads stale signal. Setup fails without
samples, and a lost or stalled stream raises through the normal error path. At `/console`, the
same viewer also starts one `--once` MindArm process per cycle, forwards the page's keys to it
(`/mindarm/key`), and publishes its states, options, and planner calls (`/mindarm/events`).

`EEGRecording` checks every block, retains copies in memory, and writes a compressed NPZ
on orderly shutdown. It creates no archive when recording is disabled or no samples were
captured. Acquisition resumes after reengagement; the stored signal omits idle wall time.
With `decoder: keyboard`, intent stays independent of the EEG signal.

### Keyboard and perception

Keyboard keys **1-4** produce SELECT(0..3); **e** toggles engagement; **q** produces QUIT.
EOF also quits; Ctrl+C triggers cleanup. Windows console input uses `msvcrt`; POSIX TTYs
use cbreak mode with restoration. Redirected input uses a background reader. Control
input is polled before state work on every tick. Selection outside SELECTING is ignored.

`BlinkDecoder` (`decoder: blink`) wraps the keyboard decoder, so **e** and **q** keep working
and take priority. `TappedEEGSource` hands every acquired block to `BlinkDetector`, so
detection runs only while engaged. The rule uses numpy only: notch at 60 and 120 Hz, a 1 s
moving-mean baseline, 40 ms smoothing, positive peaks above `blink_threshold_mv` and at most
0.4 s wide, and two peaks `blink_min_gap_s`..`blink_max_gap_s` apart, at most once per
`blink_refractory_s`. A gesture emits SELECT_TARGET and carries no index. Outside SELECTING it
is dropped. The signal is a voluntary blink, not brain activity, and neither raw samples nor
the trigger decision ever reach the planner.

`FakePerception` reloads and validates `fixtures/scene_table.json` per detection. The fixture
contains cup, bottle, and sports ball with normalized image coordinates and confidence.

### Planners

`FakePlanner` provides up to three unique object-pickup options plus Home, padding short
scenes with generic options. An object pickup yields MOVE_ABOVE, GRASP, LIFT,
MOVE_TO_POSE(handover), RELEASE; generic options yield HOME.

`NemotronPlanner` uses `openai.OpenAI` chat completions against the configured Nebius
endpoint. Defaults: `https://api.tokenfactory.us-central1.nebius.com/v1/` and
`nvidia/nemotron-3-super-120b-a12b`. The API key comes from `NEBIUS_API_KEY`.
Requests use temperature 0.1, a configurable token budget (default 4096), and request
timeout (default 30 seconds). Prompts include the schema, action constraints, and scene as
text. No images are sent to the planner.

Each operation has its own options (`planner_propose`, `planner_plan`):

- `strict_schema: true` sends `response_format={"type":"json_schema","json_schema":
  {"name": <operation>, "strict": true, "schema": ...}}`; `false` sends `{"type":"json_object"}`.
- `reasoning` sends `extra_body={"chat_template_kwargs":{"enable_thinking": <bool>}}`.
- Defaults: both operations without reasoning and with the strict schema, chosen from the
  benchmark in "Nemotron measurements".

Prompt content that the schema cannot carry: an object option means picking that object up
and handing it to the user, named exactly as in the scene, and its plan must be exactly
MOVE_ABOVE(object), GRASP, LIFT, MOVE_TO_POSE(handover), RELEASE. Before 2026-10-29 the
motion prompt only allowed that sequence, and live plans sometimes stopped at LIFT; HOME
afterwards then opens the gripper halfway and drops the object.

The plan schema is built per scene (`prompts.motion_schema`): `primitives` holds 1..32 items,
each matching one `anyOf` branch with `additionalProperties: false`. HOME, GRASP, LIFT, and
RELEASE take only `action`. MOVE_TO_POSE requires `pose_name` in `home`/`handover`. MOVE_ABOVE
requires `object_label` from an enum of the current scene's labels, and the branch is
omitted for an empty scene. The options schema fixes four items with bounded `label` and
`description` strings. "Exactly one Home" and distinct labels are not expressed in the
schema. The schema constrains what the service may return; local validation stays as the
second layer and is the one the safety argument relies on:

1. Validate JSON, Pydantic fields, exact option count/Home, and motion arguments.
2. Reject non-stop finish reasons (including token-truncated responses).
3. On invalid output, retry once with the validation error included in the conversation.
4. After the second invalid output, return Wait / Rest / Reset / Home options or `[HOME]`.
5. Home and those generic resting labels must map to `[HOME]` even after a later valid response.

SDK automatic retries are disabled. A transient failure (timeout, connection error, or HTTP
5xx) is retried once per operation with the same messages; this is not the repair attempt.
An operation therefore makes at most three calls (`MAX_CALLS`): two content attempts plus one
transient retry. With the 30 s timeout, the worst case is about 90 s before the operation
fails. Other API errors (authentication, other 4xx) and a second transient failure are
raised as `PlannerUnavailable`; the orchestrator returns HOME + IDLE. Errors log the API
exception type, not raw upstream bodies or credentials. Missing credentials fail startup.
Every API call logs latency and usage (null when the API did not supply it). Safe fallback
is explicit in logs.

### Simulated arm and limits

`config/joint_limits.yaml` defines six joints by default. The joint set is configurable,
with min/max degrees and maximum degrees/second for each joint. Both named poses must
specify all joints. Primitive targets can modify subsets of configured joints.

The simulation maps primitives to these local constants. MOVE_ABOVE does not perform
inverse kinematics or convert image coordinates into physical coordinates; different
object labels currently share the same approach target.

For each submitted plan, `execute()` revalidates its models and preflights all intermediate
joint targets before modifying any joint. It rejects an invalid submission with
`success=False`, zero executed primitives, unchanged angles, and an error. It does not clamp.
Speed is `abs(target - current) / primitive_duration_s`; default virtual duration is 2 s.
HOME may lengthen that duration to satisfy speed limits; it never bypasses angle limits.

After validation, execution logs each primitive and updates angles instantly using virtual
time. There are no motors or movement sleeps. `state()` returns a defensive copy.
`cancel()` is a no-op because virtual motion is instantaneous.

The orchestrator submits one primitive at a time, making disengagement possible between
primitives (and, on the Jetson arm, during one). Consequently a later primitive's joint
rejection can occur after earlier primitives completed; the cycle records partial progress,
then attempts HOME. Atomic preflight applies to each `execute()` submission, not the entire
orchestrated cycle.

### Jetson arm

Hardware: a PCA9685 at `0x40` on Jetson I2C bus 1, at 50 Hz (prescaler 121). Channels 0–5
are base, shoulder, elbow, wrist_pitch, wrist_roll, and gripper. The servos have no position
feedback. `config/jetson_arm.yaml` uses the simulator's schema in µs and µs/s. The service
knows exactly these six joints, and the configured joint set must match them.

`JetsonArm` (laptop) applies the simulator's preflight unchanged: every target within
limits, `abs(target - current) / duration <= max_speed`, no clamping, and whole-submission
rejection before any motion. It then sends each primitive as one `move` with integer
absolute targets for all joints and the checked duration. The service ramps linearly over
that duration, so the physical speed is the checked speed. After each reply, including
rejected or stopped moves, `state()` takes the positions the service reports. A lock
serializes `execute()` calls. `cancel()` sends `stop` and marks executions that started
before it as cancelled, so they return before sending further moves.

Wire protocol: one JSON object per line over the SSH exec channel's stdin/stdout. The
Jetson's ufw and its disabled SSH port forwarding stay unchanged.

| Direction | Message | Meaning |
| --- | --- | --- |
| Jetson → laptop | `{"id": 0, "ok": true, "stopped": false, "positions": {...}, "error": null}` | Launch report. A position is the last commanded µs, or `null` when the channel is released |
| laptop → Jetson | `{"cmd": "start", "id": n, "targets": {...}, "duration": s, "stagger": s}` | Energize released channels at their targets, then ramp the others |
| laptop → Jetson | `{"cmd": "move", "id": n, "targets": {...}, "duration": s}` | Ramp all six joints to the targets in `duration` |
| laptop → Jetson | `{"cmd": "stop"}` | Stop the current ramp at its present pulses. No reply of its own; ignored when idle |
| Jetson → laptop | `{"id": n, "ok": bool, "stopped": bool, "positions": {...}, "error": str \| null}` | Exactly one reply per `start`/`move` |

The service (`jetson/mindarm_arm.py`) runs on the Jetson's Python 3.6 with only
`python3-smbus`. It enforces its own hard envelope, independent of the laptop
configuration. That envelope is the calibration table below and a speed cap of 1000 µs/s.
It rejects with `ok: false` and no motion:

- requests whose targets are not exactly the six joints, or are not integers;
- targets outside the envelope, speeds above the cap, or durations outside (0, 10] s;
- a `stagger` outside [0, 2] s;
- a `move` while any channel is released.

A ramp writes changed pulses every 20 ms, each as one four-byte I2C transaction. Between
steps the service checks stdin: any received line or end of input stops the ramp at the
pulses already written and replies `stopped: true`. A move that changes nothing returns at once.

Startup and the first move have no position feedback. At launch the service adopts the
PCA9685 if it is awake at prescaler 121, and reads each channel's last commanded pulse.
Otherwise it releases all channels and configures 50 Hz. `JetsonArm` then sends `start`
with the HOME pose. The service energizes the released channels one at a time, in channel
order, directly at their HOME pulse, waiting 0.5 s between them. Each of those servos jumps
at its own maximum speed from an unknown position, so the operator places the arm near
HOME by hand before starting MindArm. Already-energized channels then ramp to HOME within
the configured speed limits. A startup failure fails setup with exit status 1.

Disconnection, chosen behavior "stop and hold": on end of input, SIGHUP, SIGTERM, or a
broken stdout, the service stops any ramp and exits without releasing. The PCA9685 keeps
generating the last pulses, so the arm holds its posture. A normal quit commands HOME first.
After MindArm exits, the arm therefore stays energized at HOME, and nothing moves while the
link is down. Releasing is manual: cut the servo supply, or run `servo_test.py`, which
releases every channel when it starts (the shoulder and elbow then fall).

The SSH client uses keep-alives (2 s × 3), so a lost link ends the session. `JetsonArm`
waits for each reply up to the move duration + 5 s and otherwise returns a failure. The SSH
client runs in its own process group, so Ctrl+C reaches only MindArm, which can still
command HOME over the link.

Poses: HOME is the calibration rest vector, with the gripper half-open at 1500 µs. HOME
after an error therefore releases a grasped object. The other targets were taught on
2026-10-28 for one sponge at a taped pick spot, which every object label shares:

- MOVE_ABOVE: base 1600, shoulder 1230, elbow 2000, wrist 1400/1010, gripper open at 1900.
- GRASP: gripper 1150.
- LIFT: about 5 cm up.
- handover: base 1000, shoulder 1430, elbow 1800, gripper 1150; RELEASE drops the sponge there.
- RELEASE: gripper 1900.

The teaching operator was Claude, watching the camera, with the user confirming depth
and contact by eye. The teaching surfaced these observations:

- The straight joint-space path from HOME reaches the pick spot partly from the side. It
  surrounds the sponge only when the sponge sits exactly on the taped spot.
- A grip of 1450 µs dropped the sponge during a base swing; 1150 µs held it.
- The elbow (2000) and base (1000) targets sit at the configured limits. Those limits are
  the bring-up clamp, not measured stops.
- The wrist_pitch MG90S stalled silently once, from 1550 to 1750 µs, until it was freed by
  hand.
- The pick→handover move runs the base at its 400 µs/s limit.

### Pose teaching tool

`python -m mindarm.teach` energizes the arm exactly like MindArm's startup and serves a page
on `127.0.0.1:8766` only. The page shows the Jetson camera as MJPEG, from a second SSH session
running `v4l2-ctl` at 640×480 (`--no-camera` skips it). The command requests 15 fps; on
2026-10-28 the camera delivered about 23 fps at ~54 KB per frame (~1.3 MB/s over the USB link). It has one slider and
±10/±50 µs buttons per joint, "save as" buttons for the five taught targets, "visit" buttons,
and STOP (`cancel()`).

Operator moves use `JetsonArm.move_to()`. It applies the same limit and speed check, with the
duration set by the slowest joint's `max_speed` (at least 0.05 s), and the service envelope
still applies. It is not part of the `ArmController` Protocol, so planner output can never
reach it.

Saving rewrites only that target's line in `config/jetson_arm.yaml` and keeps the comments:

- MOVE_ABOVE stores all six joints; teach it at the grasp position with the gripper open.
- GRASP and RELEASE store only the gripper.
- LIFT stores the five arm joints.
- handover stores the five arm joints, plus the GRASP gripper value, so the object stays held.

HOME is not taught here. After each save, the tool preflights two consecutive demo pickup
cycles on the simulator at `primitive_duration_s` and reports any limit or speed violation.
A visit follows the same straight joint-space path that MindArm will take, only at the
speed-limited duration. Ctrl+C commands HOME and exits.

## Orchestrator and cancellation

| Current state | Event or completion | Next state / action |
| --- | --- | --- |
| IDLE | TOGGLE_ENGAGE | PROPOSING |
| PROPOSING | Scene and four options ready | SELECTING |
| SELECTING | SELECT(index) | PLANNING |
| SELECTING | SELECT_TARGET, exactly one non-Home option mentions `target_label` | PLANNING with that index |
| SELECTING | SELECT_TARGET, no single match | Stay in SELECTING; print and log `gesture_ignored` |
| PLANNING | Valid motion plan ready | EXECUTING |
| EXECUTING | Last primitive succeeds | Log cycle, then PROPOSING |
| Any engaged state | TOGGLE_ENGAGE | Discard pending result, HOME, IDLE |
| Any state | Error | HOME, IDLE, log error/recovery outcome |
| Any state | QUIT / EOF / Ctrl+C | HOME, IDLE, save recording, exit |

IDLE's engagement behavior takes precedence over the generic toggle-to-IDLE rule.
Choosing Home executes HOME and reproposes options while staying engaged. Pressing **e**
homes and disengages. An error is recoverable with a later **e**. Failed HOME is reported
explicitly; the program never claims recovery succeeded when it did not.

At most one daemon worker handles perception, planning, and arm execution. Toggling
invalidates its future; its eventual response cannot change state or execute a plan.
Reengagement waits for that worker to finish before issuing another request. This bounds
work during repeated toggles. An in-flight HTTP request itself is not cancelled; it can
consume tokens and emit a late call record while the process remains alive. Quit does not
wait for a planner worker.

Interruptible execution: in EXECUTING, each primitive runs on the worker while the main
loop keeps polling input. Disengagement, errors, and quit during EXECUTING call
`arm.cancel()`, wait for the in-flight `execute()` to return, and only then submit HOME, so
arm execution stays serialized. The HOME recovery runs on the main thread and is not
interruptible; keys pressed meanwhile are processed afterwards.

Default poll interval is 20 ms. EEG uses its independent chunk cadence. No new motion
primitive is submitted before the next input poll.

`--once` exits after one successful cycle or error, attempts HOME, and saves recordings.
CLI exit status is 0 after a normal run, or 1 for setup failures or an unrecovered recorded
error. A subsequent successful cycle clears the prior recoverable error status. The CLI
replaces characters that its console encoding cannot represent (for example U+2011 under
cp1252 when output is redirected on Windows), so model text cannot crash a cycle.

## JSONL schema and observability

Files: `logs/<UTC-session-id>-<unique-suffix>.jsonl`. Every line is one JSON object with
`type`, UTC `timestamp`, and `session_id`. Writes are locked across threads and flushed
by closing the file after each append. Logs never contain configured API keys.

Record types: `session_start`, `configuration`, `planner_call`, `planner_fallback`,
`cycle`, `gesture`, `gesture_ignored`, `disengage`, `session_end`, `startup_or_shutdown_error`.

Each started cycle emits exactly one `cycle` record on completion/cancellation/error:

- `started_at`, `finished_at`, `status` (completed / cancelled / error).
- `scene`, `options`, zero-based `selection`, `plan`, and `result` (null until available).
- `latencies`: available `perception_ms`, `propose_ms`, `plan_ms`, `execute_ms`, measured with a monotonic timer.
- Error/cancellation records add reason, error/type, HOME recovery result and recovery error.
- Joint values in `result` use the configured arm's unit; the `configuration` record names the arm.

`planner_call` adds operation, model, attempt (content attempt: 1, or 2 for the repair),
call (1..`max_calls` within the operation), `max_calls` (3), `reasoning`, `strict_schema`,
latency_ms, usage, status (`ok`, `invalid_output`, `transient_error`, `api_error`), and
validation/error information. `planner_fallback` identifies two rejected responses.

`python -m mindarm.bench` writes its own file in the logs directory with the same
`planner_call` and `planner_fallback` records, plus `bench_configuration` (model, N, token
budget, per-operation options) and `bench_summary` (the reported numbers).
The session end links the NPZ recording and completed-cycle count. Abrupt process
termination can prevent final cycle/recording persistence; crash-safe recording is a later concern.

## Configuration and validation

Implementation keys: `eeg: fake | replay | stream`, `decoder: keyboard | blink`, `perception: fake`,
`planner: nemotron | fake`, `arm: simulated | jetson`. Unknown values/fields fail validation.
The `--planner` override supports offline runs without modifying defaults.

`planner_propose` and `planner_plan` each hold `reasoning` (required) and `strict_schema`
(default true). They choose how the model is called, never a motion parameter.

`stream_url`, `target_label`, `blink_threshold_mv`, `blink_min_gap_s`, `blink_max_gap_s`, and
`blink_refractory_s` configure the gesture path. None of them can set a motion parameter.
The target resolves by plain text match against the planner's option label and description,
so the closed motion vocabulary and the rule that the LLM sets no angles stay unchanged.

`jetson_arm_path` (default `jetson_arm.yaml`) holds the real arm's µs limits and poses.
`jetson_command` is the SSH argv that starts the service. It defaults to Windows' OpenSSH
client at its full path, because the Git `ssh` found first on this laptop's PATH does not
work over the Jetson's USB network link. The argv sets `BatchMode` and keep-alives,
connects to the alias `jetson-usb`, and runs `python3 -u ~/mindarm_arm.py`.

Committed configuration may name that SSH host alias and remote command. Credentials,
keys, secret values, and serial ports do not belong in configuration committed to this
repository. Servo pulses never come from the LLM.

## Tests

`pytest` excludes the `integration` marker by default and blocks socket connections in
offline test cases. It covers schemas, fake signals, alpha bursts, replay/recording,
keyboard mapping/EOF, fixtures, factory/path selection, model retries/fallbacks, speed and
joint rejection, simulated state, state transitions, stale planner cancellation, and the
actual module CLI in a subprocess. API tests mock SDK responses, including token usage:
the strict schema and reasoning flag sent per operation, the per-scene plan schema, one
transient retry that is not the repair, no retry for client errors, and the three-call bound.
The benchmark's bookkeeping (first-try/repair/fallback counts, rejection reasons, cached
tokens, cost, prices from the model list) is tested with a simulated client.
The double-blink detector, gesture decoder, EEG tap, stream source, and target resolution are
tested with synthetic signals and a fake HTTP response.
It also covers cancellation of an in-flight arm primitive before HOME, and `JetsonArm`
with an in-memory fake transport: startup, preflight rejection without messages, service
rejection, cancellation, and link loss. The Jetson service logic is tested with a fake I2C
bus: chip adoption, envelope and speed rejection, staggered start, and stopping a ramp on
input. The pose teaching tool is tested for MJPEG frame splitting, the per-target save rules,
comment-preserving YAML rewriting, the demo-plan preflight, and limit-checked operator
moves and visits. The default suite never opens SSH.

`pytest -m integration` explicitly enables the live proposal + motion test. It skips if
`NEBIUS_API_KEY` is absent, validates both responses, executes the resulting plan in the
simulator, and asserts neither operation used fallback. A valid key alone does not prove
model access or service availability; a real successful test run is required.

## Hardware measurements

Jetson Nano 4GB Developer Kit, L4T R32.7.5, Python 3.6.9, MAXN mode. The laptop reaches it
over the micro-USB network link (`jetson-usb`, 192.168.55.1). Servo supply: 5 V 10 A,
measured at 5.0–5.2 V at the PCA9685 terminal (2026-10-11).

Calibration (2026-10-11, `servo_test.py`; * = the script's 1000–2000 µs clamp, not a
physical stop). The servos were not centered before the horns were mounted.

| Ch | Joint | Servo | Min µs | Max µs | HOME µs | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | base | MG996R | 1000* | 2000* | 1500 | ~90° sweep, front at HOME |
| 1 | shoulder | MG996R | 1000* | 2000* | 1450 | 1500 buzzes (backlash at vertical balance) |
| 2 | elbow | MG996R | 1000* | 2000* | 1500 | Window may be offset |
| 3 | wrist_pitch | MG90S | 1180 | 1980 | 1600 | Real stops |
| 4 | wrist_roll | MG90S | 1000* | 2000* | 1500 | |
| 5 | gripper | SG90 | 1130 | 1950 | 1500 | Higher µs opens; 1950 open, 1130 closed |

HOME as a whole passed on 2026-10-11: the arm holds an upright "candle" posture with a
slight hum. On 2026-10-28, read-only checks over SSH showed the following:

- Bus 1 shows `0x40` and `0x70` (PCA9685 and its all-call address). The PCA9685 was at its
  power-on defaults: MODE1 `0x11`, MODE2 `0x04`, prescaler `0x1E`.
- Bus 0 shows `0x48` and `0x49` (two ADS1115, config `0x8583`).
- A USB UVC camera (`0c45:636b`, MJPG up to 1920×1080 at 30 fps) is on `/dev/video0`.
  No step uses it yet.

First real-arm cycles (2026-10-28, `--planner fake --once`, untaught poses, so only the
gripper moves):

- Startup energized all six released channels at HOME without incident.
- Later launches adopted the energized channels; none jumped.
- Four cycles completed: Home once, and three object pickups.
- `execute_ms` was 42 for Home (already at HOME). For the five-primitive pickups it was
  4766–4926 ms, against 4.5 s of ramping (three 1.5 s gripper moves). That is about 0.1 s
  of overhead per real move. Moves that change nothing return at once.

EEG gesture measurements (2026-10-30, laptop read as on battery before the first run and after
the last; the operator did not confirm the Jetson and servo power sources; one AD8232 channel
on the forehead, ADS1115 at 250 Hz, read through the viewer
stream; files `eeg_gesture_blink_20261030_*.npz` and `eeg_calibration_20261030_*.json` in
`../hardware_bringup`, outside the repo):

- Alpha, eyes closed (three valid calibration runs, forehead): 8–13 Hz rms stayed at 4–5 µV
  input-referred with eyes open or closed. The console's "alpha ratio" came from 1–4 Hz eye
  movement dropping when the eyes close, not from an alpha increase. No eyes-closed detector
  is justified at the forehead.
- Voluntary double blink (one cued run, 10 cues, 120 s rest, fixed rule: notch 60/120 Hz,
  1 s moving-mean removal, 40 ms smoothing, two positive peaks 0.1–0.8 s apart, 1.5 s
  refractory): 8/10 cues detected at 250 mV with 4 false events in the 120 s rest and 9 between
  cues; 7/10 at 300 mV with 2 and 7; 4/10 at 350 mV with 1 and 5. These thresholds were compared
  on this same run, so no held-out run exists. Cued double blinks reached 450–750 mV
  at the ADC, but peaks in the first 30 s of rest reached 300–615 mV. No video shows
  whether those were spontaneous blinks or a continued gesture. The
  acceptance target (8/10 and at most 1 false event in 2 min) was not met, so `decoder: keyboard`
  stays the default. Two earlier runs were discarded: one without visible cues, one with a
  flat 1.651 V signal (2 mV p-p) during every cue.
- End-to-end gesture chain without arm motion (07:17, session 20261030T121648-c7701ea22f36,
  `eeg: replay`, `decoder: blink`, simulated arm, one-sponge scene): the recorded double blink
  selected the sponge option 17.05 s into the signal. One proposal and one plan, no repair,
  no fallback, five primitives executed.
- First Nemotron cycle on the real arm (07:34, session 20261030T123400-c9a08b495b5a, `eeg: stream`,
  `decoder: blink`, one-sponge scene, laptop on its charger by operator decision): the camera shows
  the arm moving through the plan, `execute_ms` was 7855, and the arm recovered HOME. The log has no
  `gesture` event, so the selection came from the keyboard: the operator's blink pair peaked at
  286 mV, below the 300 mV threshold. The sponge stayed on the table.
- Live double-blink selection on the real arm (15 sessions from the `/console` page on
  2026-10-30, `blink_threshold_mv` 250): each completed its five primitives. These runs had no
  cues, so they do not measure accuracy. Nine of the cycles have screen recordings. In three
  of them (08:07, 08:09, 08:14 Lima) the sponge rose and was carried to the handover pose. In the
  others it stayed on the table, some with the operator's hand near it.
- Gripper fault (same morning): commanding 1500, 1900, 1150, and 1500 µs at HOME moved nothing on
  camera (0 % changed pixels around the jaws) while the service reported success. Other joints
  moved in the same session. The PCA9685 held the pulses written to channel 5, so the fault is
  downstream of the chip. A later full-open command worked about 5 s late by the operator's
  report, and a close command then showed no change on camera for 8 s. At the handover pose,
  with the jaws in full view, 12 commanded moves (6 between 1150 and 1900 µs, 6 between 1400 and
  1700 µs) produced no visible jaw motion. The operator then reported it working again, and the
  next run confirmed it on camera. The cause is not recorded.
- Live chain without arm motion (session 20261030T123012-d35b243fe626, `eeg: stream`,
  `decoder: blink`, simulated arm, one-sponge scene): a double-blink detection on the live
  signal selected the sponge option, with one proposal, one plan, and five primitives executed.
  The run had no cue, so it does not show whether that blink pair was voluntary.
- Rule comparison on that same cued run, so not held out. Pair at 300 mV: 7/10 hits, 7 false
  between cues, 2 false in the first 30 s of rest, 0 in the last 90 s. Single peak at 500 mV:
  8/10, 2, 2, 0. On 205 s of other unsaturated baseline and rest, the pair gave 2 events and the
  single peak 4. No rule meets the target, so the implemented pair rule stays.
- 60 Hz at the ADC, FFT over 4 s: 11–15 mV on battery (rest of the valid run), and 25–237 mV with
  the laptop on its charger across reads minutes apart. The spread is too wide to attribute a
  change to the charger, and none of this measures leakage current.
- A blink command is "a command from a voluntary blink captured with electrodes, not brain
  activity".

## Nemotron measurements

Model `nvidia/nemotron-3-super-120b-a12b` on Nebius Token Factory (us-central1), called from
the laptop on 2026-10-28.

`pytest -m integration` passed in 7.0 s: one proposal and one motion plan, both valid, with
no fallback.

Live demo, three proposals and two plans, with the simulated arm:

| Operation | Attempt | Latency ms | Prompt / completion tokens (reasoning) | Result |
| --- | --- | --- | --- | --- |
| propose_options | 1 | 3729 | 451 / 373 (266); 432 cached | ok |
| plan_motion (Pick up cup) | 1 | 2981 | 614 / 423 (370) | invalid: `object_label` on GRASP and LIFT |
| plan_motion (Pick up cup) | 2 | 1955 | 779 / 368 (329) | ok after the one repair |
| propose_options | 1 | 1967 | 451 / 405 (301) | ok |
| plan_motion (Home) | 1 | 1283 | 607 / 181 (167) | ok, `[HOME]` |
| propose_options | 1 | 2042 | 451 / 399 (285); 432 cached | ok |

Cycle `plan_ms` was 4940 ms with the repair and 1286 ms without one; `propose_ms` was
2.0–3.7 s. The model listed Home first. Its pickup plan stopped at LIFT, with no handover or
release; that is valid within the vocabulary. The third proposal's description contained
U+2011. Before the console fix above, that character ended the cycle with an encoding error
when output was redirected; the error path homed and disengaged as specified.

First planner benchmark (2026-10-29, a one-off script before `mindarm.bench` existed): JSON
mode, reasoning on, 10 proposals and 10 plans for fixed goals. Proposals 2.35 s p50 / 3.33 s
p95, 10/10 valid on the first try; plans 2.53 s p50 / 4.73 s p95, 7/10 on the first try and
3 repaired. It did not record why plans were rejected.

#### `python -m mindarm.bench` (2026-10-29)

Method: the fixture scene; N proposals, then N plans for the options those proposals
returned (run i plans option i mod 4 of proposal i), as the live loop does. Latency is the
wall time of one planner operation, including any repair and transient retry; p50/p95 use
linear interpolation. "Expected" means a proposal offers every scene object, and a plan for
an option naming one scene object is exactly that object's five-step handover, while any
other option plans exactly `[HOME]`. Cost uses the per-token prices in Token Factory's
`GET /v1/models?verbose=true`, with cached prompt tokens at the listed cache-read price
(USD per million tokens, input / cached input / output: Super 0.30 / 0.06 / 0.90, Ultra
1.00 / 0.12 / 3.00, Nano 0.06 / 0.012 / 0.24). Temperature 0.1, 4096-token budget.

Baseline, rerun with the tool before any change (05:37 UTC; old prompt; JSON mode;
reasoning on; plans for the four fixed goals Pick up cup / bottle / sports ball / Home):

| Operation | N | p50 | p95 | Range | First try | Repaired | Fallback | Prompt (cached) / completion (reasoning) | Cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| propose_options | 10 | 2.36 s | 3.69 s | 1.89–3.96 s | 10/10 | 0 | 0 | 4510 (3456) / 3740 (2609) | $0.0039 |
| plan_motion | 10 | 4.88 s | 6.05 s | 1.36–6.20 s | 6/10 | 4 | 0 | 9382 (2608) / 5790 (4830) | $0.0074 |

All four rejected plans failed the same rule: `object_label` on an action other than
MOVE_ABOVE (`Only MOVE_ABOVE requires object_label`).

Current defaults (06:00 UTC): Super, strict schema, reasoning off for both operations, final
prompts.

| Operation | N | p50 | p95 | Range | First try | Expected | Prompt (cached) / completion | Cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| propose_options | 30 | 0.76 s | 1.17 s | 0.55–1.20 s | 30/30 | 30/30 | 11820 (9936) / 2803 | $0.0037 |
| plan_motion | 30 | 0.54 s | 0.99 s | 0.27–1.27 s | 30/30 | 30/30 | 17212 (9376) / 1496 | $0.0043 |

No repairs, fallbacks, or API errors. One proposal plus one plan costs about $0.00027
($0.00042 at list price without the cache-read discount); 67% of prompt tokens were cached.

Variants with the final prompts (Super unless named; only the changed setting is listed):

| Variant | N | Proposals p50 / p95 | Plans p50 / p95 | Plans first try | Expected (proposals, plans) | Cached prompt tokens | Cost per cycle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Plans with reasoning | 10 | 0.58 / 0.81 s | 1.77 / 2.88 s | 10/10 | 10/10, 10/10 | 78% | $0.00040 |
| Plans in JSON mode (`json_object`) | 20 | 0.85 / 1.29 s | 0.53 / 0.81 s | 20/20 | 20/20, 20/20 | 84% | $0.00023 |
| Ultra | 10 | 0.50 / 0.52 s | 0.49 / 0.53 s | 10/10 | 10/10, 10/10 | 0 reported | $0.0015 |
| Nano | 10 | 1.38 / 3.79 s | 0.92 / 1.85 s | 7/10, 3 repaired | 10/10, 10/10 | 0 reported | $0.00012 |

Findings:

- The first-try gain came mostly from the prompt. With the per-scene schema in the prompt
  and the exact handover sequence, JSON mode alone also passed 20/20. Strict decoding makes
  the structure a property of the service's output rather than of the model's compliance;
  this sample does not show a measurable difference between the two.
- Reasoning added 1.2 s to the plan p50 (1883 reasoning tokens for 10 plans) with no gain in
  validity or completeness, so both operations run without it.
- Nano's three rejected plans broke the rule the schema cannot express: a Home or resting
  option must plan exactly `[HOME]`. The local gate rejected each one and the repair prompt
  fixed it.
- Model choice: Super for both operations. Ultra had the same validity with a tighter p95,
  at 5.6× the cost per cycle and no cached tokens reported. Nano was the cheapest, but it
  needed repairs and its proposal p95 was 3.8 s.
- With an intermediate prompt (same day), Nano with reasoning on reached a 24.8 s plan p95.
  One call timed out at 30 s and the transient retry recovered it. That was the first live
  transient retry.
- Its proposals then used identifier labels (`grasp_cup`, `GraspBall`) and named the sports
  ball "ball". The final prompt asks for the scene's object names.
- In a CLI run before the prompt fix, the model's own option "Pick cup: Grasp the cup at
  (0.25, 0.55)" planned MOVE_ABOVE, GRASP, LIFT only. That plan was valid, but the follow-up
  HOME would drop the cup. The earlier benchmark had missed this because it planned fixed
  goals whose descriptions said "hand it over".
- With the final prompts, a CLI run planned the model's option "cup: Pick up the cup and hand
  it to the user" as the full five-step handover. Its first proposal took 11.3 s with 0 cached
  tokens, against a 1.20 s maximum in the benchmark. A cold first call can be far slower than
  the p95.
- Measured from one laptop in Lima during one night; latency varies with time and load.

`pytest -m integration` with the new defaults passed in 4.6 s (7.0 s on 2026-10-28).

Nemotron 3.5 Super VL probe (same day, `api.tokenfactory.nebius.com`, which lists the model;
the `us-central1` endpoint does not). One synthetic 320×240 image, JSON mode, reasoning off:

- It answered in 2.2 s.
- The output validated as `Scene`, with object centers within 0.02 of the truth.
- It is not wired into perception.

## Open risks

1. **Deployment Python:** Jetson Nano on JetPack 4.6 ships Python 3.6; this project requires
   3.10+. Chosen for now: the orchestrator and planner stay on the laptop, and the Jetson
   runs only the Python 3.6 arm service (standard library + smbus).
2. **EEG availability:** EEG hardware is expected around week 3. Use fake acquisition and
   recorded files for independent software development.
3. **Nemotron latency:** With the current defaults, 0.76 s p50 / 1.17 s p95 per proposal and
   0.54 s p50 / 0.99 s p95 per plan (N=30, see Nemotron measurements). Those are one night's
   numbers from one laptop, and a cold first proposal once took 11.3 s. A repair or a transient retry adds a call, and a timeout costs
   30 s before the retry. With reasoning on, the reasoning can exhaust the token budget;
   truncated output is rejected.
4. **Live service dependency:** Verified on 2026-10-28 with a real key. The demo still
   depends on Token Factory availability, and a plan can need its one repair attempt.
5. **Physical safety:** Speed and envelope checks are not collision avoidance,
   reachability, force limits, calibrated kinematics, or an emergency stop.
   - The only stops are disengage/quit and cutting the servo supply.
   - There is no position feedback: the first pulse to a released servo jumps at full speed,
     and stalls or slips go undetected.
   - After a lost link the arm stays powered in its last posture.
   - The PCA9685 runs from the Jetson's 3.3 V. If the servo supply is cut and restored while
     the chip runs, every servo jumps to its last pulse.
   - A stalled servo is invisible to software: the wrist_pitch MG90S stopped silently once
     while its pulse kept changing.
   - The shoulder hums near vertical balance at HOME (gear backlash).
   - Limits and poses are judged by eye. Four of the limits are the bring-up script's clamp,
     not physical stops, and the channel directions are not yet recorded.
   - This is a supervised tabletop prototype, not an assistive device.
6. **Signal ambiguity:** Planned 10 Hz selection overlaps the alpha range. Future decoding
   must distinguish intent selection from the eyes-closed engagement gesture using recordings.
   On 2026-10-30 no eyes-closed alpha increase was measurable at the forehead, and a cued
   double-blink rule missed the acceptance target (see Hardware measurements). Keyboard
   selection stays the default. `decoder: blink` is optional, and a false event while
   SELECTING would select the target.
7. **Object identity:** Label-only references cannot distinguish two objects with the same
   label. Stable object IDs will require a documented interface revision in a later step.
8. **Persistence:** EEG buffers consume memory for the session and are written on orderly
   exit. Long or crashed sessions need incremental storage in a later step. Idle gaps are
   omitted from the NPZ signal; session logs retain wall-clock events.
9. **Input timing:** Measurements cover keyboard intent and one double-blink run; they do not
   establish SSVEP reliability, flicker timing, usability, or clinical benefit.
10. **Live gesture safety:** Electrodes on a person are only allowed while everything runs on
    battery. The servo supply runs from the mains, so with the current hardware the gesture
    drives the real arm only as replayed signal, which must be labeled as recorded, not live.
11. **Stream reads:** `StreamEEGSource.read` waits on the network. A stalled stream delays key
    polling by up to the block duration plus 2 s before the error path commands HOME. The
    orchestrator's pacing consumed about 220 of the 250 samples/s in the measured sessions, so
    the stream source drops about 10 % of live samples and the saved NPZ is that much shorter.
