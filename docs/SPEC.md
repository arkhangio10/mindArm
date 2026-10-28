# MindArm specification — Step 1

Status: implemented walking skeleton. This document is the source of truth for interfaces,
observable behavior, safety boundaries, and open risks. Update it whenever a contract changes.

## Scope and acceptance

The laptop demo runs without hardware. `python -m mindarm` uses real Nemotron by default
and fake EEG/perception, keyboard intent selection, and a simulated arm. After **e** and
four displayed options, **2** selects index 1 and a validated motion plan executes.
`--planner fake` provides the same loop offline. Python 3.10+ is required.

Out of scope: serial/Arduino integration, PWM controllers, cameras, object detectors,
SSVEP/alpha decoding, flickering UI, dashboards, web search, robot foundation models,
motor imagery, and real motor execution.

## Architecture and files

`src/mindarm/interfaces.py` defines Pydantic v2 models and runtime-checkable Protocols.
`config.py` loads validated YAML. `factory.py` is the only implementation-selection point.
The orchestrator depends on Protocols, so future adapters replace fakes without changing
the orchestration contract. `session_log.py` owns JSONL writes; `eeg/recording.py` owns
recording validation and persistence. `__main__.py` handles setup and terminal cleanup.

The default editable install locates `config/settings.yaml` relative to the source checkout.
A non-editable installation must use `--config` pointing to a retained checkout/configuration.
Relative fixture/log/recording paths resolve from the selected YAML file's directory.
`.env` is read from that directory's parent, without overriding exported environment values.

## Interfaces

All interfaces are synchronous. `Decoder.poll()` must return immediately. In Step 1,
`ArmController.execute()` and EEG reads are bounded in-memory operations. Potentially
slow perception/planner calls execute on one background worker.

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
```

### Shared data models

Models forbid extra fields, implicit coercion, NaN, and infinity. All indexes are zero-based;
displayed keys are one-based. Joint angles use degrees.

| Model | Fields and invariants |
| --- | --- |
| `DecoderEvent` | `kind`: SELECT / TOGGLE_ENGAGE / QUIT; `index`: 0..3 required only for SELECT |
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

`EEGRecording` checks every block, retains copies in memory, and writes a compressed NPZ
on orderly shutdown. It creates no archive when recording is disabled or no samples were
captured. Acquisition resumes after reengagement; the stored signal omits idle wall time.
Keyboard intent is deliberately independent of synthetic/replayed EEG in this step.

### Keyboard and perception

Keyboard keys **1-4** produce SELECT(0..3); **e** toggles engagement; **q** produces QUIT.
EOF also quits; Ctrl+C triggers cleanup. Windows console input uses `msvcrt`; POSIX TTYs
use cbreak mode with restoration. Redirected input uses a background reader. Control
input is polled before state work on every tick. Selection outside SELECTING is ignored.

`FakePerception` reloads and validates `fixtures/scene_table.json` per detection. The fixture
contains cup, bottle, and sports ball with normalized image coordinates and confidence.

### Planners

`FakePlanner` provides up to three unique object-pickup options plus Home, padding short
scenes with generic options. An object pickup yields MOVE_ABOVE, GRASP, LIFT,
MOVE_TO_POSE(handover), RELEASE; generic options yield HOME.

`NemotronPlanner` uses `openai.OpenAI` chat completions against the configured Nebius
endpoint. Defaults: `https://api.tokenfactory.us-central1.nebius.com/v1/` and
`nvidia/nemotron-3-super-120b-a12b`. The API key comes from `NEBIUS_API_KEY`.
The request uses `response_format={"type":"json_object"}`, temperature 0.1, a configurable
token budget (default 4096), and request timeout (default 30 seconds). Prompts include
the schema, action constraints, and scene as text. No images are sent.

1. Validate JSON, Pydantic fields, exact option count/Home, and motion arguments.
2. Reject non-stop finish reasons (including token-truncated responses).
3. On invalid output, retry once with the validation error included in the conversation.
4. After the second invalid output, return Wait / Rest / Reset / Home options or `[HOME]`.
5. Home and those generic resting labels must map to `[HOME]` even after a later valid response.

SDK automatic retries are disabled; at most two calls occur for an invalid-output operation.
Authentication, transport, and service errors are logged and raised as `PlannerUnavailable`;
the orchestrator returns HOME + IDLE. Errors log the API exception type, not raw upstream
bodies or credentials. Missing credentials fail startup. Every completed API attempt logs
latency and usage (null when the API did not supply it). Safe fallback is explicit in logs.

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

The orchestrator submits one primitive per tick, making disengagement possible between
primitives. Consequently a later primitive's joint rejection can occur after earlier
primitives completed; the cycle records partial progress, then attempts HOME. Atomic
preflight applies to each `execute()` submission, not the entire orchestrated cycle.

## Orchestrator and cancellation

| Current state | Event or completion | Next state / action |
| --- | --- | --- |
| IDLE | TOGGLE_ENGAGE | PROPOSING |
| PROPOSING | Scene and four options ready | SELECTING |
| SELECTING | SELECT(index) | PLANNING |
| PLANNING | Valid motion plan ready | EXECUTING |
| EXECUTING | Last primitive succeeds | Log cycle, then PROPOSING |
| Any engaged state | TOGGLE_ENGAGE | Discard pending result, HOME, IDLE |
| Any state | Error | HOME, IDLE, log error/recovery outcome |
| Any state | QUIT / EOF / Ctrl+C | HOME, IDLE, save recording, exit |

IDLE's engagement behavior takes precedence over the generic toggle-to-IDLE rule.
Choosing Home executes HOME and reproposes options while staying engaged. Pressing **e**
homes and disengages. An error is recoverable with a later **e**. Failed HOME is reported
explicitly; the program never claims recovery succeeded when it did not.

At most one daemon worker handles perception/planning. Toggling invalidates its future;
its eventual response cannot change state or execute a plan. Reengagement waits for that
worker to finish before issuing another request. This bounds work during repeated toggles.
An in-flight HTTP request itself is not cancelled; it can consume tokens and emit a late
call record while the process remains alive. Quit does not wait for the worker.

Default poll interval is 20 ms. EEG uses its independent chunk cadence. No new motion
primitive is submitted before the next input poll. A future blocking hardware controller
will need an explicit interruptible execution contract before it can replace this simulator.

`--once` exits after one successful cycle or error, attempts HOME, and saves recordings.
CLI exit status is 0 after a normal run, or 1 for setup failures or an unrecovered recorded
error. A subsequent successful cycle clears the prior recoverable error status.

## JSONL schema and observability

Files: `logs/<UTC-session-id>-<unique-suffix>.jsonl`. Every line is one JSON object with
`type`, UTC `timestamp`, and `session_id`. Writes are locked across threads and flushed
by closing the file after each append. Logs never contain configured API keys.

Record types: `session_start`, `configuration`, `planner_call`, `planner_fallback`,
`cycle`, `disengage`, `session_end`, `startup_or_shutdown_error`.

Each started cycle emits exactly one `cycle` record on completion/cancellation/error:

- `started_at`, `finished_at`, `status` (completed / cancelled / error).
- `scene`, `options`, zero-based `selection`, `plan`, and `result` (null until available).
- `latencies`: available `perception_ms`, `propose_ms`, `plan_ms`, `execute_ms`, measured with a monotonic timer.
- Error/cancellation records add reason, error/type, HOME recovery result and recovery error.

`planner_call` adds operation, model, attempt (1 or 2), latency_ms, usage, status, and
validation/error information. `planner_fallback` identifies two rejected responses.
The session end links the NPZ recording and completed-cycle count. Abrupt process
termination can prevent final cycle/recording persistence; crash-safe recording is a later concern.

## Configuration and validation

Implementation keys: `eeg: fake | replay`, `decoder: keyboard`, `perception: fake`,
`planner: nemotron | fake`, `arm: simulated`. Unknown values/fields fail validation.
The `--planner` override supports offline runs without modifying defaults.

No serial ports, device access, camera APIs, servo angles from the LLM, or secret values
belong in configuration committed to this repository.

## Tests

`pytest` excludes the `integration` marker by default and blocks socket connections in
offline test cases. It covers schemas, fake signals, alpha bursts, replay/recording,
keyboard mapping/EOF, fixtures, factory/path selection, model retries/fallbacks, speed and
joint rejection, simulated state, state transitions, stale planner cancellation, and the
actual module CLI in a subprocess. API tests mock SDK responses, including token usage.

`pytest -m integration` explicitly enables the live proposal + motion test. It skips if
`NEBIUS_API_KEY` is absent, validates both responses, executes the resulting plan in the
simulator, and asserts neither operation used fallback. A valid key alone does not prove
model access or service availability; a real successful test run is required.

## Open risks

1. **Deployment Python:** Jetson Nano on JetPack 4.6 ships Python 3.6; this project requires
   3.10+. Options: Docker, a newer Python build, or keep the planner on the laptop. No
   Jetson-specific implementation is included.
2. **EEG availability:** EEG hardware is expected around week 3. Use fake acquisition and
   recorded files for independent software development.
3. **Nemotron latency:** Proposal plus motion planning may hurt UX. Measure each API call
   from day 1. The model's reasoning can exhaust the token budget; truncated output is
   rejected and the configurable budget/timeout may need tuning from live measurements.
4. **Unverified live service:** Until tested with a configured key, real model access,
   JSON behavior, latency, and token usage remain unverified in this checkout.
5. **Physical safety:** Virtual speed checks are not collision avoidance, reachability,
   force limits, calibrated kinematics, or an emergency stop. Hardware needs a separate
   execution/cancellation design and validation before deployment.
6. **Signal ambiguity:** Planned 10 Hz selection overlaps the alpha range. Future decoding
   must distinguish intent selection from the eyes-closed engagement gesture using recordings.
7. **Object identity:** Label-only references cannot distinguish two objects with the same
   label. Stable object IDs will require a documented interface revision in a later step.
8. **Persistence:** EEG buffers consume memory for the session and are written on orderly
   exit. Long or crashed sessions need incremental storage in a later step. Idle gaps are
   omitted from the NPZ signal; session logs retain wall-clock events.
9. **Input timing:** This skeleton tests keyboard intent only; it does not establish SSVEP
   reliability, flicker timing, usability, or clinical benefit.
