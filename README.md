# MindArm

## Overview

MindArm explores a low-cost assistive arm that is **brain-controlled through intent selection**.
It is a solo entry for the Nebius x NVIDIA Global AI Hackathon, Physical AI Track,
with a project deadline of October 30, 2026.

**Step 1 is a hardware-free walking skeleton.** Keyboard input stands in for brain-derived
intent. Perception, EEG, and the arm are simulated; the default planner calls real NVIDIA
Nemotron through Nebius Token Factory. This step does not demonstrate real BCI control
or establish suitability for use by people with motor impairments.

## How it works

1. Press **e** to engage.
2. Fixture perception describes a cup, bottle, and sports ball as text.
3. Nemotron proposes four options, including **Home**.
4. Press **1-4** to select an option; **2** selects the second displayed option.
5. Nemotron produces a validated sequence of allowed motion primitives.
6. The simulated arm enforces local joint/speed limits and logs execution.
7. A fresh set of options is proposed. Press **e** to return home and disengage, or **q** to return home and quit.

`IDLE -> PROPOSING -> SELECTING -> PLANNING -> EXECUTING -> PROPOSING`

The planned BCI uses SSVEP choices at 8.57 / 10 / 12 / 15 Hz and an eyes-closed alpha
gesture for engagement. Flickering visuals, SSVEP/alpha decoding, cameras, and hardware
are outside this step. Synthetic EEG acquisition and NPZ recording/replay already run
alongside the keyboard-driven pipeline.

The contracts and behavior are defined in [docs/SPEC.md](docs/SPEC.md).

## NVIDIA Nemotron + Nebius Token Factory usage

The planner uses the OpenAI-compatible chat completions API with:

| Environment variable | Default |
| --- | --- |
| `NEBIUS_API_KEY` | Required for the real planner |
| `NEBIUS_BASE_URL` | `https://api.tokenfactory.us-central1.nebius.com/v1/` |
| `NEMOTRON_MODEL` | `nvidia/nemotron-3-super-120b-a12b` |

These defaults follow the [official Nebius Nemotron example](https://github.com/nebius/token-factory-cookbook/blob/main/models/nemotron/nemotron3-super-120B.md).
Requests contain text scene data and request JSON output. Every returned object is
validated locally. Invalid output receives one repair attempt containing the validation
error; a second invalid response produces generic resting options or a `[HOME]` plan.
API/transport errors return the application to HOME + IDLE and are logged. A missing key
fails startup with an actionable message.

Each API attempt records latency, token usage when supplied by the API, and outcome in
the session JSONL. Prompts live in `src/mindarm/planner/prompts.py`. The model cannot
specify angles, speeds, or arbitrary commands.

## Setup

Requires Python 3.10 or newer. Run these commands from this project's `mindarm/` directory.
The supported development installation is editable so configuration and fixtures remain
beside the source.

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Edit `.env` locally and set `NEBIUS_API_KEY`. Do not commit it. Existing environment
variables take precedence over `.env`. You can skip the `.env` step for offline runs.

On macOS/Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
cp .env.example .env
source .venv/bin/activate
```

On PowerShell, activate with `.\.venv\Scripts\Activate.ps1` if your execution policy
permits it, or use `.\.venv\Scripts\python.exe` directly in place of `python` below.

## Run

With the environment active and the API key configured:

```console
python -m mindarm
```

Press **e**, wait for four options, then press **2**. In a terminal, keys are immediate;
no Enter is required. Selections outside SELECTING are ignored. During a slow API call,
**e** still disengages and discards its eventual result. **q**, EOF, or Ctrl+C exits safely.

Offline demo, including one-cycle mode:

```console
python -m mindarm --planner fake
python -m mindarm --planner fake --once
```

`--once` exits after one completed or failed cycle and returns the arm home. Without it,
the loop continues until you quit. Offline option 2 picks up the bottle; live options vary.

Configuration:

```console
python -m mindarm --config config/settings.yaml
```

Select implementations in `config/settings.yaml`. All file paths in YAML resolve relative
to that file. For replay, set `eeg: replay` and `replay_path` to an existing recording;
`replay_loop: false` disengages with an EOF error when insufficient samples remain.
Keyboard input remains the intent source during EEG replay.

Each session writes `logs/<session-id>.jsonl`. Engaged samples are saved on orderly exit
to `recordings/<session-id>.npz`. Both directories are gitignored. Records have UTC
timestamps, and each cycle includes the scene, options, selection, plan, result, and
latencies. The NPZ keys are `data` (channels x samples), `sample_rate`, and `n_channels`.

Tests:

```console
pytest
pytest -m integration
```

The default suite is offline and excludes integration tests even when a key is present.
The opt-in integration test calls the real API, skips if the key is missing, and fails
if either operation falls back instead of producing a valid model response.

## Prior work

Maha Nawaz, Abdul Basit, and Muhammad Shafique,
["MindArm: Mechanized Intelligent Non-Invasive Neuro-Driven Prosthetic Arm System"](https://arxiv.org/abs/2403.19992)
(2024), describes a DNN-based EEG system with three actions: idle, handshake, and cup pickup.

This project's planned approach uses SSVEP intent selection instead of a DNN on raw EEG,
and scene-aware Nemotron options instead of three fixed actions. Those BCI and perception
components are future work; this implementation establishes their replaceable contracts.

## What is new

The 3D-printed arm existed before August 26, 2026. The BCI pipeline, planner, perception,
and software are new work for this entry. The current deliverable implements the software
skeleton and real planner integration; it does not include new hardware or a completed BCI.

## Roadmap

- Step 1: This walking skeleton, validated motion vocabulary, logs, replay, and offline tests.
- Later steps: Replace one fake module at a time with recorded/real EEG, SSVEP decoding,
  perception, and calibrated arm hardware, preserving the interfaces and updating SPEC.
- Track latency and hardware availability before committing to deployment architecture.

See the open risks in [docs/SPEC.md](docs/SPEC.md). No later-step implementation is included.

## License

[MIT](LICENSE).
