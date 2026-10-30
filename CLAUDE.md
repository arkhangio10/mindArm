# Working on MindArm

Read `docs/SPEC.md` before editing. It is the source of truth; update it in the same change
as any model, interface, state-machine, configuration, or safety behavior change.

- Use the walking-skeleton method: keep the end-to-end laptop loop runnable and replace
  one fake module at a time. Do not implement later steps without an explicit request.
- Define module boundaries as `typing.Protocol` in `src/mindarm/interfaces.py`; validate
  exchanged data with Pydantic v2. Select implementations through configuration and factory.
- Keep Python 3.10 compatibility, type hints, public-class docstrings, and focused files.
  Runtime dependencies are openai, pydantic, numpy, pyyaml, python-dotenv; pytest is for tests.
- Never let LLM output set angles or speeds. Preserve the closed primitive and pose sets.
  The arm owns targets, limits, and speed enforcement; rejected moves must report failure.
- Preserve input polling during planner work, discard stale results after disengagement,
  serialize arm execution, and attempt HOME on errors/quit.
- Keep API calls offline in default tests. Never weaken the integration test to accept a
  fallback as proof the service worked. Never commit `.env`, logs, recordings, or API keys.
- Record measurements and open risks in SPEC. Keep session JSONL and NPZ replay compatible.
- Describe the project as "brain-controlled through intent selection". Identify this step
  as a keyboard-driven simulation; do not imply real brain control or clinical validation.
- Preserve the prior-work citation. The arm was designed for this hackathon and, like the
  software, built during the submission period; never describe it as pre-existing.
  Do not add third-party branding, logos, or character names.

Development (from this directory):

```console
python -m pip install -e ".[dev]"
python -m mindarm --planner fake
pytest
pytest -m integration
```

The integration command is explicit and requires `NEBIUS_API_KEY`; it skips without one.
Real demo: configure `.env`, run `python -m mindarm`, press **e**, wait for four options,
then press **2**. Press **e** to disengage or **q** to quit. Use `--once` for one cycle.

Local PowerShell commands can use `.\.venv\Scripts\python.exe` and
`.\.venv\Scripts\pytest.exe` without activating the virtual environment.
