"""Run with python -m mindarm after an editable installation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from mindarm.config import default_config_path, load_settings
from mindarm.factory import Components, create_components
from mindarm.orchestrator import Orchestrator
from mindarm.session_log import SessionLog


def main(argv: list[str] | None = None) -> int:
    """Configure a session, run the keyboard loop, and clean up terminal state."""
    parser = argparse.ArgumentParser(description="MindArm hardware-free intent selection demo")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--planner", choices=("fake", "nemotron"), help="Override the configured planner")
    parser.add_argument("--once", action="store_true", help="Exit after one completed or failed cycle")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    components: Components | None = None
    log: SessionLog | None = None
    try:
        settings = load_settings(args.config)
        load_dotenv(args.config.resolve().parent.parent / ".env", override=False)
        if args.planner is not None:
            settings.planner = args.planner
        log = SessionLog(settings.logs_dir)
        log.write({"type": "configuration", "settings": settings.model_dump(mode="json")})
        components = create_components(settings, log)
        app = Orchestrator(components.eeg, components.decoder, components.perception,
                           components.planner, components.arm, settings, log,
                           emit=lambda message: print(message, flush=True))
        print(f"Session log: {log.path}", flush=True)
        app.run(once=args.once)
        return 1 if app.last_error else 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"MindArm: {error}", file=sys.stderr)
        if log is not None:
            log.write({"type": "startup_or_shutdown_error", "error": str(error)})
        return 1
    finally:
        if components is not None:
            components.close()


if __name__ == "__main__":
    raise SystemExit(main())
