"""Construct protocol implementations from validated configuration."""

from dataclasses import dataclass

import yaml

from mindarm.arm.jetson import JetsonArm, SshTransport
from mindarm.arm.simulated import ArmSettings, SimulatedArm
from mindarm.config import Settings
from mindarm.decoder.blink import BlinkDecoder, BlinkDetector, TappedEEGSource
from mindarm.decoder.keyboard import KeyboardDecoder
from mindarm.eeg.fake import FakeEEGSource
from mindarm.eeg.replay import ReplayEEGSource
from mindarm.eeg.stream import StreamEEGSource
from mindarm.interfaces import ArmController, Decoder, EEGSource, Perception, Planner
from mindarm.perception.fake import FakePerception
from mindarm.planner.fake import FakePlanner
from mindarm.planner.nemotron import NemotronPlanner
from mindarm.session_log import SessionLog


@dataclass
class Components:
    """A swappable set of implementations exposed only through their contracts."""

    eeg: EEGSource
    decoder: Decoder
    perception: Perception
    planner: Planner
    arm: ArmController

    def close(self) -> None:
        """Release input adapter and arm link resources."""
        if isinstance(self.decoder, (KeyboardDecoder, BlinkDecoder)):
            self.decoder.close()
        if isinstance(self.arm, JetsonArm):
            self.arm.close()


def create_components(settings: Settings, log: SessionLog) -> Components:
    """Select each implementation; initialize keyboard only after fallible setup."""
    planner: Planner = (FakePlanner() if settings.planner == "fake" else NemotronPlanner(
        timeout=settings.planner_timeout, max_tokens=settings.planner_max_tokens, record=log.write,
        propose=settings.planner_propose, plan=settings.planner_plan))
    eeg: EEGSource
    if settings.eeg == "stream":
        eeg = StreamEEGSource(settings.stream_url, settings.sample_rate)
    elif settings.eeg == "replay":
        eeg = ReplayEEGSource(settings.replay_path, loop=settings.replay_loop)
    else:
        eeg = FakeEEGSource(settings.sample_rate, settings.n_channels, settings.frequency,
                            settings.noise_std, settings.alpha_bursts, settings.seed)
    perception = FakePerception(settings.scene_path)
    perception.detect()  # Fail before entering terminal input mode for an invalid fixture.
    arm: ArmController
    if settings.arm == "jetson":
        # Validate the configuration before the SSH session can energize anything.
        arm_settings = ArmSettings.model_validate(yaml.safe_load(settings.jetson_arm_path.read_text(encoding="utf-8")))
        arm = JetsonArm(arm_settings, SshTransport(settings.jetson_command))
    else:
        arm = SimulatedArm.from_file(settings.joint_limits_path)
    decoder: Decoder = KeyboardDecoder()
    if settings.decoder == "blink":
        # The gesture only selects; keys keep toggling and quitting, and the LLM never sees samples.
        detector = BlinkDetector(eeg.sample_rate, settings.blink_threshold_mv, settings.blink_min_gap_s,
                                 settings.blink_max_gap_s, settings.blink_refractory_s)
        decoder = BlinkDecoder(decoder, detector, record=log.write)
        eeg = TappedEEGSource(eeg, decoder.feed)
    return Components(eeg, decoder, perception, planner, arm)
