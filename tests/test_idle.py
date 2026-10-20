import random
import shutil
from unittest.mock import Mock, patch

import numpy as np
import pytest

from evil_jarvis.idle import IdleSounds, build_idle_sounds, prepare_idle_clip
from evil_jarvis.persona import IdleSettings, load_persona
from evil_jarvis.tts import build_speaker
from evil_jarvis.tts.playback import PLAYBACK_RATE, AudioPlayer


class Clock:
    now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def playlist():
    clock = Clock()
    player = Mock(spec=AudioPlayer)
    player.play.side_effect = lambda *_: clock.advance(1)
    settings = IdleSettings(sounds=("a.wav", "b.wav", "c.wav"))
    clips = tuple(np.full(100, (i + 1) / 10, dtype=np.float32) for i in range(3))
    idle = IdleSounds(settings, clips, player, clock=clock, rng=random.Random(7), sleep=clock.advance)
    return idle, player, clock


def test_waits_for_quiet_period_then_random_interval_from_playback_end(playlist):
    idle, player, clock = playlist
    clock.now = 29.9
    assert idle.next_action() is None
    clock.now = 30
    action = idle.next_action()
    assert action is not None
    assert idle.next_action() is None  # a pending action is not queued twice
    action()
    assert player.play.call_count == 1
    assert idle._next_at - clock.now >= 45
    assert idle._next_at - clock.now <= 90
    clock.advance(44.9)
    assert idle.next_action() is None
    clock.advance(46)
    assert idle.next_action() is not None


def test_activity_restarts_quiet_period(playlist):
    idle, player, clock = playlist
    clock.now = 29
    idle.reset()
    clock.now = 30
    assert idle.next_action() is None
    clock.now = 59
    assert idle.next_action() is not None
    player.play.assert_not_called()


def test_idle_playlist_never_repeats_a_clip_consecutively(playlist):
    idle, player, clock = playlist
    for _ in range(15):
        clock.advance(200)
        idle.next_action()()
    selected = [float(call.args[0][0]) for call in player.play.call_args_list]
    assert len(set(selected)) == 3
    assert all(a != b for a, b in zip(selected, selected[1:], strict=False))
    assert all(call.args[1] == PLAYBACK_RATE for call in player.play.call_args_list)


def test_one_clip_persona_can_still_play_idle(playlist):
    _, player, clock = playlist
    idle = IdleSounds(
        IdleSettings(sounds=("only.wav",), after_seconds=0),
        (np.zeros(10, dtype=np.float32),), player, clock=clock, sleep=lambda _: None,
    )
    idle.next_action()()
    clock.advance(100)
    idle.next_action()()
    assert player.play.call_count == 2


def test_persona_without_idle_assets_has_no_scheduler(playlist):
    _, player, _ = playlist
    assert build_idle_sounds(load_persona("jarvis"), player, "clean") is None


def test_clean_idle_without_ffmpeg_respects_configured_gain():
    audio = load_persona("clown").idle_audio[0]
    with patch("evil_jarvis.idle.shutil.which", return_value=None):
        original = prepare_idle_clip(audio, "clean", 1)
        quiet = prepare_idle_clip(audio, "clean", 0.4)
    np.testing.assert_allclose(quiet, original * 0.4)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg unavailable")
def test_idle_processing_applies_effects_and_quiet_gain():
    audio = load_persona("clown").idle_audio[0]
    full = prepare_idle_clip(audio, "metallic", 1)
    quiet = prepare_idle_clip(audio, "metallic", 0.4)
    np.testing.assert_allclose(quiet, full * 0.4)
    assert np.isfinite(quiet).all()
    assert 0 < np.max(np.abs(quiet)) < 0.4


def test_reply_voices_accept_the_same_player_as_idle_sounds():
    player = Mock(spec=AudioPlayer)
    with patch("evil_jarvis.tts.PiperSpeaker") as piper:
        build_speaker("voice", voices_by_language={"en": "other"}, player=player)
    assert len(piper.call_args_list) == 2
    assert all(call.kwargs["player"] is player for call in piper.call_args_list)
